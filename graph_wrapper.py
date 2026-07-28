from structures.agent_state import Paper,AgentState
from tools import find_papers,extract,critic,compatibility,compare_papers,summarize_comparison
from langchain_core.tools import tool

import os
from google import genai
from google.genai import types
from langgraph.types import interrupt


 
def orchestrator(state: AgentState) -> dict:
    """
    Phase 1: hardcoded to 'discover'. Real version (phase 2) will use an LLM call
    to classify intent + detect whether files are attached, per the router design
    from earlier in the conversation.
    """
    print(f"[orchestrator] routing query: {state['query']!r}")
    return {"intent": "summarize"}



 
def set_papers(state: AgentState) -> dict:
    """
    Generates keywords related to user query, searches those keywords on arxiv and fetches
    the best and most relevant results in the form of paper titles and their urls.
    """
    user_query = state.get("query", "")
    
    # 1. Obtain keywords & raw arXiv results via pure helpers
    keywords = find_papers.find_keywords(user_query)
    print(f"[set_papers] keywords: {keywords}")
    raw_arxiv_results = find_papers.fetch_papers(keywords, max_papers_per_keyword=1)
    print(f"[set_papers] raw arXiv results: {len(raw_arxiv_results)}")

    formatted_papers: list[Paper] = []
    candidate_titles: list[tuple[str, str]] = []

    # 2. Construct Paper instances and candidate tuples inside set_papers
    for result in raw_arxiv_results:
        paper_id = result.entry_id.split('/')[-1]
        clean_title = result.title.replace('\n', ' ')
        pdf_url = result.pdf_url or ""
        clean_summary = result.summary.replace('\n', ' ')

        # Instantiate Paper schema
        paper_obj: Paper = {
            "id": paper_id,
            "title": clean_title,
            "arxiv_id": paper_id,
            "url": pdf_url,
            "extracted": False,
            "summary": clean_summary,
            "attempts": 0
        }
        
        formatted_papers.append(paper_obj)
        candidate_titles.append((clean_title, pdf_url))

    # 3. Return updated dictionary slice to merge into state
    return {
        "papers": formatted_papers,
        "candidate_titles": candidate_titles
    }

 
def display_papers(state: AgentState) -> dict:
    """
    Formats the fetched papers into a user-facing message.
    Real version: this is where a nicer frontend rendering (cards, links) would
    hook in -- purely a display concern per the earlier "later, frontend thing" note.
    """
    lines = [f"Found {len(state['papers'])} papers for: {state['query']!r}\n"]
    for p in state["papers"]:
        lines.append(f"- {p['title']}  ({p['url']})")
    message = "\n".join(lines)
    print(f"[display_papers]\n{message}")
    return {"final_message": message}




# -------------------------------------------------------------
# File Check Routers & Nodes
# -------------------------------------------------------------
# -------------------------------------------------------------
# File Check Routers & Nodes
# -------------------------------------------------------------

 
def check_pdf_input(state: AgentState) -> str:
    """Routes execution based on the number of provided PDF paths in state.

    Args:
        state (AgentState): Current graph state containing 'pdf_paths'.

    Returns:
        str: Route directive ('ask_upload', 'ask_single_file', or 'process_file').
    """
    paths = state.get("pdf_paths", [])
    if len(paths) == 0:
        return "ask_upload"
    elif len(paths) > 1:
        return "ask_single_file"
    return "process_file"


 
def ask_upload_node(state: AgentState) -> dict:
    """Interrupts execution to request a PDF upload when none is present.

    Args:
        state (AgentState): Current graph state.

    Returns:
        dict: Partial state update containing the uploaded PDF path.
    """
    user_response = interrupt("No PDF detected. Please upload a PDF file to proceed.")
    return {"pdf_paths": [user_response.get("pdf_path")]}


 
def ask_single_file_node(state: AgentState) -> dict:
    """Interrupts execution to prompt the user for a single PDF path.

    Args:
        state (AgentState): Current graph state containing multiple file paths.

    Returns:
        dict: Partial state update containing the selected single PDF path.
    """
    user_response = interrupt("Multiple PDFs detected. Please specify or upload a single PDF file.")
    return {"pdf_paths": [user_response.get("pdf_path")]}


# -------------------------------------------------------------
# Extractor & Execution Nodes
# -------------------------------------------------------------

 
def extract_formatted_node(state: AgentState) -> dict:
    """Extracts raw text content from the target PDF file and initializes state parameters.

    Args:
        state (AgentState): Current graph state containing 'pdf_paths'.

    Returns:
        dict: Partial state update with extracted paper text, initial iteration count,
              max iterations, and an empty feedback list.
    """
    pdf_path = state["pdf_paths"][0]
    extracted_text = extract.extract_text_from_pdf(pdf_path)
    return {
        "paper_text": extracted_text,
        "iteration": 0,
        "max_iterations": state.get("max_iterations", 1),
        "feedback": []
    }


 
def summarise_paper_node(state: AgentState) -> dict:
    """Generates a structured research paper summary incorporating critic feedback.

    Args:
        state (AgentState): Current graph state containing 'paper_text', 'feedback', 
                            and 'iteration'.

    Returns:
        dict: Partial state update containing the generated summary markdown and 
              an incremented iteration counter.
    """
    paper_text = state["paper_text"]
    feedback = state.get("feedback", [])

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    prompt = f"""
    You are an elite research scientist reviewing an academic paper. 
    Your job is to read the following text and synthesize its true intent, methodology, and outcome.
    You are explaining this to both a scientist and a layperson, so clarity is paramount.

    --- START OF PAPER TEXT ---
    {paper_text}
    --- END OF PAPER TEXT ---
    Include a super short summary of the paper in 1-2 sentences at the top. 
    Provide the summary using this exact Markdown structure:

    ## Core Intent & Problem
    - What precise problem does this paper target?
    - Why do existing solutions fail (according to the authors)?

    ## Key Mechanism & Methodology
    - How does their proposed solution work step-by-step?
    - What datasets or evaluation setups did they use?

    ## Primary Results & Claims
    - What are the major quantitative metrics or takeaways? 
    - Keep this strictly factual based only on the text. Do not hallucinate numbers.

    ## Critical Limitations
    - What flaws, edge cases, or gaps do the authors acknowledge?

    Take this feedback into account when summarizing the paper: {feedback}
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
        )
    )
    return {
        "current_summary": response.text,
        "iteration": state.get("iteration", 0) + 1
    }


 
def summary_critic_node(state: AgentState) -> dict:
    """Evaluates the candidate summary against the source text to extract actionable feedback.

    Args:
        state (AgentState): Current graph state containing 'paper_text' and 'current_summary'.

    Returns:
        dict: Partial state update containing a list of feedback points.
    """
    eval_result = critic.evaluate_summary(state["paper_text"], state["current_summary"])
    return {"feedback": eval_result.get("feedback", [])}


 
def return_with_caveat_node(state: AgentState) -> dict:
    """Formats the final output message with an added warning when max iterations are exhausted.

    Args:
        state (AgentState): Current graph state containing 'current_summary', 'iteration', 
                            and 'feedback'.

    Returns:
        dict: Partial state update containing the flagged final message.
    """
    caveat_text = (
        f"{state['current_summary']}\n\n"
        f"> **Notice:** Generated summary reached the maximum iteration threshold ({state.get('iteration')} iterations) "
        f"with unresolved feedback: {', '.join(state.get('feedback', []))}"
    )
    return {"final_message": caveat_text}


 
def finalize_summary_node(state: AgentState) -> dict:
    """Commits the approved candidate summary to the final output state.

    Args:
        state (AgentState): Current graph state containing 'current_summary'.

    Returns:
        dict: Partial state update setting 'final_message' to 'current_summary'.
    """
    return {"final_message": state["current_summary"]}


# -------------------------------------------------------------
# Decision Routers
# -------------------------------------------------------------

 
def evaluate_critic_result(state: AgentState) -> str:
    """Determines whether to accept the summary, retry refinement, or terminate with a warning.

    Args:
        state (AgentState): Current graph state containing 'feedback', 'iteration', 
                            and 'max_iterations'.

    Returns:
        str: Routing decision ('pass', 'max_reached', or 'retry').
    """
    if not state.get("feedback") or len(state["feedback"]) == 0:
        return "pass"
    
    if state["iteration"] >= state.get("max_iterations", 1):
        return "max_reached"

    return "retry"



def compare_router(state: dict) -> str:
    pdf_paths = state.get("pdf_paths") or []
    if len(pdf_paths) <= 1:
        return "search"
    return "uploaded"

import json

def compatibility_node(state: AgentState) -> dict:
    """Nodes check compatibility and write result to state."""
    pdf_paths = state.get("pdf_paths") or []
    
    raw_res = compatibility.verify_papers_compatibility(pdf_paths)
    data = json.loads(raw_res)
    
    return {
        "is_compatible": data.get("result", False),
        "compatibility_reason": data.get("reason", "")
    }

def compare_papers_node(state: AgentState) -> dict:
    """Runs deep comparison matrix generation if compatible."""
    pdf_paths = state.get("pdf_paths") or []
    
    # Calls your compare_research_papers function
    matrix_text = compare_papers.compare_research_papers(pdf_paths)
    
    return {
        "comparison_analysis": matrix_text
    }

def summarize_comparison_node(state: AgentState) -> dict:
    """Summarizes the generated comparison matrix."""
    matrix_text = state.get("comparison_analysis") or ""
    
    summary_json_str = summarize_comparison.summarize_comparison_matrix(matrix_text)
    
    return {
        "comparison_summary": summary_json_str,
        "final_message": summary_json_str
    }


#validate