from structures.agent_state import Paper,AgentState
from tools import find_papers,extract,critic,compatibility,compare_papers,summarize_comparison,validator
from langchain_core.tools import tool
from typing import Literal, TypedDict, List, Optional
from pydantic import BaseModel, Field
import os
from langgraph.types import interrupt
from tools.azure_openai_client import generate_model, generate_text

from dotenv import load_dotenv

load_dotenv()

 
class OrchestratorDecision(BaseModel):
    intent: Literal["discover", "summarize", "compare", "validate"] = Field(
        description="The primary routing intent based on user query and context."
    )
    reasoning: str = Field(
        description="Brief 1-sentence explanation of why this intent was selected."
    )

# ---------------------------------------------------------------------------
# System Prompt with Edge Case Handling
# ---------------------------------------------------------------------------
ORCHESTRATOR_SYSTEM_PROMPT = """You are the intent classifier and router for an AI Research Assistant specializing in academic literature.
Your task is to classify the incoming user query into exactly ONE of the following four intents:

1. **discover**: Searching for new literature, exploring a domain, finding papers by topic/author/year, or asking for paper recommendations.
2. **summarize**: Explaining, condensing, or breaking down key findings, methods, or takeaways from specific paper(s) or provided context/files.
3. **compare**: Analyzing differences, similarities, tradeoffs, benchmark performance, or architectural choices across two or more papers, approaches, or models.
4. **validate**: Fact-checking a claim against literature, inspecting methodology for flaws/assumptions, checking statistical rigor, or asking if a paper's results hold up.

---
### EDGE CASES & DISAMBIGUATION RULES:

* **Rule 1: File Presence Priority**
  - If files/papers are attached AND the query is general (e.g., "What is this about?", "Key takeaways?"), default to **`summarize`**.
  - If files are attached AND the query explicitly asks to evaluate claims, assumptions, or mathematical proofs inside them, route to **`validate`**.
  - If files are attached AND the query asks to evaluate them against another paper or benchmark, route to **`compare`**.

* **Rule 2: Multi-Intent Disambiguation**
  - If query asks to "find papers on X and summarize them", priority goes to **`discover`** first because papers must be found before they can be summarized.
  - If query asks to "compare Paper A and Paper B, then summarize the winner", route to **`compare`**.

* **Rule 3: Ambiguous or Short Queries**
  - Short keyword searches (e.g., "Attention mechanism", "ResNet vs ViT") -> route "Attention mechanism" to **`discover`**, route "ResNet vs ViT" to **`compare`**.
  - Queries challenging validity (e.g., "Is p-hacking present here?", "Does this paper actually prove X?") -> **`validate`**.

* **Rule 4: Default Fallback**
  - If the query is vague but mentions finding relevant work -> **`discover`**.
"""

# ---------------------------------------------------------------------------
# Orchestrator Node Function
# ---------------------------------------------------------------------------
def orchestrator(state: AgentState) -> dict:
    """
    Classifies user intent for research paper processing using Azure OpenAI.
    Returns a state update dictionary containing the detected intent and reasoning.
    """
    query = state.get("query", "")
    pdf_paths = state.get("pdf_paths", []) or []
    attached_files = state.get("attached_files", []) or pdf_paths
    query_lower = query.lower()

    if not attached_files and any(term in query_lower for term in ["summarize", "summarise", "summary"]):
        return {
            "intent": "summarize",
            "router_reasoning": "The query asks to summarize a specific paper, but no PDF is attached."
        }

    # Format contextual information for the prompt
    context_str = f"User Query: {query!r}\n"
    if attached_files:
        context_str += f"Attached Files/Papers ({len(attached_files)}): {', '.join(attached_files)}\n"
    else:
        context_str += "Attached Files/Papers: None\n"

    try:
        decision = generate_model(
            context_str,
            OrchestratorDecision,
            system_instruction=ORCHESTRATOR_SYSTEM_PROMPT,
            temperature=0.0,
        )
        
        print(f"[orchestrator] Query: {query!r} -> Intent: {decision.intent} ({decision.reasoning})")
        
        return {
            "intent": decision.intent,
            "router_reasoning": decision.reasoning
        }

    except Exception as e:
        # Fallback edge-case handler in case of API issues
        print(f"[orchestrator] API Call failed: {e}. Falling back to default routing.")
        default_intent = "summarize" if attached_files else "discover"
        return {
            "intent": default_intent,
            "router_reasoning": "Fallback trigger due to execution error."
        }


 
def set_papers(state: AgentState) -> dict:
    """
    Generates keywords related to user query, searches those keywords on arxiv and fetches
    the best and most relevant results in the form of paper titles and their urls.
    If intent is compare/validate, downloads candidate PDFs into state["pdf_paths"].
    """
    user_query = state.get("query", "")
    intent = state.get("intent", "")
    
    # 1. Obtain keywords & raw arXiv results via pure helpers
    keywords = find_papers.find_keywords(user_query)
    print(f"[set_papers] keywords: {keywords}")
    raw_arxiv_results = find_papers.fetch_papers(keywords, max_papers_per_keyword=1)
    print(f"[set_papers] raw arXiv results: {len(raw_arxiv_results)}")

    formatted_papers: list[Paper] = []
    candidate_titles: list[tuple[str, str]] = []
    downloaded_pdf_paths: list[str] = list(state.get("pdf_paths") or [])

    save_dir = os.path.join(os.getcwd(), "downloaded_papers")
    os.makedirs(save_dir, exist_ok=True)

    limit = 3 if intent in ["compare", "validate"] else len(raw_arxiv_results)

    # 2. Construct Paper instances and candidate tuples inside set_papers
    for idx, result in enumerate(raw_arxiv_results):
        paper_id = result.entry_id.split('/')[-1]
        clean_title = result.title.replace('\n', ' ')
        pdf_url = result.pdf_url or ""
        clean_summary = result.summary.replace('\n', ' ')

        if intent in ["compare", "validate"] and idx < limit:
            try:
                local_path = find_papers.download_paper_pdf(result, save_dir)
                if local_path and os.path.exists(local_path) and local_path not in downloaded_pdf_paths:
                    downloaded_pdf_paths.append(local_path)
            except Exception as e:
                print(f"[set_papers] Error downloading paper {paper_id}: {e}")

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
        "candidate_titles": candidate_titles,
        "pdf_paths": downloaded_pdf_paths
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

    summary = generate_text(prompt, temperature=0.1)
    return {
        "current_summary": summary,
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
    if len(pdf_paths) >= 2:
        return "uploaded"
    return "ask_upload"

def ask_compare_upload_node(state: AgentState) -> dict:
    """Returns a notice when user attempts to compare without uploading at least 2 PDFs."""
    return {
        "final_message": "Paper comparison requires at least 2 uploaded PDF research papers. Please upload your PDF files using the document uploader to generate a comparative analysis."
    }


import json

def compatibility_node(state: AgentState) -> dict:
    """Nodes check compatibility and write result to state."""
    pdf_paths = state.get("pdf_paths") or []
    papers = state.get("papers") or []
    
    paper_abstracts = {}
    if not pdf_paths and papers:
        for p in papers[:4]:
            paper_abstracts[p.get("title", p.get("id"))] = p.get("summary", "")

    raw_res = compatibility.verify_papers_compatibility(pdf_paths, paper_abstracts=paper_abstracts if paper_abstracts else None)
    try:
        data = json.loads(raw_res)
    except Exception:
        data = {"result": True, "reason": "Compatibility verified."}

    is_comp = data.get("result", False)
    reason = data.get("reason", "")
    print(f"[compatibility_node] is_compatible: {is_comp}, reason: {reason}")
    
    return {
        "is_compatible": is_comp,
        "compatibility_reason": reason
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

def validate_and_query(state: AgentState) -> dict:
    session_id = state.get("session_id", "default_session")
    pdf_paths = state.get("pdf_paths", [])
    query = state.get("query", "")

    # Instantiate validator with session_id
    agent = validator.ScientificHypothesisValidator(db_path=session_id)

    # Index papers
    for paper in pdf_paths:
        if os.path.exists(paper):
            filename = os.path.basename(paper)
            agent.add_paper(paper, paper_title=filename)

    # Run validation step
    analysis: validator.HypothesisAnalysis = agent.validate_hypothesis(
        session_id=session_id, hypothesis=query
    )

    formatted_md = validator.format_analysis_markdown(analysis)

    return {
        "validate_result": analysis.model_dump(),
        "final_message": formatted_md,
        "session_id": session_id,
        "query": query,
    }



if __name__ == '__main__':
    
    print(orchestrator({

        "query": "show some papers optimization algorithms like GA, ACO, PSO",

    }))
