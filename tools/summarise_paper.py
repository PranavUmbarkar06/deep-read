import os
from pypdf import PdfReader
from google import genai
from google.genai import types
from dotenv import load_dotenv
from extract import extract_text_from_pdf
import logger
from critic import evaluate_summary

load_dotenv()
MODEL=os.getenv("MODEL", "gemini-2.5-flash")  # Default to gemini-2.5-flash if not set
def summarise(pdf_path: str,feedback: list=None,max_iterations:int=1) -> str:

    """
    Validates paper length and generates a structured summary.
    Rejects papers outright if they exceed 25 pages.
    Params - pdf_path: str - Path to the PDF file to summarize. 
    feedback: list - List of feedback items to sent by critic 
    Returns - str - Structured summary or error message.
    """
    # 1. Instantly check the metadata for page count
    
    
    paper_text=extract_text_from_pdf(pdf_path)

    # 3. Initialize the Google GenAI client
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    feedback=[]
    # 4. Prompt optimized for extracting core scientific intent
    while max_iterations>=0:
        max_iterations-=1
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

        # 5. Execute the call using gemini-2.5-flash
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,  # Keeps the model strictly factual
            )
        )
        print("Summary generated successfully.")
        print(response.text)
        logger.log("Summarised paper", f"PDF Path: '{pdf_path}', Summary : {response.text}")
        if(max_iterations>0):
            feedback=evaluate_summary(paper_text,response.text)['feedback']
    return response.text

if __name__ == "__main__":
    pdf_path = "../downloaded_papers/test.pdf"  # Replace with your PDF path
    # Example usage:
    result = summarise("../downloaded_papers/autmotive_nvh_technology/nvh.pdf")
    print(result)
    # reader = PdfReader(pdf_path)
    # total_pages = len(reader.pages)
    
    # # Strict boundary check
    # if total_pages > 25:
    #     print(f"Paper is too long, can't summarize this. Max limit is 25 pages (This paper has {total_pages} pages).")

    # # 2. Extract full text since it falls within the safe limit
    # extracted_text = []
    # for i in range(total_pages):
    #     page_text = reader.pages[i].extract_text()
    #     if page_text:
    #         extracted_text.append(f"--- PAGE {i+1} ---\n{page_text}")
            
    # paper_text = "\n\n".join(extracted_text)
    
    # if not paper_text.strip():
    #     print("Error: Could not extract any readable text from the PDF.")
        

    
    