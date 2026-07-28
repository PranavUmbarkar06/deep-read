# compatibility.py
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from pypdf import PdfReader
from .extract import extract_text_from_pdf
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize client (automatically detects GEMINI_API_KEY from env)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL=os.getenv("MODEL", "gemini-3.5-flash")  # Default to gemini-3.5-flash if not set
class CompatibilityResult(BaseModel):
    result: bool = Field(
        alias="compatible", # Maps internally to handle your required final key
        description="True if papers address the same core machine learning/algorithmic problem or baseline, making a direct comparison highly valuable. False otherwise."
    )
    reason: str = Field(
        description="A strict 1-2 sentence technical logging reason explaining why they are or are not comparable."
    )



def verify_papers_compatibility(pdf_paths: list[str]) -> str:
    """
    Main entry point: Extracts abstracts locally, sends them to Gemini 2.5 Flash,
    and returns a clean JSON string matching the required schema.
    """
    if len(pdf_paths) > 4:
        return json.dumps({
            "result": False,
            "reason": "Aborted: Pipeline constraint violated. Input exceeds maximum limit of 4 papers."
        })

    # 1. Cheap local text extraction
    abstracts_map = {}
    for path in pdf_paths:
        abstracts_map[path] = extract_text_from_pdf(path)
        
    # 2. Frame the payload for the model
    user_payload = "Evaluate the following research paper abstracts for deep domain compatibility:\n\n"
    for path, abstract in abstracts_map.items():
        user_payload += f"--- Paper File: {path} ---\n{abstract}\n\n"

    system_instruction = (
        "You are an elite research gatekeeper examining paper abstracts to determine if a technical comparison is viable. "
        "Focus heavily on the underlying algorithmic framework, mathematical structures, and optimization paradigms rather than the superficial application target.\n\n"
        "Compatibility Guidelines:\n"
        "1. Treat different applications (e.g., Robot Path Planning and Traveling Salesman Problem) as COMPATIBLE "
        "if they share the same foundational mathematical abstraction (e.g., discrete combinatorial optimization, graph traversal, continuous surface mapping).\n"
        "2. Mark as INCOMPATIBLE only if the methodologies cannot be cross-referenced at all (e.g., trying to compare an LLM pruning strategy with an image segmentation loss function).\n"
        "3. If papers utilize the same core metaheuristics (like GA, ACO, PSO) to optimize a objective function, they are legible for comparison."
    )

    try:
        response = client.models.generate_content(
            model=MODEL, # Low-cost, fast model for gatekeeping
            contents=user_payload,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=CompatibilityResult,
                temperature=0.1
            ),
        )
        
        # Parse against pydantic schema to validate structural integrity
        validated_data = CompatibilityResult.model_validate_json(response.text)
        
        # Return exact flat JSON structure requested
        return json.dumps({
            "result": validated_data.result,
            "reason": validated_data.reason
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "result": False,
            "reason": f"Pipeline failure during LLM gatecheck: {str(e)}"
        }, indent=2)
    


if __name__ == "__main__":
    # Example usage
    test_paths = ["../downloaded_papers/compare_test/1.pdf", "../downloaded_papers/compare_test/2.pdf"]
    compatibility_json = verify_papers_compatibility(test_paths)
    print("Compatibility Check Result:")
    print(compatibility_json)