# compatibility.py
import json
from pydantic import BaseModel, Field, ConfigDict
from .extract import extract_text_from_pdf
from .azure_openai_client import generate_json
import os
from dotenv import load_dotenv
load_dotenv()

class CompatibilityResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    result: bool = Field(
        alias="compatible", # Maps internally to handle required final key
        default=True,
        description="True if papers address the same core machine learning/algorithmic problem or baseline, making a direct comparison highly valuable. False otherwise."
    )
    reason: str = Field(
        description="A strict 1-2 sentence technical logging reason explaining why they are or are not comparable."
    )


def verify_papers_compatibility(pdf_paths: list[str], paper_abstracts: dict[str, str] = None) -> str:
    """
    Main entry point: Extracts abstracts locally or uses paper_abstracts, sends them to Azure OpenAI,
    and returns a clean JSON string matching the required schema.
    """
    if len(pdf_paths) > 4:
        return json.dumps({
            "result": False,
            "reason": "Aborted: Pipeline constraint violated. Input exceeds maximum limit of 4 papers."
        })

    # 1. Text / abstract extraction
    abstracts_map = {}
    if paper_abstracts:
        abstracts_map = paper_abstracts
    else:
        for path in pdf_paths:
            if os.path.exists(path):
                try:
                    text = extract_text_from_pdf(path)
                    abstracts_map[path] = text[:3000] # Take first 3000 chars as abstract/intro
                except Exception as e:
                    abstracts_map[path] = f"Sample text extract unavailable: {e}"
            else:
                abstracts_map[path] = f"Paper reference: {path}"

    if not abstracts_map:
        return json.dumps({
            "result": False,
            "reason": "No papers provided for compatibility check."
        })
        
    # 2. Frame the payload for the model
    user_payload = "Evaluate the following research paper abstracts for deep domain compatibility:\n\n"
    for path, abstract in abstracts_map.items():
        user_payload += f"--- Paper File / Title: {path} ---\n{abstract}\n\n"

    system_instruction = (
        "You are an elite research gatekeeper examining paper abstracts to determine if a technical comparison is viable. "
        "Focus heavily on the underlying algorithmic framework, mathematical structures, and optimization paradigms rather than the superficial application target.\n\n"
        "Compatibility Guidelines:\n"
        "1. Treat different applications (e.g., Robot Path Planning and Traveling Salesman Problem) as COMPATIBLE "
        "if they share the same foundational mathematical abstraction (e.g., discrete combinatorial optimization, graph traversal, continuous surface mapping).\n"
        "2. Mark as INCOMPATIBLE only if the methodologies cannot be cross-referenced at all (e.g., trying to compare an LLM pruning strategy with an image segmentation loss function).\n"
        "3. If papers utilize the same core metaheuristics (like GA, ACO, PSO) or machine learning techniques to optimize an objective function, they are legible for comparison."
    )

    try:
        response_text = generate_json(
            user_payload,
            system_instruction=system_instruction,
            schema=CompatibilityResult,
            schema_name="CompatibilityResult",
            temperature=0.1,
        )
        
        # Parse against pydantic schema to validate structural integrity
        try:
            validated_data = CompatibilityResult.model_validate_json(response_text)
            res_val = validated_data.result
            reason_val = validated_data.reason
        except Exception:
            parsed_raw = json.loads(response_text)
            res_val = parsed_raw.get("result", parsed_raw.get("compatible", True))
            reason_val = parsed_raw.get("reason", "Compatibility check completed.")

        return json.dumps({
            "result": bool(res_val),
            "reason": str(reason_val)
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
