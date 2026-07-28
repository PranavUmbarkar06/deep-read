import json
from pydantic import BaseModel, Field
from google.genai import types
from google import genai
import os

from dotenv import load_dotenv

load_dotenv()

class ComparisonSummaryResult(BaseModel):
    executive_summary: str = Field(
        description="A concise 2-3 sentence overarching takeaway summarizing the primary architectural/algorithmic differences."
    )
    key_tradeoffs: list[str] = Field(
        description="A list of 3-5 core technical or mechanical trade-offs identified across the compared papers."
    )
    selection_guide: str = Field(
        description="Clear decision guidance on when to pick which paper/approach based on specific real-world constraints (e.g., latency, dataset size, memory)."
    )

def summarize_comparison_matrix(comparison_matrix_text: str) -> str:
    """
    Ingests the generated Markdown comparison matrix text and returns a clean, 
    structured executive summary JSON string.
    """
    client=genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    # Guard against invalid or error inputs
    if not comparison_matrix_text or "Pipeline failed" in comparison_matrix_text:
        return json.dumps({
            "error": "Cannot summarize: Invalid or missing comparison matrix text."
        }, indent=2)

    system_instruction = (
        "You are a lead AI researcher summarizing a complex paper comparison matrix for executive leadership. "
        "Extract the core technical trade-offs and turn the detailed matrix into actionable, high-impact key takeaways. "
        "Keep the language direct, precise, and completely free of conversational filler."
    )

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=f"Summarize this detailed research paper comparison matrix:\n\n{comparison_matrix_text}",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=ComparisonSummaryResult,
                temperature=0.1
            ),
        )

        # Validate structured output against Pydantic model
        validated_summary = ComparisonSummaryResult.model_validate_json(response.text)

        return json.dumps(validated_summary.model_dump(), indent=2)

    except Exception as e:
        return json.dumps({
            "error": f"Failed to summarize comparison matrix: {str(e)}"
        }, indent=2)