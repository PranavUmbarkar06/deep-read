# compare_papers.py
import json
import os
from pypdf import PdfReader
from .compatibility import verify_papers_compatibility
from .extract import extract_text_from_pdf
from .azure_openai_client import generate_text


def compare_research_papers(pdf_paths: list[str]) -> str:
    """
    Orchestrates the entire comparison pipeline. Checks compatibility first;
    if passed, ingests full text and returns a deep technical matrix.
    """
    # 1. Run the compatibility check
    
    
    # 3. Pull raw text locally only after confirmation (saves massive token costs)
    full_text_payload = ""
    for path in pdf_paths[:4]:
        full_text_payload += f"\n=========================================\n"
        full_text_payload += f"FULL DOCUMENT RAW TEXT: {path}\n"
        full_text_payload += f"=========================================\n"
        full_text_payload += extract_text_from_pdf(path) + "\n"

    # 4. Synthesize with Azure OpenAI.
    system_instruction = (
        "You are an expert technical reviewer and research lead. Your task is to generate "
        "a highly professional, objective Markdown comparison matrix comparing the provided papers.\n\n"
        "Matrix Rules:\n"
        "1. Do not use generic axes like 'Methodology' or 'Results'. Instead, dynamically deduce "
        "3-4 highly specific algorithmic/technical axes unique to this domain (e.g., 'Convergence Rate', "
        "'State-Space Complexity', 'Memory Bounds', 'Inference Latency Overhead').\n"
        "2. Break down each paper across its Core Theoretical Novelty, Key Mechanical Trade-off, "
        "and Verified Empirical Improvements.\n"
        "3. Provide brief, dense insights rather than vague summaries. Keep the language direct and high-impact."
    )

    print("Generating comprehensive comparison matrix via Azure OpenAI...")
    try:
        return generate_text(
            f"Execute a deep comparative analysis on these documents:\n\n{full_text_payload}",
            system_instruction=system_instruction,
            temperature=0.2,
        )

    except Exception as e:
        return f"Pipeline failed during matrix generation: {str(e)}"

# Example Usage Block
if __name__ == "__main__":
    # Add your test paths here (Max 4)
    path="../downloaded_papers/compare_test/"
    sample_papers = [f"../downloaded_papers/{path}/1.pdf", f"../downloaded_papers/{path}/2.pdf"] 
    
    output_matrix = compare_research_papers(sample_papers)
    print("\n--- Final Pipeline Output ---")
    print(output_matrix)
