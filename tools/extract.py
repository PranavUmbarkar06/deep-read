import os
from pypdf import PdfReader
import logger
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file structured by page.
    Params - pdf_path: str - Path to the PDF file."""

    
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    
    # Strict boundary check
    if total_pages > 25:
        return f"Paper is too long, can't summarize this. Max limit is 25 pages (This paper has {total_pages} pages)."

    # 2. Extract full text since it falls within the safe limit
    extracted_text = []
    for i in range(total_pages):
        page_text = reader.pages[i].extract_text()
        if page_text:
            extracted_text.append(f"--- PAGE {i+1} ---\n{page_text}")
            
    paper_text = "\n\n".join(extracted_text)
    logger.log("Extracted text from PDF", f"PDF Path: '{pdf_path}', Extracted Text Length: {len(paper_text)} characters")
    return paper_text