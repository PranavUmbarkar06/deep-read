
from typing import Literal, Optional, List, Tuple
from typing_extensions import TypedDict

class Paper(TypedDict):
    id: str                 # stable id (arxiv id once we're using the real fetch tool)
    title: str
    arxiv_id: str | None
    url: str | None
    extracted: bool          # has extract_formatted run on this paper yet? (phase 3+)
    summary: str | None      # phase 2+
    attempts: int  





class AgentState(TypedDict, total=False):
    session_id: str 
    query: str
    intent: Literal["discover", "summarize", "compare", "validate"]
    candidate_titles: List[Tuple[str, str]]  # list of (title, url) tuples
    papers: List[Paper]                       # output of fetch_papers
    final_message: str
    
    # --- New fields for the summary execution loop ---
    pdf_paths: List[str]                      # uploaded paper path(s)
    paper_text: Optional[str]                 # extracted text from format_extractor
    current_summary: Optional[str]            # generated candidate summary
    feedback: List[str]                       # feedback provided by critic
    iteration: int                            # current feedback iteration counter
    max_iterations: int                      # loop threshold

    is_compatible: bool
    compatibility_reason: str
    comparison_analysis: Optional[str]  # Detailed Markdown comparison matrix
    comparison_summary: Optional[str]
