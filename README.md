optimisations
- compare_papers - orignally computed summary for every paper, then compared summary so that is 5 expensive and redundant llms calls. now send raw text of papers directly into Azure OpenAI and gets commparison.
- added logging to access previously generated llm call to reuse in the case of a two step llm call 

problems
- how to implement validate_hypothesis recallable 
- what architecture to create with the tools i have made?????


find_papers: Searches and retrieves relevant academic papers based on input topics, keywords, or queries.

fetch_papers: Downloads or fetches full-text PDFs/documents corresponding to the identified research papers.

extract: Systematically extracts structured content (e.g., abstract, methodology, results, conclusions) from raw paper text.

summarise_paper: Generates comprehensive summaries of individual papers based on extracted content and feedback loops.

critic: Evaluates generated summaries against quality benchmarks, providing constructive feedback for summarise_paper to iterate on.

compatibility: Assesses whether candidate papers share common themes, methodologies, or structural baselines to determine if they can be meaningfully compared.

compare_papers: Performs detailed comparative analyses across multiple compatible papers, highlighting similarities, differences, and key trade-offs.

validator: Validates research hypotheses and answers contextual queries using a Retrieval-Augmented Generation (RAG) framework over the repository.

logger: Handles system-wide logging to record agent operations, execution traces, and errors (logs.txt).
