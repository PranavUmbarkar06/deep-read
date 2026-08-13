import json
import arxiv
from typing import List, Dict
import os
from dotenv import load_dotenv
from .azure_openai_client import generate_json

load_dotenv()

def find_keywords(user_query: str) -> list[str]:
    """
    Pure utility: Generates search keywords from a user query string.
    """
    prompt = f"""
        The user is looking for academic research papers related to: "{user_query}"
        
        Generate a JSON list containing 10 to 15 precise academic keywords, subtopics, 
        or search phrases suitable for querying arXiv.
        
        Return ONLY a JSON object like this:
        "keywords": ["keyword 1", "keyword 2", "keyword 3"]
    """

    response_text = generate_json(
        prompt,
        schema={
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["keywords"],
            "additionalProperties": False,
        },
        schema_name="ArxivKeywords",
        temperature=0.1,
    )

    try:
        payload = json.loads(response_text)
        keywords = payload.get("keywords", payload)
        if isinstance(keywords, list):
            if user_query not in keywords:
                keywords.insert(0, user_query)
            return keywords
    except json.JSONDecodeError:
        pass
    
    return [user_query]


def fetch_papers(keywords: list[str], max_papers_per_keyword: int = 1) -> list[arxiv.Result]:
    """
    Pure utility: Fetches raw arXiv Result objects for given keywords without 
    coupling to any application-level state or custom types.
    """
    arxiv_client = arxiv.Client()
    seen_ids = set()
    raw_results: list[arxiv.Result] = []

    for kw in keywords:
        clean_kw = kw.replace('"', '').strip()
        if not clean_kw:
            continue
        search = arxiv.Search(
            query=clean_kw,
            max_results=max_papers_per_keyword,
            sort_by=arxiv.SortCriterion.Relevance
        )

        try:
            results = list(arxiv_client.results(search))
            for res in results:
                paper_id = res.entry_id.split('/')[-1]
                if paper_id not in seen_ids:
                    seen_ids.add(paper_id)
                    raw_results.append(res)
        except Exception as e:
            print(f"Error searching arXiv for keyword '{clean_kw}': {e}")

    return raw_results[:5]


import urllib.request

def download_paper_pdf(result: arxiv.Result, save_dir: str) -> str:
    """
    Downloads the PDF for an arxiv.Result object into save_dir and returns the local file path.
    """
    os.makedirs(save_dir, exist_ok=True)
    paper_id = result.entry_id.split('/')[-1].replace('/', '_')
    filename = f"{paper_id}.pdf"
    file_path = os.path.join(save_dir, filename)
    
    if os.path.exists(file_path):
        return file_path
        
    try:
        url = result.pdf_url or f"https://arxiv.org/pdf/{paper_id}.pdf"
        if not url.endswith(".pdf") and "arxiv.org/pdf" in url:
            url += ".pdf"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
            out_file.write(response.read())
        return file_path
    except Exception as e:
        print(f"Error downloading PDF for paper {result.entry_id}: {e}")
        return ""




if __name__ == "__main__":
    query = "How do we make Large Language Models more efficient for edge devices?"
    
    # Step 1: Get 10-15 keywords from Azure OpenAI
    print("Generating keywords...")
    keywords = find_keywords(query)
    print(f"\nGenerated {len(keywords)} Keywords:")
    for i, kw in enumerate(keywords, 1):
        print(f"  {i}. {kw}")
    
    # Step 2: Fetch papers from arXiv using those keywords
    print("\nFetching papers from arXiv...")
    papers = fetch_papers(keywords, max_papers_per_keyword=1)
    
    # Display results
    print(f"\nRetrieved {len(papers)} unique real papers:\n" + "="*50)
    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper['title']}")
        print(f"   Matched Term: {paper['matched_keyword']}")
        print(f"   PDF Link:     {paper['pdf_url']}\n")
