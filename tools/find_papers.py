import json
import arxiv
from typing import List, Dict
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
# Initialize the Gemini API client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def find_keywords(user_query: str) -> list[str]:
    """
    Pure utility: Generates search keywords from a user query string.
    """
    prompt = f"""
    The user is looking for academic research papers related to: "{user_query}"
    
    Generate a JSON list containing 10 to 15 precise academic keywords, subtopics, 
    or search phrases suitable for querying arXiv.
    
    Return ONLY a JSON array of strings, like this:
    ["keyword 1", "keyword 2", "keyword 3"]
    """

    response = client.models.generate_content(
        model='gemini-3.5-flash',
        contents=prompt,
        config={'response_mime_type': 'application/json'}
    )

    try:
        keywords = json.loads(response.text)
        if isinstance(keywords, list):
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
        search_query = f'all:"{kw}"'
        search = arxiv.Search(
            query=search_query,
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
            print(f"Error searching arXiv for keyword '{kw}': {e}")

    return raw_results


if __name__ == "__main__":
    query = "How do we make Large Language Models more efficient for edge devices?"
    
    # Step 1: Get 10-15 keywords from Gemini
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