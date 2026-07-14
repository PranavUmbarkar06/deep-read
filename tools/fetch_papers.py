import os
from urllib.request import urlretrieve
import arxiv
import re
from datetime import datetime
import logger
def fetch_papers(query):
    """
    Searches for academic papers on arXiv based on a query string and downloads 
    the top 5 matching PDFs into a sanitized, query-specific directory.

    The function creates a local directory named after the query, strips out 
    invalid file-system characters from both the folder name and the paper titles, 
    and uses the arXiv API and standard web utilities to fetch and save the files.

    Args:
        query (str): The search term or logical query string used to match 
            papers on the arXiv database (e.g., "swarm optimisation").

    Returns:
        list: A python list of the titles of the downloaded papers.
    """
    clean_folder_name = re.sub(r'[^a-zA-Z0-9 \-_]', '', query)

# Replace multiple spaces or messy gaps with a single underscore
    clean_folder_name = re.sub(r'\s+', '_', clean_folder_name).strip('_')
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=5,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    os.makedirs(f"../downloaded_papers/{clean_folder_name}", exist_ok=True)
    
    print("Starting downloads...")
    papers=[]
    for result in client.results(search):
        # Sanitize the title so it's a valid filename
        clean_title = "".join(c for c in result.title if c.isalnum() or c in (" ", "_", "-")).rstrip()
        filename = f"../downloaded_papers/{clean_folder_name}/{clean_title}.pdf"
        papers.append(result.title)
        print(f"Downloading: {result.title}")
        
        #Directly download the PDF link using standard library utilities
        urlretrieve(result.pdf_url, filename)

    print("All downloads completed successfully.")
    logger.log("Fetched papers", f"Query: '{query}', Papers: {papers}")
    return papers


if __name__ == "__main__":
    
    print(fetch_papers("swarm optimisation"))