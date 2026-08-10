import os
import uuid
from typing import List, Dict
from pydantic import BaseModel, Field
import chromadb
from google import genai
from google.genai import types

# Import your extraction function
from .extract import extract_text_from_pdf
from dotenv import load_dotenv
load_dotenv()
# 1. Define the rigorous evaluation schema
class HypothesisAnalysis(BaseModel):
    verdict: str = Field(
        description="Must be one of: LOGICALLY_SOUND (the extension works), FLAWED (logical/algorithmic loophole found), INCOMPATIBLE (violates paper's rules), or NEED_CLARIFICATION."
    )
    compatibility_with_paper: str = Field(
        description="How the researcher's theory relates to the baseline paper's mechanics (e.g., optimization, alternative approach)."
    )
    structural_analysis: str = Field(
        description="Detailed step-by-step logical evaluation of the researcher's proposal using computer science/scientific principles."
    )
    potential_bottlenecks: List[str] = Field(
        description="What could go wrong? Edge cases, increased complexity, physical limitations, or mathematical loopholes."
    )
    suggested_refinements: List[str] = Field(
        description="Actionable advice to improve or patch the researcher's proposed idea."
    )

ABSOLUTE_PATH="../database/vector_database/"
class ScientificHypothesisValidator:
    def __init__(self, db_path: str=f"NewDB", collection_name: str = "papers_collection"):
        """
        Initializes the Gemini Client, persistent ChromaDB client, and session history.
        """
        # Initialize Gemini Client (requires GEMINI_API_KEY env variable)
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Initialize Persistent Vector Database
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.db_path=ABSOLUTE_PATH+db_path
        
        # Session state: session_id -> list of message dictionaries
        self.sessions: Dict[str, List[Dict[str, str]]] = {}
        

    def _get_embedding(self, text: str) -> List[float]:
        """Generates native vector embeddings using gemini-embedding-001."""
        response = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=text
        )
        if isinstance(response.embeddings, list):
            return response.embeddings[0].values
        return response.embedding.values

    def _chunk_text(self, text: str, chunk_size: int = 1200, overlap: int = 250) -> List[str]:
        """Splits raw paper text into overlapping chunks while preserving page headers."""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunks.append(text[start:end])
            if end == text_len:
                break
            start += chunk_size - overlap
        return chunks

    def add_paper(self, pdf_path: str, paper_title: str) -> bool:
        """
        Extracts PDF text, chunks it, and writes it to the Vector database.
        Returns True if successful, False if extraction failed/limit hit.
        """
        raw_text = extract_text_from_pdf(pdf_path)
        
        # Check if the extraction utility returned the error message
        if raw_text.startswith("Paper is too long"):
            print(f"Error indexing '{paper_title}': {raw_text}")
            return False

        chunks = self._chunk_text(raw_text)
        
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{uuid.uuid4()}_chunk_{idx}"
            embedding = self._get_embedding(chunk)
            
            ids.append(chunk_id)
            embeddings.append(embedding)
            documents.append(chunk)
            metadatas.append({
                "source_title": paper_title,
                "chunk_index": idx
            })
            
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        
        print(f"Indexed '{paper_title}': Successfully stored {len(chunks)} text chunks.")
        return True

    def validate_hypothesis(self, session_id: str, hypothesis: str) -> HypothesisAnalysis:
        """
        Retrieves top-4 matches, evaluates the researcher's theory contextually,
        updates the session's conversational history, and returns structured JSON analysis.
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        # 1. Retrieve the top 4 most relevant chunks
        query_vector = self._get_embedding(hypothesis)
        search_results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=4
        )
        
        retrieved_documents = search_results.get("documents", [[]])[0]
        retrieved_metadata = search_results.get("metadatas", [[]])[0]
        
        context_block = ""
        for i, (doc, meta) in enumerate(zip(retrieved_documents, retrieved_metadata)):
            source = meta.get("source_title", "Unknown Source")
            context_block += f"\n--- Baseline Paper: {source} (Chunk {meta.get('chunk_index')}) ---\n{doc}\n"

        # 2. Re-compile previous conversation history
        history_context = ""
        if self.sessions[session_id]:
            history_context = "### Previous Conversation History:\n"
            for msg in self.sessions[session_id]:
                history_context += f"{msg['role'].upper()}: {msg['content']}\n"
            history_context += "\n"

        user_prompt = (
            f"{history_context}"
            f"### Baseline Paper Context:\n{context_block}\n\n"
            f"### Researcher's Proposed Theory / Doubt:\n{hypothesis}"
        )

        # 3. Request analytical validation from Gemini 3.5 Flash
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[user_prompt],
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are an elite, highly pragmatic scientific peer reviewer and collaborative research partner. "
                    "The researcher is proposing a new theory, optimization, or extension built on top of the baseline paper.\n\n"
                    "Your job is NOT to look for direct keyword matches. Instead, use your deep logical, mathematical, "
                    "and structural reasoning to test the soundness of the researcher's proposal. "
                    "Identify structural logic flaws, point out edge cases, analyze performance/complexity impacts, "
                    "and suggest modifications to help patch their theory."
                ),
                response_mime_type="application/json",
                response_schema=HypothesisAnalysis,
                temperature=0.2  # Keep deterministic but allow logical deductive reasoning
            )
        )

        # Update Session History
        self.sessions[session_id].append({"role": "user", "content": hypothesis})
        self.sessions[session_id].append({"role": "assistant", "content": response.text})

        return HypothesisAnalysis.model_validate_json(response.text)
    



if __name__ == "__main__":
    # Example usage
    validator = ScientificHypothesisValidator()
    
    # Add a paper to the database
    pdf_path = "../downloaded_papers/validate_test/bitcoin.pdf"
    paper_title = "Bitcoin: A Peer-to-Peer Electronic Cash System"
    if validator.add_paper(pdf_path, paper_title):
        print(f"Paper '{paper_title}' added successfully.")
    
    # Validate a hypothesis
    session_id = str(uuid.uuid4())
    hypothesis = """In the paper's design, the block generation time is regulated to be roughly 10 minutes to allow transactions to propagate globally and prevent split-brain forks. 

I propose a modification to drastically increase transaction speeds: we should dynamically reduce the Block Target Time (difficulty adjustment) from 10 minutes down to 2 seconds. Since cryptography remains secure and the Proof-of-Work hash requirement is still strictly calculated, this will increase the transaction throughput by 300x while maintaining the absolute security of the ledger."""
    analysis = validator.validate_hypothesis(session_id, hypothesis)
    
    print("Hypothesis Analysis:")
    print(analysis.model_dump_json(indent=4))