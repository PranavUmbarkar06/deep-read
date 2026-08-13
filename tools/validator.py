import os
import uuid
from typing import List, Dict
from pydantic import BaseModel, Field
import chromadb
from .azure_openai_client import generate_embedding, generate_json

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

DB_ROOT_DIR = os.path.join(os.getcwd(), "database", "vector_database")
os.makedirs(DB_ROOT_DIR, exist_ok=True)

class ScientificHypothesisValidator:
    def __init__(self, db_path: str = "default_db", collection_name: str = "papers_collection"):
        """
        Initializes persistent ChromaDB client and session history.
        """
        target_path = os.path.join(DB_ROOT_DIR, db_path)
        os.makedirs(target_path, exist_ok=True)
        self.db_path = target_path

        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.sessions: Dict[str, List[Dict[str, str]]] = {}

    def _get_embedding(self, text: str) -> List[float]:
        """Generates Azure OpenAI vector embeddings."""
        return generate_embedding(text)

    def _chunk_text(self, text: str, chunk_size: int = 1200, overlap: int = 250) -> List[str]:
        """Splits raw paper text into overlapping chunks."""
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
        Extracts PDF text, chunks it, and writes it to Vector database.
        Prevents duplicate indexing if paper is already in collection.
        """
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            return False

        # Deduplication check
        existing = self.collection.get(where={"source_title": paper_title}, limit=1)
        if existing and existing.get("ids"):
            print(f"Paper '{paper_title}' already indexed in vector database.")
            return True

        raw_text = extract_text_from_pdf(pdf_path)
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
        
        print(f"Indexed '{paper_title}': Stored {len(chunks)} text chunks.")
        return True

    def validate_hypothesis(self, session_id: str, hypothesis: str) -> HypothesisAnalysis:
        """
        Retrieves top matches, evaluates hypothesis against literature,
        and returns structured analysis.
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        context_block = ""
        count = self.collection.count()
        if count > 0:
            query_vector = self._get_embedding(hypothesis)
            search_results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=min(4, count)
            )
            
            retrieved_documents = search_results.get("documents", [[]])[0]
            retrieved_metadata = search_results.get("metadatas", [[]])[0]
            
            for i, (doc, meta) in enumerate(zip(retrieved_documents, retrieved_metadata)):
                source = meta.get("source_title", "Unknown Source")
                context_block += f"\n--- Baseline Paper: {source} (Chunk {meta.get('chunk_index')}) ---\n{doc}\n"
        else:
            context_block = "No PDF research papers currently indexed in session vector store."

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

        response_text = generate_json(
            user_prompt,
            system_instruction=(
                "You are an elite scientific peer reviewer. Your task is to evaluate the researcher's proposal "
                "against literature context. Identify logical flaws, edge cases, performance trade-offs, and refinements."
            ),
            schema=HypothesisAnalysis,
            schema_name="HypothesisAnalysis",
            temperature=0.2,
        )

        self.sessions[session_id].append({"role": "user", "content": hypothesis})
        self.sessions[session_id].append({"role": "assistant", "content": response_text})

        return HypothesisAnalysis.model_validate_json(response_text)

def format_analysis_markdown(analysis: HypothesisAnalysis) -> str:
    """Formats HypothesisAnalysis Pydantic model into structured Markdown text."""
    verdict_emoji = {
        "LOGICALLY_SOUND": "✅ **LOGICALLY SOUND**",
        "FLAWED": "⚠️ **FLAWED (Logical / Algorithmic Loophole Detected)**",
        "INCOMPATIBLE": "❌ **INCOMPATIBLE (Violates Baseline Mechanics)**",
        "NEED_CLARIFICATION": "❓ **NEEDS CLARIFICATION**"
    }.get(analysis.verdict, f"**{analysis.verdict}**")

    md = f"### Scientific Hypothesis Validation Analysis\n\n"
    md += f"**Overall Verdict**: {verdict_emoji}\n\n"
    md += f"#### 🧬 Baseline Paper Compatibility\n{analysis.compatibility_with_paper}\n\n"
    md += f"#### 📐 Structural & Logical Analysis\n{analysis.structural_analysis}\n\n"

    if analysis.potential_bottlenecks:
        md += f"#### ⚠️ Potential Bottlenecks & Edge Cases\n"
        for item in analysis.potential_bottlenecks:
            md += f"- {item}\n"
        md += "\n"

    if analysis.suggested_refinements:
        md += f"#### 💡 Suggested Refinements & Modifications\n"
        for item in analysis.suggested_refinements:
            md += f"- {item}\n"

    return md

    



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
