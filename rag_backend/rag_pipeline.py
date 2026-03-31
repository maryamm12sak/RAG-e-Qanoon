
# ─────────────────────────────────────────────
# 1. INSTALL DEPENDENCIES (run once in terminal)
# ─────────────────────────────────────────────
# pip install "pinecone>=3.0.0" sentence-transformers rank_bm25 \
#             huggingface_hub scikit-learn numpy langchain \
#             langchain-text-splitters pypdf


# ─────────────────────────────────────────────
# 2. IMPORTS & CONFIGURATION
# ─────────────────────────────────────────────
import os
import re
import time
import json
import glob
from dotenv import load_dotenv
import numpy as np
from typing import List, Dict, Tuple, Optional

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from pinecone import Pinecone, ServerlessSpec
from huggingface_hub import InferenceClient
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LLM Models
LLM_MODELS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "aya":  "CohereForAI/aya-23-8B",
}

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM        = 1024

# CrossEncoder for re-ranking
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

PINECONE_INDEX_NAME  = "legal-urdu-rag-v2"

# Chunking settings
FIXED_CHUNK_SIZE        = 300
FIXED_CHUNK_OVERLAP     = 50
RECURSIVE_CHUNK_SIZE    = 400
RECURSIVE_CHUNK_OVERLAP = 80

# Retrieval: fetch 20 candidates, re-rank to 5 for LLM
TOP_K_RETRIEVAL = 20
TOP_K_FINAL     = 5

print("Configuration set")


# ─────────────────────────────────────────────
# 3. LOAD API KEYS
# ─────────────────────────────────────────────
def load_api_keys() -> Tuple[str, str]:
    """
    Read HF_TOKEN and PINECONE_KEY from environment variables.
    Set them before running:
        export HF_TOKEN="hf_..."
        export PINECONE_KEY="pcsk_..."
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Go up one level to the main project folder (RAG-e-Qanoon)
    root_dir = os.path.dirname(current_dir)
    # 3. Point directly to the .env file
    env_path = os.path.join(root_dir, '.env')

    # Load the .env file from that specific path
    load_dotenv(dotenv_path=env_path)
    
    hf_token     = os.environ.get("HF_TOKEN")
    pinecone_key = os.environ.get("PINECONE_KEY")

    if not hf_token:
        raise ValueError("HF_TOKEN not found. Set it with: export HF_TOKEN='hf_...'")
    if not pinecone_key:
        raise ValueError("PINECONE_KEY not found. Set it with: export PINECONE_KEY='pcsk_...'")

    print(f"HF_TOKEN    : {hf_token[:8]}...  ")
    print(f"PINECONE_KEY: {pinecone_key[:8]}...  ")
    return hf_token, pinecone_key


# ─────────────────────────────────────────────
# 4. LOAD YOUR REAL DATA FROM scrapper/data/raw
# ─────────────────────────────────────────────
from typing import List, Dict, Tuple
import os
import glob

def load_documents_from_folder(folder_path: str) -> Tuple[List[str], List[Dict]]:
    """
    Reads PDFs using EasyOCR for Urdu (Nastaliq) text.
    Uses pdf2image to convert pages to high-DPI images.
    """
    import numpy as np
    import easyocr
    from pdf2image import convert_from_path

    # Initialize EasyOCR for Urdu + English (handles mixed pages)
    print("Loading Urdu OCR Engine...")
    reader = easyocr.Reader(['ur', 'en'], gpu=False)  # set gpu=True if available

    texts = []
    metadata = []

    pdf_files = sorted(glob.glob(os.path.join(folder_path, "*.pdf")))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{folder_path}'.")

    print(f"Found {len(pdf_files)} PDF files. Starting OCR...")

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        try:
            print(f"  Converting {filename} to images...")
            # 250 DPI — critical for Nastaliq ligatures
            pages = convert_from_path(
    pdf_path,
    dpi=250,
    poppler_path=r"C:\Uni stuff\Uni stuff\NLP\project\RAG-e-Qanoon\rag_backend\poppler-25.12.0\Library\bin"
)
            full_text = ""

            for page_num, page_img in enumerate(pages):
                print(f"    OCR on page {page_num + 1}/{len(pages)}...")
                img_array = np.array(page_img)

                # paragraph=True groups text into reading-order blocks
                result = reader.readtext(img_array, paragraph=True)

                if result:
                    # Each result: (bbox, text, confidence)
                    page_text = "\n".join([item[1] for item in result])
                    full_text += page_text + "\n\n"

            if len(full_text.strip()) > 50:
                texts.append(full_text)
                metadata.append({"source": filename, "pages": len(pages)})
                print(f"  ✓ Done: {filename}")
            else:
                print(f"  ⚠ No text found: {filename}")

        except Exception as e:
            print(f"  ✗ Error in {filename}: {e}")

    print(f"\nTotal documents OCR'd: {len(texts)}")
    return texts, metadata
# ─────────────────────────────────────────────
# 5. CHUNKING STRATEGIES
# ─────────────────────────────────────────────
def chunk_fixed(text: str,
                chunk_size: int = FIXED_CHUNK_SIZE,
                overlap: int = FIXED_CHUNK_OVERLAP) -> List[str]:
    """
    Split text into chunks of exactly 'chunk_size' characters,
    with 'overlap' characters shared between consecutive chunks.
    """
    chunks = []
    start  = 0
    text   = text.strip()

    while start < len(text):
        end   = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def chunk_recursive(text: str,
                    chunk_size: int = RECURSIVE_CHUNK_SIZE,
                    overlap: int = RECURSIVE_CHUNK_OVERLAP) -> List[str]:
    """
    Uses LangChain's RecursiveCharacterTextSplitter which tries to split
    on natural boundaries: paragraphs -> sentences -> words -> characters.
    Uses Urdu separators.
    """
    separators = ["\n\n", "\n", "۔", "؟", "!", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c.strip()]


def chunk_sentence(text: str,
                   min_length: int = 50,
                   max_length: int = 500) -> List[str]:
    """
    Splits on Urdu sentence boundaries (۔ ؟ !) then merges short
    adjacent sentences up to max_length characters.
    """
    raw     = re.split(r'(?<=[۔؟!])\s*', text.strip())
    chunks  = []
    current = ""
    for sent in raw:
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) <= max_length:
            current = (current + " " + sent).strip()
        else:
            if len(current) >= min_length:
                chunks.append(current)
            current = sent
    if len(current) >= min_length:
        chunks.append(current)
    return chunks


def chunk_documents(texts: List[str],
                    strategy: str = "recursive",
                    metadata: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Chunks a list of raw documents and returns a list of chunk dicts.
    Each dict: {"id": str, "text": str, "metadata": dict}
    """
    if metadata is None:
        metadata = [{"source": f"doc_{i}"} for i in range(len(texts))]

    chunk_fn = {"recursive": chunk_recursive,
                "fixed":     chunk_fixed,
                "sentence":  chunk_sentence}.get(strategy, chunk_recursive)
    all_chunks = []
    chunk_idx  = 0

    for doc_idx, (text, meta) in enumerate(zip(texts, metadata)):
        for position, chunk_text in enumerate(chunk_fn(text)):
            all_chunks.append({
                "id":   f"chunk_{chunk_idx}",
                "text": chunk_text,
                "metadata": {
                    **meta,
                    "doc_idx":  doc_idx,
                    "position": position,
                    "strategy": strategy,
                }
            })
            chunk_idx += 1

    print(f"Chunked {len(texts)} docs -> {len(all_chunks)} chunks (strategy: {strategy})")
    return all_chunks


# ─────────────────────────────────────────────
# 6. EMBEDDING MODEL
# ─────────────────────────────────────────────
class EmbeddingModel:

    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model loaded")

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Convert a list of strings into embedding vectors."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        return self.model.encode(query, normalize_embeddings=True)


# ─────────────────────────────────────────────
# 7. PINECONE VECTOR DATABASE
# ─────────────────────────────────────────────
class PineconeDB:
    """Handles connecting, upserting, and querying Pinecone index."""

    def __init__(self, api_key: str, index_name: str = PINECONE_INDEX_NAME):
        self.pc = Pinecone(api_key=api_key)

        existing = [idx.name for idx in self.pc.list_indexes()]
        if index_name not in existing:
            print(f"  Creating Pinecone index '{index_name}'...")
            self.pc.create_index(
                name=index_name,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            time.sleep(30)
            print(f"Index '{index_name}' created")
        else:
            print(f"Connected to existing Pinecone index: '{index_name}'")

        self.index = self.pc.Index(index_name)

    def upsert_chunks(self, chunks: List[Dict], embeddings: np.ndarray,
                      batch_size: int = 100):
        """Upload chunk texts + their embeddings to Pinecone in batches."""
        vectors = []
        for chunk, emb in zip(chunks, embeddings):
            vectors.append({
                "id":     chunk["id"],
                "values": emb.tolist(),
                "metadata": {
                    **chunk["metadata"],
                    "text": chunk["text"][:1000],
                }
            })

        total = 0
        for i in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[i : i + batch_size])
            total += len(vectors[i : i + batch_size])
            print(f"  ↑ Upserted {total}/{len(vectors)} vectors")

        print(f"All {len(vectors)} chunks stored in Pinecone")

    def semantic_search(self, query_embedding: np.ndarray,
                        top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
        """Search Pinecone for chunks most similar to the query embedding."""
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
        )
        return [{
            "id":       m.id,
            "score":    m.score,
            "text":     m.metadata.get("text", ""),
            "metadata": m.metadata,
        } for m in results.matches]

    def get_stats(self) -> Dict:
        """Returns how many vectors are stored in index."""
        return self.index.describe_index_stats()


# ─────────────────────────────────────────────
# 8. BM25 KEYWORD SEARCH
# ─────────────────────────────────────────────
class BM25Retriever:

    def __init__(self):
        self.bm25   = None
        self.chunks = []

    def build_index(self, chunks: List[Dict]):
        """Build the BM25 index from a list of chunk dicts."""
        self.chunks   = chunks
        tokenized     = [self._tokenize(c["text"]) for c in chunks]
        self.bm25     = BM25Okapi(tokenized)
        print(f"BM25 index built on {len(chunks)} chunks")

    def _tokenize(self, text: str) -> List[str]:
        """Whitespace + punctuation tokenizer."""
        text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
        return [t for t in text.split() if t]

    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
        """Search using BM25. Returns chunks sorted by keyword match score."""
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built.")

        scores      = self.bm25.get_scores(self._tokenize(query))
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [{
            "id":       self.chunks[i]["id"],
            "score":    float(scores[i]),
            "text":     self.chunks[i]["text"],
            "metadata": self.chunks[i].get("metadata", {}),
        } for i in top_indices]


# ─────────────────────────────────────────────
# 9. RECIPROCAL RANK FUSION (RRF)
# ─────────────────────────────────────────────
def reciprocal_rank_fusion(
    semantic_hits: List[Dict],
    bm25_hits: List[Dict],
    k: int = 60,
    semantic_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> List[Dict]:
    """
    Merges semantic search + BM25 results using RRF.
    Formula: score += weight / (k + rank)
    """
    fused_scores: Dict[str, float] = {}
    chunk_store:  Dict[str, Dict]  = {}

    for rank, hit in enumerate(semantic_hits, start=1):
        cid = hit["id"]
        fused_scores[cid] = fused_scores.get(cid, 0.0) + semantic_weight / (k + rank)
        chunk_store[cid]  = hit

    for rank, hit in enumerate(bm25_hits, start=1):
        cid = hit["id"]
        fused_scores[cid] = fused_scores.get(cid, 0.0) + bm25_weight / (k + rank)
        if cid not in chunk_store:
            chunk_store[cid] = hit

    sorted_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    results    = []
    for cid in sorted_ids:
        hit = chunk_store[cid].copy()
        hit["rrf_score"] = fused_scores[cid]
        results.append(hit)

    return results


# ─────────────────────────────────────────────
# 10. CROSS-ENCODER RE-RANKING
# ─────────────────────────────────────────────
class Reranker:

    def __init__(self):
        print("Loading CrossEncoder re-ranker...")
        self.model = CrossEncoder(RERANKER_MODEL_NAME)
        print("Re-ranker loaded")

    def rerank(self, query: str, candidates: List[Dict],
               top_k: int = TOP_K_FINAL) -> List[Dict]:
        """
        Re-ranks candidates using cross-attention scoring.
        Returns top_k chunks sorted by cross-encoder score.
        """
        if not candidates:
            return []

        pairs  = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        for chunk, score in zip(candidates, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]


# ─────────────────────────────────────────────
# 11. LLM GENERATION
# ─────────────────────────────────────────────
class LLMGenerator:
    """Connects to HuggingFace Inference API for LLM text generation."""

    def __init__(self, hf_token: str, model_choice: str = "qwen"):
        if model_choice not in LLM_MODELS:
            raise ValueError(f"model_choice must be one of: {list(LLM_MODELS.keys())}")

        self.model_name   = LLM_MODELS[model_choice]
        self.model_choice = model_choice
        self.client       = InferenceClient(model=self.model_name, token=hf_token)
        print(f"LLM ready: {self.model_name}")

    def _build_generation_prompt(self, query: str,
                                  context_chunks: List[Dict]) -> str:
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            context_text += f"\n[دستاویز {i}]:\n{chunk['text']}\n"

        return f"""آپ ایک ماہر پاکستانی قانونی مشیر ہیں۔ آپ کا کام صرف نیچے دیے گئے قانونی دستاویزات کی بنیاد پر سوال کا جواب دینا ہے۔
اگر جواب دستاویزات میں موجود نہ ہو تو صاف کہیں: "یہ معلومات دستیاب دستاویزات میں موجود نہیں ہیں۔"
اپنی طرف سے کوئی بات نہ بنائیں۔

=== قانونی دستاویزات ===
{context_text}
=== دستاویزات ختم ===

سوال: {query}

جواب (صرف دستاویزات کی بنیاد پر):"""

    def generate(self, query: str, context_chunks: List[Dict],
                 max_new_tokens: int = 400) -> Tuple[str, float]:
        """Generate an answer. Returns (answer_text, generation_time_seconds)."""
        prompt = self._build_generation_prompt(query, context_chunks)
        start  = time.time()

        try:
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=0.3,
            )
            answer = response.choices[0].message["content"].strip()
        except Exception as e:
            answer = f"[LLM Error: {str(e)}]"

        return answer, round(time.time() - start, 2)


# ─────────────────────────────────────────────
# 12. LLM-AS-A-JUDGE (Faithfulness + Relevancy)
# ─────────────────────────────────────────────
class LLMJudge:
    """Automated LLM-as-a-Judge evaluation."""

    def __init__(self, hf_token: str, embedding_model: EmbeddingModel,
                 judge_model: str = "qwen"):
        self.client          = InferenceClient(model=LLM_MODELS[judge_model], token=hf_token)
        self.embedding_model = embedding_model
        print(f"LLM Judge ready: {LLM_MODELS[judge_model]}")

    def extract_claims(self, answer: str) -> List[str]:
        prompt = f"""نیچے دیے گئے جواب سے تمام حقائق اور دعوے (claims) نکالیں۔
ہر دعوہ ایک الگ لائن پر لکھیں، نمبر کے ساتھ (1. 2. 3. ...)
صرف فہرست لکھیں، کوئی اضافی جملہ نہیں۔

جواب:
{answer}

دعووں کی فہرست:"""

        try:
            response      = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1,
            )
            response_text = response.choices[0].message["content"]
            lines         = response_text.strip().split("\n")
            claims        = []
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    clean = re.sub(r"^[\d\-\.\)]+\s*", "", line).strip()
                    if clean:
                        claims.append(clean)
            return claims if claims else [answer]
        except Exception as e:
            print(f"  Claim extraction error: {e}")
            return [answer]

    def verify_claim(self, claim: str, context: str) -> bool:
        prompt = f"""نیچے دیا گیا دعویٰ (claim) دیے گئے متن (context) سے ثابت ہوتا ہے یا نہیں؟
صرف ایک لفظ میں جواب دیں: "ہاں" یا "نہیں"

متن (Context):
{context[:1500]}

دعویٰ (Claim): {claim}

جواب (صرف ہاں یا نہیں):"""

        try:
            response    = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1,
            )
            answer_text = response.choices[0].message["content"].strip().lower()
            return any(w in answer_text for w in ["ہاں", "yes", "supported", "correct", "true", "han"])
        except Exception:
            return False

    def compute_faithfulness(self, answer: str, context_chunks: List[Dict]) -> Dict:
        """Full faithfulness pipeline. Returns score (0–1) + claim details."""
        context_text  = " ".join([c["text"] for c in context_chunks])
        claims        = self.extract_claims(answer)
        verifications = [self.verify_claim(c, context_text) for c in claims]
        supported     = sum(verifications)
        total         = len(verifications) if verifications else 1

        return {
            "score":           round(supported / total, 3),
            "claims":          claims,
            "verifications":   verifications,
            "supported_count": supported,
            "total_claims":    total,
        }

    def generate_questions_from_answer(self, answer: str) -> List[str]:
        prompt = f"""نیچے دیے گئے جواب کو پڑھ کر 3 سوالات بنائیں جو اس جواب سے پوچھے جا سکتے ہیں۔
ہر سوال ایک نئی لائن پر لکھیں۔ صرف سوالات لکھیں، کچھ اور نہیں۔

جواب:
{answer}

تین سوالات:
1."""

        try:
            response  = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.4,
            )
            full_text = "1." + response.choices[0].message["content"]
            lines     = full_text.strip().split("\n")
            questions = []
            for line in lines:
                line  = line.strip()
                clean = re.sub(r"^[\d\.\)]+\s*", "", line).strip()
                if clean and len(clean) > 5:
                    questions.append(clean)
            return questions[:3]
        except Exception as e:
            print(f"  Question generation error: {e}")
            return []

    def compute_relevancy(self, original_query: str, answer: str) -> Dict:
        """Full relevancy pipeline. Returns score (0–1) + generated questions."""
        generated_questions = self.generate_questions_from_answer(answer)

        if not generated_questions:
            return {"score": 0.0, "generated_questions": [], "similarities": []}

        query_emb     = self.embedding_model.embed_query(original_query)
        question_embs = self.embedding_model.embed(generated_questions)

        similarities = [
            round(float(cosine_similarity(query_emb.reshape(1, -1), q.reshape(1, -1))[0][0]), 3)
            for q in question_embs
        ]

        return {
            "score":               round(float(np.mean(similarities)), 3) if similarities else 0.0,
            "generated_questions": generated_questions,
            "similarities":        similarities,
        }


# ─────────────────────────────────────────────
# 13. MAIN RAGPipeline CLASS
# ─────────────────────────────────────────────
class RAGPipeline:
    """
    Full end-to-end RAG pipeline for Urdu Legal QA.

    Usage:
        rag = RAGPipeline(model_choice="qwen")
        rag.ingest_documents(texts=["doc1...", "doc2..."])
        result = rag.query("خلع کے لیے کیا کرنا ہوگا؟")
    """

    def __init__(self, model_choice: str = "qwen",
                 chunking_strategy: str = "recursive"):
        print("\n" + "="*60)
        print("  Initializing Urdu Legal RAG Pipeline")
        print("="*60)

        self.chunking_strategy = chunking_strategy
        self.model_choice      = model_choice
        self.hf_token, pinecone_key = load_api_keys()

        self.embedding_model = EmbeddingModel()
        self.pinecone_db     = PineconeDB(api_key=pinecone_key)
        self.bm25_retriever  = BM25Retriever()
        self.reranker        = Reranker()
        self.llm             = LLMGenerator(self.hf_token, model_choice)
        self.judge           = LLMJudge(self.hf_token, self.embedding_model,
                                        judge_model=model_choice)
        self.all_chunks: List[Dict] = []

        print("\nRAG Pipeline fully initialized!\n")

    def ingest_documents(self, texts: List[str],
                         metadata: Optional[List[Dict]] = None,
                         chunking_strategy: Optional[str] = None):
        """Chunk -> Embed -> Upsert to Pinecone -> Build BM25 index."""
        strategy = chunking_strategy or self.chunking_strategy
        print(f"\n Ingesting {len(texts)} documents (strategy: {strategy})...")

        t0 = time.time()
        chunks = chunk_documents(texts, strategy=strategy, metadata=metadata)
        print(f"  Chunking:        {round(time.time()-t0, 2)}s")

        t0 = time.time()
        embeddings = self.embedding_model.embed([c["text"] for c in chunks])
        print(f"  Embedding:       {round(time.time()-t0, 2)}s")

        t0 = time.time()
        self.pinecone_db.upsert_chunks(chunks, embeddings)
        print(f"  Pinecone upsert: {round(time.time()-t0, 2)}s")

        t0 = time.time()
        self.all_chunks.extend(chunks)
        self.bm25_retriever.build_index(self.all_chunks)
        print(f"  BM25 index:      {round(time.time()-t0, 2)}s")

        print(f"\n Ingestion complete. Total chunks in system: {len(self.all_chunks)}")

    def retrieve(self, query: str,
                 use_reranker: bool = True) -> Tuple[List[Dict], Dict]:
        """Semantic -> BM25 -> RRF fusion -> (optional) CrossEncoder re-ranking."""
        timings = {}

        t0 = time.time()
        query_emb     = self.embedding_model.embed_query(query)
        semantic_hits = self.pinecone_db.semantic_search(query_emb, top_k=TOP_K_RETRIEVAL)
        timings["semantic_retrieval"] = round(time.time() - t0, 3)

        t0 = time.time()
        bm25_hits = self.bm25_retriever.search(query, top_k=TOP_K_RETRIEVAL)
        timings["bm25_retrieval"] = round(time.time() - t0, 3)

        t0 = time.time()
        fused = reciprocal_rank_fusion(semantic_hits, bm25_hits)
        timings["rrf_fusion"] = round(time.time() - t0, 3)

        if use_reranker and fused:
            t0 = time.time()
            final_chunks = self.reranker.rerank(query, fused, top_k=TOP_K_FINAL)
            timings["reranking"] = round(time.time() - t0, 3)
        else:
            final_chunks = fused[:TOP_K_FINAL]

        timings["total_retrieval"] = sum(timings.values())
        return final_chunks, timings

    def query(self, user_query: str,
              run_evaluation: bool = True,
              use_reranker: bool = True) -> Dict:
        """
        Main function - takes a query, returns everything.
        """
        print(f"\n Query: {user_query}")
        total_start = time.time()

        retrieved_chunks, retrieval_timings = self.retrieve(user_query, use_reranker=use_reranker)
        print(f"  Retrieved {len(retrieved_chunks)} chunks")

        t0 = time.time()
        answer, _ = self.llm.generate(user_query, retrieved_chunks)
        generation_time = round(time.time() - t0, 3)
        print(f"  Answer generated ({generation_time}s)")

        faithfulness_result = {"score": None}
        relevancy_result    = {"score": None}

        if run_evaluation and retrieved_chunks:
            print("  Running LLM-as-a-Judge...")
            faithfulness_result = self.judge.compute_faithfulness(answer, retrieved_chunks)
            relevancy_result    = self.judge.compute_relevancy(user_query, answer)
            print(f"  Faithfulness: {faithfulness_result['score']:.2%}  "
                  f"Relevancy: {relevancy_result['score']:.2%}")

        return {
            "query":            user_query,
            "answer":           answer,
            "retrieved_chunks": retrieved_chunks,
            "faithfulness":     faithfulness_result,
            "relevancy":        relevancy_result,
            "timings": {
                **retrieval_timings,
                "generation": generation_time,
                "total":      round(time.time() - total_start, 3),
            },
        }

    def run_ablation(self, test_queries: List[str],
                     texts: List[str],
                     metadata: Optional[List[Dict]] = None) -> Dict:
        """
        Run the ablation study across 4 configurations:
          A: Fixed chunking   + Semantic only
          B: Fixed chunking   + Hybrid + Re-ranking
          C: Recursive        + Semantic only
          D: Recursive        + Hybrid + Re-ranking  ← expected best
        """
        print("\n" + "="*60)
        print("  ABLATION STUDY")
        print("="*60)

        configs = [
            {"label": "A: Fixed + Semantic Only",       "chunking": "fixed",     "reranker": False},
            {"label": "B: Fixed + Hybrid + Rerank",     "chunking": "fixed",     "reranker": True},
            {"label": "C: Recursive + Semantic Only",   "chunking": "recursive", "reranker": False},
            {"label": "D: Recursive + Hybrid + Rerank", "chunking": "recursive", "reranker": True},
        ]

        all_results = {}

        for cfg in configs:
            print(f"\n Running: {cfg['label']}")
            self.all_chunks = []
            self.ingest_documents(texts, metadata=metadata,
                                  chunking_strategy=cfg["chunking"])

            faith_scores, rel_scores, latencies = [], [], []

            for q in test_queries:
                try:
                    chunks, timings = self.retrieve(q, use_reranker=cfg["reranker"])
                    answer, _       = self.llm.generate(q, chunks)
                    faith           = self.judge.compute_faithfulness(answer, chunks)
                    rel             = self.judge.compute_relevancy(q, answer)
                    faith_scores.append(faith["score"])
                    rel_scores.append(rel["score"])
                    latencies.append(timings["total_retrieval"])
                except Exception as e:
                    print(f"   Query failed: {e}")

            result = {
                "avg_faithfulness": round(np.mean(faith_scores), 3) if faith_scores else 0.0,
                "avg_relevancy":    round(np.mean(rel_scores), 3)    if rel_scores   else 0.0,
                "avg_latency_s":    round(np.mean(latencies), 3)     if latencies    else 0.0,
                "n_queries":        len(faith_scores),
            }
            all_results[cfg["label"]] = result
            print(f"  Faithfulness: {result['avg_faithfulness']:.2%}  "
                  f"Relevancy: {result['avg_relevancy']:.2%}  "
                  f"Latency: {result['avg_latency_s']}s")

        print("\n" + "="*65)
        print("  ABLATION RESULTS TABLE")
        print("="*65)
        print(f"{'Configuration':<35} {'Faithful':>10} {'Relevancy':>10} {'Latency':>8}")
        print("-"*65)
        for label, res in all_results.items():
            print(f"{label:<35} {res['avg_faithfulness']:>10.2%} "
                  f"{res['avg_relevancy']:>10.2%} {res['avg_latency_s']:>7.2f}s")

        return all_results

    def switch_model(self, model_choice: str):
        """Switch the LLM without re-loading embeddings."""
        self.llm          = LLMGenerator(self.hf_token, model_choice)
        self.judge        = LLMJudge(self.hf_token, self.embedding_model, model_choice)
        self.model_choice = model_choice
        print(f"Switched to: {LLM_MODELS[model_choice]}")

    def get_pinecone_stats(self):
        """How many vectors are stored in Pinecone index."""
        return self.pinecone_db.get_stats()


# ─────────────────────────────────────────────
# 14. MAIN - RUNS WHEN YOU EXECUTE THIS FILE
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # ── Load your real PDF data ──
    # Point this to your actual folder
    # ── Load your real PDF data ──
    # Point this to your actual folder (going up one level)
    DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scrapper", "data", "raw")

    print(f"\nLoading documents from: {DATA_FOLDER}")
    texts, metadata = load_documents_from_folder(DATA_FOLDER)

    # ── Initialize the pipeline ──
    rag = RAGPipeline(model_choice="qwen", chunking_strategy="recursive")

    # ── Ingest your real documents ──
    rag.ingest_documents(texts, metadata=metadata)

    # ── Run a sample query ──
    MY_QUERY = "پاکستان کا ریاستی مذہب کیا ہے اور آئین کے تحت شہریوں کے بنیادی حقوق کیا ہیں؟"
# (What is the state religion of Pakistan and what are the fundamental rights of citizens under the constitution?)

    result = rag.query(MY_QUERY, run_evaluation=True)

    print("\n" + "="*60)
    print("QUERY RESULT")
    print("="*60)
    print(f"Question:     {result['query']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nFaithfulness: {result['faithfulness']['score']:.2%}")
    print(f"Relevancy:    {result['relevancy']['score']:.2%}")
    print(f"Total Time:   {result['timings']['total']}s")
    print("\nRetrieved Chunks:")
    for i, chunk in enumerate(result["retrieved_chunks"], 1):
        print(f"  [{i}] {chunk['text'][:100]}...")

    # ── (Optional) Run the ablation study ──
    # test_queries = [
    #     "خلع کے لیے کیا کرنا ہوگا؟",
    #     "گرفتاری کے وقت میرے کیا حقوق ہیں؟",
    #     "CNIC گم ہو جائے تو کیا کریں؟",
    # ]
    # ablation_results = rag.run_ablation(
    #     test_queries=test_queries,
    #     texts=texts,
    #     metadata=metadata,
    # )

    # ── (Optional) Switch model ──
    # rag.switch_model("aya")
    # result2 = rag.query(MY_QUERY)

    # ── Index stats ──
    stats = rag.get_pinecone_stats()
    print(f"\nPinecone Index Stats:\n{stats}")