# ─────────────────────────────────────────────────────────────────────────────
# RAG-e-Qanoon  |  rag_backend/rag_pipeline.py
# Best config from ablation study:
#   Chunking  : fixed, 512 chars / 80 overlap
#   RRF       : semantic=0.5, bm25=0.5
#   Top-k     : 5 final chunks
#   Temperature: 0.3
#   Re-ranker : ON (bge-reranker-v2-m3)
#   LLM       : Qwen/Qwen2.5-14B-Instruct  (fallback → Qwen2.5-7B-Instruct)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────
# 1. IMPORTS & CONFIGURATION
# ─────────────────────────────────────────────
import os
import re
import time
import json
import numpy as np
from typing import List, Dict, Tuple, Optional

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from pinecone import Pinecone, ServerlessSpec
from huggingface_hub import InferenceClient
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
import traceback

DEBUG = True

def dbg(*args):
    if DEBUG:
        print("[DEBUG]", *args)
# ── On HF Spaces, secrets are injected directly as env vars.
# ── Locally, you can still use a .env file — just call load_dotenv() before
# ── importing this module.  We do NOT call load_dotenv() here so that HF
# ── Spaces (which has no .env) doesn't throw an import error.

# ── LLM model registry ────────────────────────────────────────────────────
# ── LLM model registry ────────────────────────────────────────────────────
LLM_PRIMARY  = "Qwen/Qwen3-14B:nscale"
LLM_FALLBACK = "Qwen/Qwen2.5-7B-Instruct"
# ── Embedding / re-ranker ─────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM        = 1024
RERANKER_MODEL_NAME  = "BAAI/bge-reranker-v2-m3"

# ── Pinecone ──────────────────────────────────────────────────────────────
PINECONE_INDEX_NAME = "legal-urdu-rag-v3"

# ── BEST chunking config (ablation winner: fixed 512/80) ─────────────────
FIXED_CHUNK_SIZE    = 512
FIXED_CHUNK_OVERLAP = 80

# ── Kept for ingestion flexibility ───────────────────────────────────────
RECURSIVE_CHUNK_SIZE    = 400
RECURSIVE_CHUNK_OVERLAP = 80

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL = 20   # candidates fetched from each retriever
TOP_K_FINAL     = 5    # chunks handed to LLM after re-ranking

# ── RRF weights (ablation winner: 0.5 / 0.5) ─────────────────────────────
RRF_SEMANTIC_WEIGHT = 0.5
RRF_BM25_WEIGHT     = 0.5

# ── Generation (ablation winner: temp 0.3) ────────────────────────────────
GENERATION_TEMPERATURE = 0.3
GENERATION_MAX_TOKENS  = 512

print("RAG-e-Qanoon configuration loaded.")


# ─────────────────────────────────────────────
# 2. LOAD API KEYS  (env-first, no hard crash on missing .env)
# ─────────────────────────────────────────────
def load_api_keys() -> Tuple[str, str]:
    """
    Read HF_TOKEN and PINECONE_KEY from environment variables.
    On HF Spaces: set them in Settings → Secrets.
    Locally: export them, or call load_dotenv() before importing this module.
    """
    hf_token     = os.environ.get("HF_TOKEN", "")
    pinecone_key = os.environ.get("PINECONE_KEY", "")

    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment. Add it to HF Space Secrets.")
    if not pinecone_key:
        raise ValueError("PINECONE_KEY not found in environment. Add it to HF Space Secrets.")

    print(f"HF_TOKEN    : {hf_token[:8]}…")
    print(f"PINECONE_KEY: {pinecone_key[:8]}…")
    return hf_token, pinecone_key


# ─────────────────────────────────────────────
# 3. LOAD CLEANED JSON DATA
# ─────────────────────────────────────────────
def load_documents_from_json(json_path: str) -> Tuple[List[str], List[Dict]]:
    """Load pre-processed cleaned Urdu text from JSON produced by cleancode.py."""
    print(f"Loading documents from {json_path}…")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Cannot find {json_path}. Run cleancode.py first.")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts    = [item["text"]  for item in data]
    metadata = [item["meta"]  for item in data]
    print(f"Loaded {len(texts)} documents.")
    return texts, metadata


# ─────────────────────────────────────────────
# 4. CHUNKING STRATEGIES
# ─────────────────────────────────────────────
def chunk_fixed(text: str,
                chunk_size: int = FIXED_CHUNK_SIZE,
                overlap: int    = FIXED_CHUNK_OVERLAP) -> List[str]:
    chunks, start = [], 0
    text = text.strip()
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def chunk_recursive(text: str,
                    chunk_size: int = RECURSIVE_CHUNK_SIZE,
                    overlap: int    = RECURSIVE_CHUNK_OVERLAP) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "۔", "؟", "!", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


def chunk_sentence(text: str,
                   min_length: int = 50,
                   max_length: int = 500) -> List[str]:
    raw, chunks, current = re.split(r'(?<=[۔؟!])\s*', text.strip()), [], ""
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
                    strategy: str = "fixed",
                    metadata: Optional[List[Dict]] = None) -> List[Dict]:
    if metadata is None:
        metadata = [{"source": f"doc_{i}"} for i in range(len(texts))]

    chunk_fn = {"fixed": chunk_fixed,
                "recursive": chunk_recursive,
                "sentence":  chunk_sentence}.get(strategy, chunk_fixed)

    all_chunks, chunk_idx = [], 0
    for doc_idx, (text, meta) in enumerate(zip(texts, metadata)):
        for position, chunk_text in enumerate(chunk_fn(text)):
            all_chunks.append({
                "id":   f"chunk_{chunk_idx}",
                "text": chunk_text,
                "metadata": {**meta, "doc_idx": doc_idx,
                             "position": position, "strategy": strategy},
            })
            chunk_idx += 1

    print(f"Chunked {len(texts)} docs → {len(all_chunks)} chunks (strategy: {strategy})")
    return all_chunks


# ─────────────────────────────────────────────
# 5. EMBEDDING MODEL
# ─────────────────────────────────────────────
class EmbeddingModel:

    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model loaded.")

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size,
                                 show_progress_bar=True, normalize_embeddings=True)

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(query, normalize_embeddings=True)


# ─────────────────────────────────────────────
# 6. PINECONE VECTOR DATABASE
# ─────────────────────────────────────────────
class PineconeDB:

    def __init__(self, api_key: str, index_name: str = PINECONE_INDEX_NAME):
        self.pc = Pinecone(api_key=api_key)
        existing = [idx.name for idx in self.pc.list_indexes()]
        if index_name not in existing:
            print(f"Creating Pinecone index '{index_name}'…")
            self.pc.create_index(
                name=index_name, dimension=EMBEDDING_DIM, metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            time.sleep(30)
        else:
            print(f"Connected to existing Pinecone index: '{index_name}'")
        self.index = self.pc.Index(index_name)

    def upsert_chunks(self, chunks: List[Dict], embeddings: np.ndarray,
                      batch_size: int = 100):
        vectors = [
            {"id": c["id"], "values": e.tolist(),
             "metadata": {**c["metadata"], "text": c["text"][:1000]}}
            for c, e in zip(chunks, embeddings)
        ]
        total = 0
        for i in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[i : i + batch_size])
            total += len(vectors[i : i + batch_size])
            print(f"  ↑ Upserted {total}/{len(vectors)} vectors")
        print(f"All {len(vectors)} chunks stored in Pinecone.")

    def semantic_search(self, query_embedding: np.ndarray,
                        top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
        results = self.index.query(vector=query_embedding.tolist(),
                                   top_k=top_k, include_metadata=True)
        return [{"id": m.id, "score": m.score,
                 "text": m.metadata.get("text", ""), "metadata": m.metadata}
                for m in results.matches]

    def get_stats(self) -> Dict:
        return self.index.describe_index_stats()


# ─────────────────────────────────────────────
# 7. BM25 KEYWORD SEARCH
# ─────────────────────────────────────────────
class BM25Retriever:

    def __init__(self):
        self.bm25   = None
        self.chunks = []

    def build_index(self, chunks: List[Dict]):
        self.chunks = chunks
        self.bm25   = BM25Okapi([self._tokenize(c["text"]) for c in chunks])
        print(f"BM25 index built on {len(chunks)} chunks.")

    def _tokenize(self, text: str) -> List[str]:
        text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
        return [t for t in text.split() if t]

    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built yet.")
        scores      = self.bm25.get_scores(self._tokenize(query))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{"id": self.chunks[i]["id"], "score": float(scores[i]),
                 "text": self.chunks[i]["text"],
                 "metadata": self.chunks[i].get("metadata", {})}
                for i in top_indices]


# ─────────────────────────────────────────────
# 8. RECIPROCAL RANK FUSION  (best: 0.5 / 0.5)
# ─────────────────────────────────────────────
def reciprocal_rank_fusion(
    semantic_hits: List[Dict],
    bm25_hits: List[Dict],
    k: int = 60,
    semantic_weight: float = RRF_SEMANTIC_WEIGHT,
    bm25_weight: float     = RRF_BM25_WEIGHT,
) -> List[Dict]:
    fused_scores: Dict[str, float] = {}
    chunk_store:  Dict[str, Dict]  = {}

    for rank, hit in enumerate(semantic_hits, 1):
        cid = hit["id"]
        fused_scores[cid] = fused_scores.get(cid, 0.0) + semantic_weight / (k + rank)
        chunk_store[cid]  = hit

    for rank, hit in enumerate(bm25_hits, 1):
        cid = hit["id"]
        fused_scores[cid] = fused_scores.get(cid, 0.0) + bm25_weight / (k + rank)
        chunk_store.setdefault(cid, hit)

    results = []
    for cid in sorted(fused_scores, key=fused_scores.get, reverse=True):
        hit = chunk_store[cid].copy()
        hit["rrf_score"] = fused_scores[cid]
        results.append(hit)
    return results


# ─────────────────────────────────────────────
# 9. CROSS-ENCODER RE-RANKER  (always ON)
# ─────────────────────────────────────────────
class Reranker:

    def __init__(self):
        print("Loading CrossEncoder re-ranker…")
        self.model = CrossEncoder(RERANKER_MODEL_NAME)
        print("Re-ranker loaded.")

    def rerank(self, query: str, candidates: List[Dict],
               top_k: int = TOP_K_FINAL) -> List[Dict]:
        if not candidates:
            return []
        scores = self.model.predict([(query, c["text"]) for c in candidates])
        for chunk, score in zip(candidates, scores):
            chunk["rerank_score"] = float(score)
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


# ─────────────────────────────────────────────
# 10. LLM GENERATOR  (14B primary, 7B fallback)
# ─────────────────────────────────────────────
class LLMGenerator:
    """
    Calls the HF Inference API.
    Tries Qwen2.5-14B-Instruct first; falls back to 7B if the free tier
    returns a model-too-large / memory error.
    Conversation memory (last N turns from MongoDB) is injected into the
    system prompt so the model has cross-turn context.
    """

    def __init__(self, hf_token: str):
        self.hf_token     = hf_token
        self.model_name   = LLM_PRIMARY
        self.client       = InferenceClient(model=self.model_name, token=hf_token)
        print(f"LLM primary: {self.model_name}")

    def _switch_to_fallback(self):
        print(f"Switching to fallback LLM: {LLM_FALLBACK}")
        self.model_name = LLM_FALLBACK
        self.client     = InferenceClient(model=self.model_name, token=self.hf_token)

    # ── Prompt builders ────────────────────────────────────────────────────

    @staticmethod
    def _format_context(chunks: List[Dict]) -> str:
        return "\n\n".join(
            f"{c['text']}" for c in chunks
        )

    @staticmethod
    def _format_memory(conversation_history: List[Dict]) -> str:
        """
        Converts the last N messages from MongoDB into a readable Urdu
        conversation string that is injected into the system prompt.
        Each entry: {"role": "user"|"assistant", "content": "..."}
        """
        if not conversation_history:
            return ""
        lines = []
        for msg in conversation_history:
            role_label = "صارف" if msg["role"] == "user" else "معاون"
            lines.append(f"{role_label}: {msg['content']}")
        return "\n".join(lines)

    def _build_messages(self, query: str, context_chunks: List[Dict],
                        conversation_history: List[Dict]) -> List[Dict]:
        """
        Returns a messages list for chat_completion.

        System prompt contains:
          - Role definition (Urdu legal advisor)
          - Retrieved legal context
          - Recent conversation memory (from MongoDB)

        User turn contains the current query only.
        """
        context_text = self._format_context(context_chunks)
        memory_text  = self._format_memory(conversation_history)

        memory_section = ""
        if memory_text:
            memory_section = (
                "\n\n=== گزشتہ گفتگو (حوالے کے لیے) ===\n"
                f"{memory_text}\n"
                "=== گزشتہ گفتگو ختم ===\n"
            )

        system_prompt = (
            "آپ ایک ماہر پاکستانی قانونی مشیر ہیں۔ "
            "آپ کا کام صرف نیچے دیے گئے قانونی دستاویزات کی بنیاد پر سوال کا جواب دینا ہے۔\n"
            "اگر جواب دستاویزات میں موجود نہ ہو تو صاف کہیں: "
            "\"یہ معلومات دستیاب دستاویزات میں موجود نہیں ہیں۔\"\n"
            "اپنی طرف سے کوئی بات نہ بنائیں۔\n\n"
            "اگر ممکن ہو تو جواب مختصر اور سادہ رکھیں، اور دستاویز نمبر یا غیر ضروری حوالہ نہ دیں جب تک وہ بالکل واضح نہ ہو۔\n"
            "=== قانونی دستاویزات ===\n"
            f"{context_text}\n"
            "=== دستاویزات ختم ==="
            f"{memory_section}"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": query},
        ]

    # ── Public generate method ─────────────────────────────────────────────

    def generate(self, query: str, context_chunks: List[Dict],
             conversation_history: Optional[List[Dict]] = None,
             max_new_tokens: int = GENERATION_MAX_TOKENS) -> Tuple[str, float]:

        if conversation_history is None:
            conversation_history = []

        messages = self._build_messages(query, context_chunks, conversation_history)
        start = time.time()

        dbg("=" * 80)
        dbg("GENERATION START")
        dbg("model:", self.model_name)
        dbg("query:", query)
        dbg("context chunk count:", len(context_chunks))
        dbg("history count:", len(conversation_history))
        for i, c in enumerate(context_chunks[:3]):
            dbg(f"context[{i}] text={c.get('text','')[:200]}")
        dbg("system prompt preview:", messages[0]["content"][:500])
        dbg("user prompt preview:", messages[1]["content"][:300])

        try:
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=GENERATION_TEMPERATURE,
            )
            answer = response.choices[0].message["content"].strip()
            dbg("raw answer preview:", answer[:500])

        except Exception as e:
            print("[ERROR] primary LLM call failed:", e)
            traceback.print_exc()
            err_str = str(e).lower()

            if any(kw in err_str for kw in
                ["model too large", "out of memory", "oom", "422",
                    "loading", "quota", "rate limit", "too large"]):
                self._switch_to_fallback()
                dbg("switched to fallback:", self.model_name)
                try:
                    response = self.client.chat_completion(
                        messages=messages,
                        max_tokens=max_new_tokens,
                        temperature=GENERATION_TEMPERATURE,
                    )
                    answer = response.choices[0].message["content"].strip()
                    dbg("fallback answer preview:", answer[:500])
                except Exception as e2:
                    print("[ERROR] fallback LLM call failed:", e2)
                    traceback.print_exc()
                    answer = f"[LLM Error (fallback): {e2}]"
            else:
                answer = f"[LLM Error: {e}]"

        dbg("GENERATION END")
        dbg("=" * 80)
        return answer, round(time.time() - start, 2)

# ─────────────────────────────────────────────
# 11. LLM-AS-A-JUDGE
# ─────────────────────────────────────────────
class LLMJudge:
    """Faithfulness (claim verification) + Relevancy (query similarity)."""

    def __init__(self, hf_token: str, embedding_model: EmbeddingModel):
        # Judge always uses the primary LLM; falls back transparently
        self.llm_gen         = LLMGenerator(hf_token)
        self.embedding_model = embedding_model
        # Lightweight client for short judge calls (no memory needed)
        self.client = InferenceClient(model=self.llm_gen.model_name, token=hf_token)
        print(f"LLM Judge ready: {self.llm_gen.model_name}")

    def _judge_call(self, prompt: str, max_tokens: int = 300,
            temperature: float = 0.1) -> str:
        try:
            r = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            msg = r.choices[0].message

            # Extract content safely without re-using variable before assignment
            if isinstance(msg, dict):
                raw_content = msg.get("content")
            else:
                raw_content = getattr(msg, "content", None)

            if raw_content is None:
                return ""

            if isinstance(raw_content, list):
                parts = []
                for item in raw_content:
                    if isinstance(item, dict):
                        parts.append(item.get("text", ""))
                    else:
                        parts.append(str(item))
                raw_content = "".join(parts)

            result = str(raw_content).strip()

            # Strip Qwen3 thinking block
            result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()

            return result

        except Exception as e:
            print("[ERROR] judge call failed:", e)
            traceback.print_exc()
            return ""
    def extract_claims(self, answer: str) -> List[str]:
        prompt = (
            "نیچے دیے گئے جواب سے تمام حقائق اور دعوے نکالیں۔\n"
            "ہر دعوہ الگ لائن پر نمبر کے ساتھ لکھیں (1. 2. 3.)\n"
            "صرف فہرست لکھیں۔\n\nجواب:\n" + answer + "\n\nدعووں کی فہرست:"
        )
        text   = self._judge_call(prompt, max_tokens=300)
        claims = []
        for line in text.split("\n"):
            line  = line.strip()
            clean = re.sub(r"^[\d\-\.\)]+\s*", "", line).strip()
            if clean:
                claims.append(clean)
        return claims or [answer]

    def verify_claim(self, claim: str, context: str):
        prompt = (
            "/no_think\n"  # Qwen3-specific: disables thinking mode
            "نیچے دیا گیا دعویٰ متن سے ثابت ہوتا ہے؟\n"
            "صرف ایک لفظ میں جواب دیں: \"ہاں\" یا \"نہیں\"\n\n"
            f"متن:\n{context[:1800]}\n\nدعویٰ: {claim}\n\nجواب:"
        )
        raw = self._judge_call(prompt, max_tokens=100).lower().strip()  # ← 20 → 100
        dbg("verify_claim raw response:", repr(raw))

        if not raw:
            return None
        TRUE_WORDS  = ["ہاں", "جی", "yes", "supported", "true", "han", "بالکل", "درست"]
        FALSE_WORDS = ["نہیں", "نہ", "no", "false", "unsupported", "nahin"]
        if any(w in raw for w in TRUE_WORDS):
            return True
        if any(w in raw for w in FALSE_WORDS):
            return False
        return None
    def compute_faithfulness(self, answer: str, context_chunks: List[Dict]) -> Dict:
        dbg("FAITHFULNESS START")
        dbg("answer preview:", answer[:300])

        ctx_text = " ".join(c["text"] for c in context_chunks)
        dbg("context length:", len(ctx_text))

        claims = self.extract_claims(answer)
        dbg("claims extracted:", claims)

        verified = []
        raw_verifications = []

        for c in claims:
            v = self.verify_claim(c, ctx_text)
            dbg("claim check:", c[:120], "=>", v)
            raw_verifications.append(v)

            if v is not None:
                verified.append(v)

        if not verified:
            return {
                "score": None,
                "claims": claims,
                "verifications": raw_verifications,
                "supported_count": 0,
                "total_claims": 0,
                "error": "Judge returned no usable verification output"
            }

        supported = sum(verified)
        total = len(verified)

        result = {
            "score": round(supported / total, 3),
            "claims": claims,
            "verifications": raw_verifications,
            "supported_count": supported,
            "total_claims": total,
        }
        dbg("faithfulness result:", result)
        dbg("FAITHFULNESS END")
        return result


    def compute_relevancy(self, original_query: str, answer: str) -> Dict:
        dbg("RELEVANCY START")
        dbg("original query:", original_query)
        dbg("answer preview:", answer[:300])

        questions = self.generate_questions_from_answer(answer)
        dbg("generated questions:", questions)

        if not questions:
            return {
                "score": None,
                "generated_questions": [],
                "similarities": [],
                "error": "Judge returned no usable questions"
            }
        q_emb = self.embedding_model.embed_query(original_query)
        q_embs = self.embedding_model.embed(questions)

        sims = [
            round(float(cosine_similarity(q_emb.reshape(1, -1), e.reshape(1, -1))[0][0]), 3)
            for e in q_embs
        ]

        result = {
            "score": round(float(np.mean(sims)), 3) if sims else 0.0,
            "generated_questions": questions,
            "similarities": sims,
        }
        dbg("relevancy result:", result)
        dbg("RELEVANCY END")
        return result

    def generate_questions_from_answer(self, answer: str) -> List[str]:
        prompt = (
            "نیچے دیے گئے جواب سے 3 سوالات بنائیں۔\n"
            "صرف JSON array میں جواب دیں، کوئی اور متن نہ لکھیں۔\n"
            "مثال: [\"سوال 1\", \"سوال 2\", \"سوال 3\"]\n\n"
            "جواب:\n" + answer + "\n\nJSON:"
        )
        raw = self._judge_call(prompt, max_tokens=300, temperature=0.4)
        dbg("generate_questions raw response:", repr(raw))  # ← ADD THIS

        # Try JSON parse first
        try:
            match = re.search(r'\[.*?\]', raw, re.DOTALL)
            if match:
                questions = json.loads(match.group())
                return [q.strip() for q in questions if isinstance(q, str) and len(q.strip()) > 5][:3]
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: line-by-line parsing
        questions = []
        for line in raw.split("\n"):
            clean = re.sub(r"^[\d\.\)\-\s]+", "", line.strip()).strip()
            clean = clean.strip('"').strip("'").strip()
            if clean and len(clean) > 5:
                questions.append(clean)
        return questions[:3]

    

# ─────────────────────────────────────────────
# 12. MAIN RAGPipeline CLASS
# ─────────────────────────────────────────────
class RAGPipeline:
    """
    Full end-to-end Urdu Legal RAG pipeline.

    Typical usage (HF Spaces / production):
        rag = RAGPipeline()
        # Documents must already be indexed in Pinecone + BM25 built from JSON
        rag.load_bm25_from_json("scrapper/cleaned_ocr_output.json")
        result = rag.query("خلع کے لیے کیا کرنا ہوگا؟",
                           conversation_history=[...])

    Ingestion (run once offline):
        rag.ingest_documents(texts, metadata)
    """

    def __init__(self, chunking_strategy: str = "fixed"):
        print("\n" + "="*60)
        print("  Initializing RAG-e-Qanoon Pipeline")
        print("="*60)

        self.chunking_strategy = chunking_strategy
        self.hf_token, pinecone_key = load_api_keys()

        self.embedding_model = EmbeddingModel()
        self.pinecone_db     = PineconeDB(api_key=pinecone_key)
        self.bm25_retriever  = BM25Retriever()
        self.reranker        = Reranker()
        self.llm             = LLMGenerator(self.hf_token)
        self.judge           = LLMJudge(self.hf_token, self.embedding_model)
        self.all_chunks: List[Dict] = []

        print("\nRAG Pipeline initialized.\n")

    # ── BM25 warm-up (call at app startup, after Pinecone is already filled) ──
    def load_bm25_from_json(self, json_path: str,
                             strategy: Optional[str] = None):
        """
        Re-builds the in-memory BM25 index from the cleaned JSON file.
        Call this at startup so the pipeline is ready to serve queries
        without re-ingesting into Pinecone.
        """
        texts, metadata = load_documents_from_json(json_path)
        strategy        = strategy or self.chunking_strategy
        self.all_chunks = chunk_documents(texts, strategy=strategy, metadata=metadata)
        self.bm25_retriever.build_index(self.all_chunks)
        print(f"BM25 index warm-up complete ({len(self.all_chunks)} chunks).")

    # ── Full ingestion (offline, run once) ────────────────────────────────
    def ingest_documents(self, texts: List[str],
                         metadata: Optional[List[Dict]] = None,
                         chunking_strategy: Optional[str] = None):
        """Chunk → Embed → Upsert to Pinecone → Build BM25."""
        strategy = chunking_strategy or self.chunking_strategy
        print(f"\nIngesting {len(texts)} documents (strategy: {strategy})…")

        chunks     = chunk_documents(texts, strategy=strategy, metadata=metadata)
        embeddings = self.embedding_model.embed([c["text"] for c in chunks])
        self.pinecone_db.upsert_chunks(chunks, embeddings)
        self.all_chunks.extend(chunks)
        self.bm25_retriever.build_index(self.all_chunks)
        print(f"Ingestion complete. Total chunks: {len(self.all_chunks)}")

    # ── Retrieval ─────────────────────────────────────────────────────────
    def retrieve(self, query: str, use_reranker: bool = True) -> Tuple[List[Dict], Dict]:
        timings = {}

        dbg("=" * 80)
        dbg("RETRIEVE START")
        dbg("query:", query)

        try:
            t0 = time.time()
            q_emb = self.embedding_model.embed_query(query)
            timings["semantic_embed"] = round(time.time() - t0, 3)
            dbg("query embedding shape:", getattr(q_emb, "shape", None))
        except Exception as e:
            print("[ERROR] embed_query failed:", e)
            traceback.print_exc()
            raise

        try:
            t0 = time.time()
            semantic_hits = self.pinecone_db.semantic_search(q_emb, top_k=TOP_K_RETRIEVAL)
            timings["semantic_search"] = round(time.time() - t0, 3)
            dbg("semantic hits:", len(semantic_hits))
            for i, hit in enumerate(semantic_hits[:3]):
                dbg(f"semantic[{i}] id={hit.get('id')} score={hit.get('score')} text={hit.get('text','')[:120]}")
        except Exception as e:
            print("[ERROR] semantic_search failed:", e)
            traceback.print_exc()
            raise

        try:
            t0 = time.time()
            bm25_hits = self.bm25_retriever.search(query, top_k=TOP_K_RETRIEVAL)
            timings["bm25"] = round(time.time() - t0, 3)
            dbg("bm25 hits:", len(bm25_hits))
            for i, hit in enumerate(bm25_hits[:3]):
                dbg(f"bm25[{i}] id={hit.get('id')} score={hit.get('score')} text={hit.get('text','')[:120]}")
        except Exception as e:
            print("[ERROR] bm25 search failed:", e)
            traceback.print_exc()
            raise

        try:
            t0 = time.time()
            fused = reciprocal_rank_fusion(semantic_hits, bm25_hits)
            timings["rrf"] = round(time.time() - t0, 3)
            dbg("fused hits:", len(fused))
            for i, hit in enumerate(fused[:5]):
                dbg(f"fused[{i}] id={hit.get('id')} rrf={hit.get('rrf_score')} text={hit.get('text','')[:120]}")
        except Exception as e:
            print("[ERROR] RRF failed:", e)
            traceback.print_exc()
            raise

        try:
            if use_reranker and fused:
                t0 = time.time()
                final_chunks = self.reranker.rerank(query, fused, top_k=TOP_K_FINAL)
                timings["rerank"] = round(time.time() - t0, 3)
                dbg("reranked final chunks:", len(final_chunks))
                for i, hit in enumerate(final_chunks):
                    dbg(f"final[{i}] id={hit.get('id')} rerank={hit.get('rerank_score')} text={hit.get('text','')[:120]}")
            else:
                final_chunks = fused[:TOP_K_FINAL]
                dbg("reranker skipped")
        except Exception as e:
            print("[ERROR] reranker failed:", e)
            traceback.print_exc()
            raise

        timings["total_retrieval"] = round(sum(timings.values()), 3)
        dbg("retrieval timings:", timings)
        dbg("RETRIEVE END")
        dbg("=" * 80)

        return final_chunks, timings
    # ── Main query entry-point ────────────────────────────────────────────
    def query(self, user_query: str,
          conversation_history: Optional[List[Dict]] = None,
          run_evaluation: bool = True,
          use_reranker: bool = True) -> Dict:

        if conversation_history is None:
            conversation_history = []

        print(f"\nQuery: {user_query}")
        total_start = time.time()

        retrieved_chunks, retrieval_timings = self.retrieve(
            user_query, use_reranker=use_reranker
        )
        print(f"  Retrieved {len(retrieved_chunks)} chunks")

        t0 = time.time()
        answer, _ = self.llm.generate(
            user_query,
            retrieved_chunks,
            conversation_history=conversation_history,
        )
        generation_time = round(time.time() - t0, 3)
        print(f"  Answer generated ({generation_time}s)")
        dbg("final answer preview:", answer[:500])

        faithfulness_result = {"score": None}
        relevancy_result = {"score": None}

        if run_evaluation and retrieved_chunks:
            print("  Running LLM-as-a-Judge…")
            try:
                faithfulness_result = self.judge.compute_faithfulness(answer, retrieved_chunks)
                relevancy_result = self.judge.compute_relevancy(user_query, answer)
                f_score = faithfulness_result.get("score")
                r_score = relevancy_result.get("score")

                f_text = f"{f_score:.2%}" if isinstance(f_score, float) else "N/A"
                r_text = f"{r_score:.2%}" if isinstance(r_score, float) else "N/A"

                print(f"  Faithfulness: {f_text}  Relevancy: {r_text}")
            except Exception as e:
                print("[ERROR] evaluation failed:", e)
                traceback.print_exc()
                faithfulness_result = {"score": None, "error": str(e)}
                relevancy_result = {"score": None, "error": str(e)}

        return {
            "query": user_query,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "faithfulness": faithfulness_result,
            "relevancy": relevancy_result,
            "timings": {
                **retrieval_timings,
                "generation": generation_time,
                "total": round(time.time() - total_start, 3),
            },
        }
    def get_pinecone_stats(self) -> Dict:
        return self.pinecone_db.get_stats()


# ─────────────────────────────────────────────
# 13. CLI ENTRY POINT  (python rag_pipeline.py)
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# 13. CLI ENTRY POINT  (python rag_pipeline.py)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import os
    from dotenv import load_dotenv

    # Load local .env file so it finds your keys
    # This looks one folder up for the .env file
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

    # Dynamically find the scrapper JSON file one folder up
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_json = os.path.join(base_dir, "scrapper", "cleaned_ocr_output.json")

    parser = argparse.ArgumentParser(description="RAG-e-Qanoon CLI")
    parser.add_argument("--json", default=default_json)
    parser.add_argument("--query", default="پاکستان کا ریاستی مذہب کیا ہے؟")
    parser.add_argument("--ingest", action="store_true",
                        help="Re-ingest documents into Pinecone (run once).")
    args = parser.parse_args()

    rag = RAGPipeline(chunking_strategy="fixed")
    texts, metadata = load_documents_from_json(args.json)

    if args.ingest:
        rag.ingest_documents(texts, metadata=metadata)
    else:
        rag.load_bm25_from_json(args.json)

    result = rag.query(args.query, run_evaluation=False)

    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    print(f"Question : {result['query']}")
    print(f"\nAnswer   :\n{result['answer']}")
    if result["faithfulness"]["score"] is not None:
        print(f"\nFaithfulness: {result['faithfulness']['score']:.2%}")
        print(f"Relevancy   : {result['relevancy']['score']:.2%}")
    print(f"Total time: {result['timings']['total']}s")
    print(f"\nPinecone stats: {rag.get_pinecone_stats()}")