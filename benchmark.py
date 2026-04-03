#!/usr/bin/env python3
"""
benchmark.py — Urdu Legal RAG Benchmarking Suite
=================================================
Tests one variable at a time across 9 experiment groups and writes every
result row to benchmark_results.csv as it goes, so you never lose progress
if a run crashes midway.

Directory assumption
--------------------
Place this file in the SAME folder as your pipeline .py file.
If the file is named differently, update PIPELINE_MODULE below.

Usage
-----
    python benchmark.py

To run only specific groups, set the others to False in RUN_GROUPS.
"""

import os, sys, csv, json, time, types, importlib
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
#  0. POINT TO YOUR PIPELINE FILE
# ═══════════════════════════════════════════════════════════════════════════
import rag_pipeline as P

# ═══════════════════════════════════════════════════════════════════════════
#  1. USER SETTINGS — edit these before running
# ═══════════════════════════════════════════════════════════════════════════

# Path to your cleaned JSON data file
JSON_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "scrapper", "cleaned_ocr_output.json"
)

# Where results will be saved
CSV_OUTPUT = "benchmark_results.csv"

# Queries to test with. 5–10 is a good balance of coverage vs runtime.
# Replace with queries that are actually answerable from your documents.
TEST_QUERIES: List[str] = [
    "زمین کی خرید و فروخت کے لیے کون سے دستاویزات ضروری ہیں؟",
    "کے افسران اور عملے کی معلومات فراہم کریں IBCC",
    "جرم زنا آرڈیننس 1979 کی دستاویز میں موجود تعریفات کے بارے میں تفصیلی معلومات فراہم کریں",


    # "پاکستان کا ریاستی مذہب کیا ہے؟",
    # "گرفتاری کے وقت میرے کیا حقوق ہیں؟",
    # "خلع کے لیے عدالت میں کیا کرنا ہوگا؟",
    # "CNIC گم ہو جائے تو کیا کریں؟",

    # "اعلی حکومتی ملازم سے کون شخص مراد ہے ؟",
    # "گن و کنٹری کلب کا پیٹرن انچیف کون ہوگا ؟",
    # "خواتین کو مقام کار پر ہراساں کرنے کے خلاف تحفظ ایکٹ پر روشنی ڈالیں",
]

# Starting LLM for all groups except llm_model group
DEFAULT_MODEL = "qwen"

# Set False to skip a group (saves time while iterating)
RUN_GROUPS = {
    "baseline":          True,   # run first — establishes reference numbers
    "top_k_final":       True,   # how many chunks the LLM sees
    "rrf_weights":       True,   # semantic vs BM25 weight ratio
    "reranker_onoff":    True,   # CrossEncoder on vs off
    "temperature":       True,   # LLM sampling temperature
    "prompt_template":   True,   # different system prompts (biggest lever)
    "llm_model":         True,   # qwen vs aya
    "chunking_strategy": True,   # fixed / recursive / sentence  (slow: re-ingests)
    "chunk_size":        True,   # size + overlap combos         (slow: re-ingests)
}


# ═══════════════════════════════════════════════════════════════════════════
#  2. PROMPT TEMPLATES
#     Each template is a callable: (query: str, chunks: List[Dict]) -> str
#     None means "use whatever is already in pipeline.py"
# ═══════════════════════════════════════════════════════════════════════════

def _fmt_chunks(chunks: List[Dict]) -> str:
    return "\n".join(
        f"[دستاویز {i+1}]:\n{c['text']}" for i, c in enumerate(chunks)
    )

PROMPT_TEMPLATES: Dict[str, Optional[callable]] = {

    # Baseline — whatever is in pipeline.py
    "pipeline_default": None,

    # Explicit grounding: never guess, always cite
    "strict_grounding": lambda q, chunks: (
        "آپ ایک پاکستانی قانونی مشیر ہیں۔\n"
        "ہدایت: صرف نیچے دی گئی دستاویزات کی بنیاد پر جواب دیں۔ "
        "اگر کوئی معلومات دستاویزات میں نہ ہو تو لکھیں: 'یہ معلومات دستاویزات میں موجود نہیں ہے۔'\n"
        "اپنی طرف سے کوئی اضافہ نہ کریں۔\n\n"
        "=== قانونی دستاویزات ===\n"
        f"{_fmt_chunks(chunks)}\n"
        "=== ختم ===\n\n"
        f"سوال: {q}\n\n"
        "جواب (صرف دستاویزات کی بنیاد پر، غیر متعلقہ معلومات شامل نہ کریں):"
    ),

    # Chain of thought: step-by-step reasoning before answer
    "chain_of_thought": lambda q, chunks: (
        "آپ ایک ماہر پاکستانی قانونی مشیر ہیں۔\n"
        "ہدایت: پہلے دستاویزات کا تجزیہ کریں، پھر مرحلہ وار جواب دیں۔\n\n"
        "=== قانونی دستاویزات ===\n"
        f"{_fmt_chunks(chunks)}\n"
        "=== ختم ===\n\n"
        f"سوال: {q}\n\n"
        "مرحلہ 1 — متعلقہ قانون یا دفعہ کی شناخت:\n"
        "مرحلہ 2 — قانونی تقاضے اور شرائط:\n"
        "مرحلہ 3 — عملی طریقہ کار:\n"
        "حتمی جواب:"
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
#  3. CSV COLUMNS
# ═══════════════════════════════════════════════════════════════════════════
COLUMNS = [
    "timestamp", "experiment_group", "config_name", "config_details",
    "query_idx", "query",
    "faithfulness", "relevancy",
    "retrieval_latency_s", "generation_latency_s", "total_latency_s",
    "n_retrieved_chunks", "answer_snippet",
]


# ═══════════════════════════════════════════════════════════════════════════
#  4. PATCH HELPERS
#     These modify pipeline behaviour without touching pipeline.py.
#     All patches are reversible.
# ═══════════════════════════════════════════════════════════════════════════

# ── 4a. Pinecone index clear (needed between chunking experiments) ─────────
def clear_pinecone_index(rag: "P.RAGPipeline", wait: int = 6):
    """Delete all vectors from the Pinecone index and reset local state."""
    print("    Clearing Pinecone index…")
    rag.pinecone_db.index.delete(delete_all=True)
    time.sleep(wait)   # let Pinecone settle
    rag.all_chunks = []
    rag.bm25_retriever.bm25   = None
    rag.bm25_retriever.chunks = []
    print("    Pinecone index cleared.")


# ── 4b. LLM temperature ───────────────────────────────────────────────────
def set_temperature(rag: "P.RAGPipeline", temp: float):
    """Monkey-patch rag.llm.generate to use a specific temperature."""
    _client       = rag.llm.client
    _build_prompt = rag.llm._build_generation_prompt

    def _patched(query, context_chunks, max_new_tokens=400):
        prompt = _build_prompt(query, context_chunks)
        start  = time.time()
        try:
            resp   = _client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temp,
            )
            answer = resp.choices[0].message.content.strip()
        except Exception as e:
            answer = f"[LLM Error: {e}]"
        return answer, round(time.time() - start, 2)

    rag.llm.generate = _patched


def reset_temperature(rag: "P.RAGPipeline"):
    """Restore the original generate method (bound to instance)."""
    rag.llm.generate = types.MethodType(P.LLMGenerator.generate, rag.llm)


# ── 4c. Prompt template ───────────────────────────────────────────────────
def set_prompt(rag: "P.RAGPipeline", template_fn: Optional[callable]):
    if template_fn is None:
        reset_prompt(rag)
    else:
        # Assign directly to instance dict — Python won't bind it as a method,
        # so self._build_generation_prompt(q, chunks) calls template_fn(q, chunks).
        rag.llm._build_generation_prompt = template_fn


def reset_prompt(rag: "P.RAGPipeline"):
    # Remove instance-level override so class method takes over again
    if "_build_generation_prompt" in rag.llm.__dict__:
        del rag.llm.__dict__["_build_generation_prompt"]


# ── 4d. TOP_K_FINAL (chunks fed to LLM) ──────────────────────────────────
# pipeline.py's retrieve() looks up TOP_K_FINAL in its module globals at
# call time, so patching P.TOP_K_FINAL is sufficient.
def set_top_k(k: int):
    P.TOP_K_FINAL = k

def reset_top_k():
    P.TOP_K_FINAL = 5


# ── 4e. RRF weights ───────────────────────────────────────────────────────
# retrieve() calls P.reciprocal_rank_fusion from module globals, so
# replacing P.reciprocal_rank_fusion is sufficient.
_ORIG_RRF = None   # filled in main()

def set_rrf_weights(sem: float, bm: float):
    def _patched(semantic_hits, bm25_hits, k=60,
                 semantic_weight=sem, bm25_weight=bm):
        return _ORIG_RRF(semantic_hits, bm25_hits,
                         k=k, semantic_weight=semantic_weight,
                         bm25_weight=bm25_weight)
    P.reciprocal_rank_fusion = _patched

def reset_rrf():
    P.reciprocal_rank_fusion = _ORIG_RRF


# ── 4f. Chunk size (fixed chunking) ───────────────────────────────────────
# chunk_fixed uses DEFAULT PARAMETER values captured at definition time,
# so we must replace the function itself rather than patching constants.
_ORIG_CHUNK_FIXED = None   # filled in main()

def set_chunk_fixed(size: int, overlap: int):
    P.chunk_fixed = lambda text: _ORIG_CHUNK_FIXED(
        text, chunk_size=size, overlap=overlap
    )

def reset_chunk_fixed():
    P.chunk_fixed = _ORIG_CHUNK_FIXED


# ═══════════════════════════════════════════════════════════════════════════
#  5. CORE QUERY RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_single_query(
    rag:           "P.RAGPipeline",
    query:         str,
    query_idx:     int,
    group:         str,
    config_name:   str,
    config_details: str,
    use_reranker:  bool = True,
) -> Dict:
    """Run one query end-to-end and return a flat dict ready for the CSV."""
    try:
        t_start = time.time()

        # Retrieval (semantic + BM25 + RRF + optional reranker)
        retrieved, timings = rag.retrieve(query, use_reranker=use_reranker)

        # Generation
        t_gen  = time.time()
        answer, _ = rag.llm.generate(query, retrieved)
        gen_lat = round(time.time() - t_gen, 3)

        # LLM-as-a-Judge evaluation
        faith = rag.judge.compute_faithfulness(answer, retrieved)
        rel   = rag.judge.compute_relevancy(query, answer)

        total_lat = round(time.time() - t_start, 3)

        return {
            "timestamp":            datetime.now().isoformat(timespec="seconds"),
            "experiment_group":     group,
            "config_name":          config_name,
            "config_details":       config_details,
            "query_idx":            query_idx,
            "query":                query,
            "faithfulness":         faith.get("score", ""),
            "relevancy":            rel.get("score", ""),
            "retrieval_latency_s":  timings.get("total_retrieval", ""),
            "generation_latency_s": gen_lat,
            "total_latency_s":      total_lat,
            "n_retrieved_chunks":   len(retrieved),
            "answer_snippet":       answer[:150].replace("\n", " "),
        }

    except Exception as exc:
        print(f"      ⚠  Query error: {exc}")
        return {
            "timestamp":            datetime.now().isoformat(timespec="seconds"),
            "experiment_group":     group,
            "config_name":          config_name,
            "config_details":       config_details,
            "query_idx":            query_idx,
            "query":                query,
            "faithfulness":         "ERROR",
            "relevancy":            "ERROR",
            "retrieval_latency_s":  "",
            "generation_latency_s": "",
            "total_latency_s":      "",
            "n_retrieved_chunks":   0,
            "answer_snippet":       str(exc)[:150],
        }


def run_config(
    rag:           "P.RAGPipeline",
    queries:       List[str],
    group:         str,
    config_name:   str,
    config_details: str,
    writer:        csv.DictWriter,
    csvfile,
    use_reranker:  bool = True,
):
    """Run all queries for one configuration and flush results to CSV."""
    print(f"\n  ► [{group}] {config_name}")
    print(f"    {config_details}")
    for i, q in enumerate(queries):
        print(f"    Query {i+1}/{len(queries)}: {q[:55]}…")
        row = run_single_query(rag, q, i, group, config_name,
                               config_details, use_reranker)
        writer.writerow(row)
        csvfile.flush()   # write immediately so partial runs are saved
        faith = row["faithfulness"]
        rel   = row["relevancy"]
        faith_str = f"{faith:.2%}" if isinstance(faith, float) else str(faith)
        rel_str   = f"{rel:.2%}"   if isinstance(rel,   float) else str(rel)
        print(f"      Faith={faith_str}  Rel={rel_str}  "
              f"Time={row['total_latency_s']}s")


# ═══════════════════════════════════════════════════════════════════════════
#  6. EXPERIMENT GROUP FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _header(title: str):
    print("\n" + "═" * 62)
    print(f"  GROUP: {title}")
    print("═" * 62)


# ── Baseline ──────────────────────────────────────────────────────────────
def exp_baseline(rag, texts, metadata, queries, writer, csvfile):
    _header("baseline")
    # Always start with a fresh index for reproducibility
    clear_pinecone_index(rag)
    rag.ingest_documents(texts, metadata=metadata, chunking_strategy="recursive")
    run_config(rag, queries, "baseline", "baseline",
               "recursive+hybrid+rerank+qwen+temp0.3+topk5",
               writer, csvfile)


# ── TOP_K_FINAL ───────────────────────────────────────────────────────────
def exp_top_k_final(rag, queries, writer, csvfile):
    """How many chunks the LLM actually sees. No re-ingestion needed."""
    _header("top_k_final")
    for k in [3, 5, 10]:
        set_top_k(k)
        run_config(rag, queries, "top_k_final", f"topk_{k}",
                   f"TOP_K_FINAL={k}", writer, csvfile)
    reset_top_k()


# ── RRF Weights ───────────────────────────────────────────────────────────
def exp_rrf_weights(rag, queries, writer, csvfile):
    """Semantic vs BM25 weight in Reciprocal Rank Fusion. No re-ingestion."""
    _header("rrf_weights")
    configs = [
        (0.7, 0.3, "sem0.7_bm0.3"),
        (0.6, 0.4, "sem0.6_bm0.4"),   # pipeline default
        (0.5, 0.5, "sem0.5_bm0.5"),
    ]
    for sem, bm, name in configs:
        set_rrf_weights(sem, bm)
        run_config(rag, queries, "rrf_weights", name,
                   f"semantic={sem} bm25={bm}", writer, csvfile)
    reset_rrf()


# ── Reranker On/Off ───────────────────────────────────────────────────────
def exp_reranker(rag, queries, writer, csvfile):
    """CrossEncoder re-ranker on vs off. No re-ingestion."""
    _header("reranker_onoff")
    for use_rr, name in [(True, "reranker_ON"), (False, "reranker_OFF")]:
        run_config(rag, queries, "reranker_onoff", name,
                   f"use_reranker={use_rr}", writer, csvfile,
                   use_reranker=use_rr)


# ── LLM Temperature ───────────────────────────────────────────────────────
def exp_temperature(rag, queries, writer, csvfile):
    """Different LLM sampling temperatures. No re-ingestion."""
    _header("temperature")
    for temp in [0.0, 0.1, 0.2, 0.3]:
        set_temperature(rag, temp)
        run_config(rag, queries, "temperature", f"temp_{temp}",
                   f"temperature={temp}", writer, csvfile)
    reset_temperature(rag)


# ── Prompt Templates ──────────────────────────────────────────────────────
def exp_prompt_template(rag, queries, writer, csvfile):
    """Different prompt designs. No re-ingestion. Usually the biggest lever."""
    _header("prompt_template")
    for name, fn in PROMPT_TEMPLATES.items():
        set_prompt(rag, fn)
        run_config(rag, queries, "prompt_template", name,
                   f"template={name}", writer, csvfile)
    reset_prompt(rag)


# ── LLM Model ─────────────────────────────────────────────────────────────
def exp_llm_model(rag, queries, writer, csvfile):
    """qwen vs aya. No re-ingestion."""
    _header("llm_model")
    for model_key, model_name in P.LLM_MODELS.items():
        rag.switch_model(model_key)
        run_config(rag, queries, "llm_model", model_key,
                   f"model={model_name}", writer, csvfile)
    rag.switch_model(DEFAULT_MODEL)


# ── Chunking Strategy ─────────────────────────────────────────────────────
def exp_chunking_strategy(rag, texts, metadata, queries, writer, csvfile):
    """fixed / recursive / sentence.  Requires full re-ingestion each time."""
    _header("chunking_strategy")
    for strategy in ["fixed", "recursive", "sentence"]:
        clear_pinecone_index(rag)
        rag.ingest_documents(texts, metadata=metadata,
                             chunking_strategy=strategy)
        run_config(rag, queries, "chunking_strategy", strategy,
                   f"strategy={strategy}", writer, csvfile)


# ── Chunk Size ────────────────────────────────────────────────────────────
def exp_chunk_size(rag, texts, metadata, queries, writer, csvfile):
    """Different size+overlap combos with fixed chunking. Re-ingests each time."""
    _header("chunk_size")
    configs = [
        (256,  40, "size256_ov40"),
        (512,  80, "size512_ov80"),
        (1024, 120, "size1024_ov120"),
    ]
    for size, overlap, name in configs:
        set_chunk_fixed(size, overlap)
        clear_pinecone_index(rag)
        rag.ingest_documents(texts, metadata=metadata,
                             chunking_strategy="fixed")
        run_config(rag, queries, "chunk_size", name,
                   f"chunk_size={size} overlap={overlap}", writer, csvfile)
    reset_chunk_fixed()


# ═══════════════════════════════════════════════════════════════════════════
#  7. SUMMARY PRINTER
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(csv_path: str):
    """Read the CSV and print per-config averages sorted by faithfulness."""
    try:
        import csv as _csv
        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                rows.append(row)

        # Group by (group, config_name)
        from collections import defaultdict
        buckets: Dict[Tuple, List] = defaultdict(list)
        for row in rows:
            key = (row["experiment_group"], row["config_name"])
            try:
                faith = float(row["faithfulness"])
                rel   = float(row["relevancy"])
                buckets[key].append((faith, rel))
            except (ValueError, TypeError):
                pass

        print("\n" + "═" * 70)
        print("  BENCHMARK SUMMARY  (averaged over all queries per config)")
        print("═" * 70)
        print(f"{'Group':<22} {'Config':<22} {'Faithful':>9} {'Relevancy':>10} {'N':>4}")
        print("─" * 70)

        results = []
        for (group, config), vals in buckets.items():
            if vals:
                avg_f = np.mean([v[0] for v in vals])
                avg_r = np.mean([v[1] for v in vals])
                results.append((group, config, avg_f, avg_r, len(vals)))

        results.sort(key=lambda x: x[2], reverse=True)
        for group, config, avg_f, avg_r, n in results:
            print(f"{group:<22} {config:<22} {avg_f:>9.2%} {avg_r:>10.2%} {n:>4}")

        print("═" * 70)
        print(f"Full results saved to: {csv_path}\n")
    except Exception as e:
        print(f"Summary error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  8. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global _ORIG_RRF, _ORIG_CHUNK_FIXED

    print("=" * 62)
    print("  Urdu Legal RAG — Benchmarking Suite")
    print("=" * 62)

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"\nLoading documents from:\n  {JSON_PATH}")
    texts, metadata = P.load_documents_from_json(JSON_PATH)
    print(f"Loaded {len(texts)} documents.\n")

    # ── Initialise RAG pipeline ───────────────────────────────────────────
    print("Initialising RAG pipeline (loading models, connecting to Pinecone)…")
    rag = P.RAGPipeline(model_choice=DEFAULT_MODEL, chunking_strategy="recursive")

    # ── Store originals before any patching ───────────────────────────────
    _ORIG_RRF         = P.reciprocal_rank_fusion
    _ORIG_CHUNK_FIXED = P.chunk_fixed

    # ── Open CSV ──────────────────────────────────────────────────────────
    print(f"\nResults will stream to: {CSV_OUTPUT}\n")
    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=COLUMNS)
        writer.writeheader()
        csvfile.flush()

        # ── Run groups ────────────────────────────────────────────────────
        # Order matters: run non-re-ingestion groups first (fast),
        # re-ingestion groups last (slow).

        if RUN_GROUPS["baseline"]:
            exp_baseline(rag, texts, metadata, TEST_QUERIES, writer, csvfile)

        # After baseline, Pinecone has recursive chunks — reuse for fast groups.
        if RUN_GROUPS["top_k_final"]:
            exp_top_k_final(rag, TEST_QUERIES, writer, csvfile)

        if RUN_GROUPS["rrf_weights"]:
            exp_rrf_weights(rag, TEST_QUERIES, writer, csvfile)

        if RUN_GROUPS["reranker_onoff"]:
            exp_reranker(rag, TEST_QUERIES, writer, csvfile)

        if RUN_GROUPS["temperature"]:
            exp_temperature(rag, TEST_QUERIES, writer, csvfile)

        if RUN_GROUPS["prompt_template"]:
            exp_prompt_template(rag, TEST_QUERIES, writer, csvfile)

        if RUN_GROUPS["llm_model"]:
            exp_llm_model(rag, TEST_QUERIES, writer, csvfile)

        # Re-ingestion groups (slowest — put last)
        if RUN_GROUPS["chunking_strategy"]:
            exp_chunking_strategy(rag, texts, metadata, TEST_QUERIES, writer, csvfile)

        if RUN_GROUPS["chunk_size"]:
            exp_chunk_size(rag, texts, metadata, TEST_QUERIES, writer, csvfile)

    print_summary(CSV_OUTPUT)
    print("  Benchmarking complete.")


if __name__ == "__main__":
    main()
