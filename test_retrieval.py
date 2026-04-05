from rag_backend.rag_pipeline import RAGPipeline
import json, os

json_path = os.path.join("scrapper", "cleaned_ocr_output.json")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
texts = [item["text"] for item in data]
metadata = [item["meta"] for item in data]

rag = RAGPipeline(model_choice="qwen")
rag.ingest_documents(texts, metadata)

queries = [
    "پاکستان کا ریاستی مذہب کیا ہے؟",
    "گرفتاری کے وقت میرے کیا حقوق ہیں؟",
    "شہریوں کے بنیادی حقوق کون سے ہیں؟",
]

for q in queries:
    result = rag.query(q, run_evaluation=True)
    print(f"\n=== {q} ===")
    print(f"Answer: {result['answer']}")
    print(f"Faithfulness: {result['faithfulness']['score']:.2%}")
    print(f"Relevancy: {result['relevancy']['score']:.2%}")