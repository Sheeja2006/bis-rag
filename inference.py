import json
import time
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.retriever import retrieve, get_model, get_collection

def run_inference(input_path, output_path):
    with open(input_path, encoding="utf-8") as f:
        queries = json.load(f)

    # ✅ Warm up — load model and vectorstore ONCE before queries
    print("⏳ Loading model and vectorstore...")
    get_model()
    get_collection()
    print("✅ Ready!\n")

    results = []
    total = len(queries)
    print(f"🔍 Running inference on {total} queries...\n")

    for i, item in enumerate(queries):
        start = time.time()

        query = item["query"]
        retrieved = retrieve(query, top_k=5)
        standard_ids = [r["standard_id"] for r in retrieved]

        latency = time.time() - start

        results.append({
            "id": item["id"],
            "retrieved_standards": standard_ids,
            "latency_seconds": round(latency, 4)
        })

        print(f"  [{i+1}/{total}] {query[:50]}")
        print(f"           → {standard_ids}")
        print(f"           ⏱  {latency:.2f}s\n")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    avg_latency = sum(r["latency_seconds"] for r in results) / len(results)
    print(f"✅ Done! Saved to {output_path}")
    print(f"📊 Avg latency: {avg_latency:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_inference(args.input, args.output)