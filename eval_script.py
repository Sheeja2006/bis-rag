"""
eval_script.py — BIS Standards Recommendation Engine
Computes: Hit Rate @3, MRR @5, Average Latency

Usage:
    python eval_script.py --results team_results.json --ground_truth ground_truth.json

Input formats:

  ground_truth.json:
  [
    {"id": "1", "query": "ordinary Portland cement", "expected_standards": ["IS 269", "IS 8112"]},
    ...
  ]

  team_results.json (output of inference.py):
  [
    {"id": "1", "retrieved_standards": ["IS 269", "IS 4031", "IS 8112"], "latency_seconds": 0.08},
    ...
  ]
"""

import json
import argparse
import sys


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found — {path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {path} — {e}")
        sys.exit(1)


def compute_hit_rate_at_k(retrieved, expected, k=3):
    """
    Hit Rate @K:
    1 if at least one expected standard appears in top-K retrieved results.
    0 otherwise.
    """
    top_k = retrieved[:k]
    for std in expected:
        if std in top_k:
            return 1
    return 0


def compute_reciprocal_rank_at_k(retrieved, expected, k=5):
    """
    Reciprocal Rank @K:
    1/rank of the first expected standard found in top-K results.
    0 if none found within top-K.
    """
    top_k = retrieved[:k]
    for rank, std in enumerate(top_k, start=1):
        if std in expected:
            return 1.0 / rank
    return 0.0


def normalise_id(std_id):
    """
    Normalise IS number format for fair comparison.
    e.g. 'IS269', 'IS-269', 'IS:269', 'is 269' all -> 'IS 269'
    """
    import re
    std_id = std_id.strip().upper()
    std_id = re.sub(r'IS[\s:\-]*', 'IS ', std_id)
    return std_id.strip()


def evaluate(results, ground_truth):
    gt_lookup = {str(item["id"]): item for item in ground_truth}

    hit_rates         = []
    reciprocal_ranks  = []
    latencies         = []
    missing_ids       = []
    missing_latency   = []
    detail_rows       = []

    for result in results:
        rid = str(result["id"])

        if rid not in gt_lookup:
            missing_ids.append(rid)
            continue

        gt_item = gt_lookup[rid]

        retrieved = [normalise_id(s) for s in result.get("retrieved_standards", [])]
        expected  = [normalise_id(s) for s in gt_item.get("expected_standards", [])]

        # Warn when latency is absent rather than silently defaulting to 0
        if "latency_seconds" not in result:
            missing_latency.append(rid)
        latency = result.get("latency_seconds", None)

        hr = compute_hit_rate_at_k(retrieved, expected, k=3)
        rr = compute_reciprocal_rank_at_k(retrieved, expected, k=5)

        hit_rates.append(hr)
        reciprocal_ranks.append(rr)
        if latency is not None:
            latencies.append(latency)

        detail_rows.append({
            "id":        rid,
            "query":     gt_item.get("query", "")[:55],
            "hit@3":     hr,
            "rr@5":      round(rr, 3),
            "latency":   round(latency, 3) if latency is not None else "N/A",
            "retrieved": retrieved[:5],
            "expected":  expected,
        })

    if missing_latency:
        print(f"  WARNING: latency_seconds missing for IDs: {missing_latency} — excluded from avg latency.")

    if not hit_rates:
        print("ERROR: No matching IDs found between results and ground truth.")
        sys.exit(1)

    n              = len(hit_rates)
    hit_rate_score = round(sum(hit_rates) / n * 100, 2)
    mrr_score      = round(sum(reciprocal_ranks) / n, 4)
    avg_latency    = round(sum(latencies) / len(latencies), 4) if latencies else None

    return {
        "n_queries":      n,
        "hit_rate_at_3":  hit_rate_score,
        "mrr_at_5":       mrr_score,
        "avg_latency_s":  avg_latency,
        "missing_ids":    missing_ids,
        "detail":         detail_rows,
    }


def print_report(metrics):
    n   = metrics["n_queries"]
    hr  = metrics["hit_rate_at_3"]
    mrr = metrics["mrr_at_5"]
    lat = metrics["avg_latency_s"]

    # FIX: label now matches the threshold (>= 80, not > 80)
    hr_status  = "PASS" if hr  >= 80   else "FAIL"
    mrr_status = "PASS" if mrr >= 0.7  else "FAIL"
    lat_status = ("PASS" if lat <= 5.0 else "FAIL") if lat is not None else "N/A"

    lat_display = f"{lat}s" if lat is not None else "N/A"

    sep = "-" * 52
    print()
    print("=" * 52)
    print("  BIS RAG — Evaluation Report")
    print("=" * 52)
    print(f"  Queries evaluated : {n}")
    print(sep)
    print(f"  Hit Rate @3       : {hr}%   (target >=80%)  [{hr_status}]")
    print(f"  MRR @5            : {mrr}    (target >=0.7)  [{mrr_status}]")
    print(f"  Avg Latency       : {lat_display:<8} (target <=5s)   [{lat_status}]")
    print(sep)

    passed = sum(1 for s in [hr_status, mrr_status, lat_status] if s == "PASS")
    print(f"  Metrics passed    : {passed}/3")
    print("=" * 52)

    if metrics["missing_ids"]:
        print(f"\n  WARNING: {len(metrics['missing_ids'])} result IDs not found in ground truth:")
        print(f"  {metrics['missing_ids']}")

    print("\n  Per-query breakdown:")
    print(f"  {'ID':<5} {'Hit@3':<7} {'RR@5':<7} {'Latency':<10} Query")
    print("  " + "-" * 60)
    for row in metrics["detail"]:
        hit_mark = "Y" if row["hit@3"] else "N"
        lat_str  = str(row["latency"])
        print(f"  {row['id']:<5} {hit_mark:<7} {row['rr@5']:<7} {lat_str:<10} {row['query']}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate BIS RAG results: Hit Rate @3, MRR @5, Latency"
    )
    parser.add_argument("--results",      required=True,
                        help="Path to inference output JSON (team_results.json)")
    parser.add_argument("--ground_truth", required=True,
                        help="Path to ground truth JSON with expected_standards")
    parser.add_argument("--output",       default=None,
                        help="Optional: save metrics to this JSON file")
    args = parser.parse_args()

    results      = load_json(args.results)
    ground_truth = load_json(args.ground_truth)

    metrics = evaluate(results, ground_truth)
    print_report(metrics)

    if args.output:
        summary = {
            "n_queries":     metrics["n_queries"],
            "hit_rate_at_3": metrics["hit_rate_at_3"],
            "mrr_at_5":      metrics["mrr_at_5"],
            "avg_latency_s": metrics["avg_latency_s"],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"  Metrics saved to: {args.output}\n")


if __name__ == "__main__":
    main()