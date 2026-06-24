import json
from collections import defaultdict
from typing import Any

from config_reader import load_config
from dedupe import semantic_dedupe
from document_reader import process_document
from file_loader import discover_files
from model_loader import load_model_tokenizer

cfg1 = load_config("config/pipeline_config.yaml")
cfg2 = load_config("config/model_config.yaml")
cfg = cfg1 | cfg2
DEFAULT_MODEL = cfg.get("default_model", "meta-llama/Llama-3.2-1B")
SEMANTIC_DEDUPE_THRESHOLD = cfg.get("semantic_dedupe_threshold", 0.88)


def run_pipeline(input_path: str, out_file: str):
    files = discover_files(input_path)
    print(f"[INFO] Found {len(files)} files")

    print(
        "[INFO] Loading model and tokenizer (device_map='auto') — this may take a minute..."
    )

    tokenizer, model = load_model_tokenizer(DEFAULT_MODEL)
    all_results = []
    report: dict[str, Any] = {
        "files_found": len(files),
        "files_processed": 0,
        "files_failed": 0,
        "qas_per_document": {},
    }

    for f in files:
        print(f"[INFO] Processing {f.name}")
        results, status = process_document(f, tokenizer, model)

        if status["status"] == "success":
            report["files_processed"] += 1
            report["qas_per_document"][f.name] = status["qas"]
            all_results.extend(results)
        else:
            report["files_failed"] += 1
            print(f"[WARN] Failed {f.name}: {status['error']}")

    # print("[INFO] Running semantic dedupe per document...")
    # # optional: semantic_dedupe here per doc_id

    # print(f"[INFO] Raw generated Q/A count before semantic dedupe: {len(all_results)}")

    # # Semantic dedupe (embedding-based) to remove near-duplicates
    # results = semantic_dedupe(all_results, threshold=SEMANTIC_DEDUPE_THRESHOLD)
    # print(f"[INFO] Q/A count after semantic dedupe: {len(all_results)}")

    print("[INFO] Running semantic dedupe per document...")

    docs = defaultdict(list)
    for qa in all_results:
        docs[qa["doc_id"]].append(qa)

    deduped_results = []
    for _, qas in docs.items():
        deduped = semantic_dedupe(qas, threshold=SEMANTIC_DEDUPE_THRESHOLD)
        deduped_results.extend(deduped)

    print(
        f"[INFO] Q/A count after per-document semantic dedupe: {len(deduped_results)}"
    )

    # Optional global exact dedupe
    final_results = []
    seen = set()
    for qa in deduped_results:
        key = (qa["question"].lower(), qa["answer"].lower())
        if key in seen:
            continue
        seen.add(key)
        final_results.append(qa)

    print(f"[INFO] Final Q/A count after global exact dedupe: {len(final_results)}")

    all_results = final_results

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved final dataset -> {out_file}")

    report["total_qas"] = len(all_results)

    report_name = "processing_report.json"
    with open(report_name, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] Saved final dataset report-> {report_name}")

    print("[INFO] Pipeline complete.")
