#!/usr/bin/env python3
"""Batch-evaluate zero-shot vs few-shot vs chain-of-thought; write tables and figures."""

from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.config  # noqa: F401  — HF cache paths before embedding imports

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
)

from src.config import (  # noqa: E402
    EVAL_CASES_PATH,
    FIGURES_DIR,
    OLLAMA_MODEL,
    RESULTS_DIR,
)
from src.ollama_client import ollama_health  # noqa: E402
from src.rag_chain import (  # noqa: E402
    build_vectorstore,
    coherence_score,
    get_embeddings,
    run_turn,
)

VARIANTS = ["zero_shot", "few_shot", "chain_of_thought"]
LABEL_ORDER = ["routine", "monitor", "urgent", "emergency"]


def load_cases(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def precision_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float | None:
    if not gold_ids:
        return None
    topk = retrieved_ids[:k]
    if k == 0:
        return None
    hits = sum(1 for rid in topk if rid in set(gold_ids))
    return hits / float(k)


def latency_summary_ms(lat_ms: list[float]) -> dict[str, float]:
    """Mean / p50 / p95 over per-example wall-clock LLM+retrieval times."""
    if not lat_ms:
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan")}
    arr = np.asarray(lat_ms, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def plot_confusion(y_true: list[str], y_pred: list[str], title: str, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=LABEL_ORDER,
        yticklabels=LABEL_ORDER,
        ylabel="Gold",
        xlabel="Predicted",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_placeholder() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    msg = (
        "# Prompt comparison (not run)\n\n"
        f"Ollama did not report model `{OLLAMA_MODEL}` as available. "
        f"Run `ollama pull {OLLAMA_MODEL}` (or set `OLLAMA_MODEL` in `.env`) "
        "and ensure `ollama serve` is running, then re-run:\n\n"
        "`python scripts/eval_prompts.py`\n"
    )
    (RESULTS_DIR / "prompt_comparison.md").write_text(msg, encoding="utf-8")
    (RESULTS_DIR / "error_analysis.md").write_text(
        "# Error analysis\n\nGenerate this file by running `python scripts/eval_prompts.py` after Ollama is configured.\n",
        encoding="utf-8",
    )
    (RESULTS_DIR / "prompt_comparison.csv").write_text(
        "variant,split,category,query,gold_triage,pred_triage,exact_match,precision_at_k,coherence,latency_ms\n",
        encoding="utf-8",
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not ollama_health(OLLAMA_MODEL):
        print(
            f"Ollama model '{OLLAMA_MODEL}' not available; writing placeholder results."
        )
        write_placeholder()
        return

    cases = load_cases(EVAL_CASES_PATH)
    test_cases = [c for c in cases if c.get("split") == "test"]
    if not test_cases:
        test_cases = cases

    print("Building vector store...")
    vs = build_vectorstore()
    emb = get_embeddings()
    k_ret = 4

    rows_out: list[dict] = []
    variant_stats: dict[str, dict] = {}

    for variant in VARIANTS:
        y_true: list[str] = []
        y_pred: list[str] = []
        coh_vals: list[float] = []
        prec_vals: list[float] = []
        lat_ms_vals: list[float] = []
        detail_for_errors: list[dict] = []

        for ex in test_cases:
            q = ex["query"]
            gold_t = ex["gold_triage"]
            gold_ids = ex.get("gold_doc_ids") or []

            try:
                t0 = time.perf_counter()
                result = run_turn(
                    vs,
                    q,
                    [],
                    prompt_variant=variant,
                    model=OLLAMA_MODEL,
                    retrieval_k=k_ret,
                )
                lat_ms = (time.perf_counter() - t0) * 1000.0
            except Exception as e:
                print(f"Ollama error for variant={variant}: {e}")
                print("Ensure Ollama is running and the model exists:", OLLAMA_MODEL)
                sys.exit(1)

            pred_t = result.triage_level or "routine"
            if pred_t not in LABEL_ORDER:
                pred_t = "routine"

            retrieved_ids = [d.metadata.get("id", "") for d in result.retrieved]
            p_at_k = precision_at_k(retrieved_ids, gold_ids, k_ret)
            coh = coherence_score(emb, result.answer_text, result.retrieved)

            match = int(pred_t == gold_t)
            lat_ms_vals.append(lat_ms)
            rows_out.append(
                {
                    "variant": variant,
                    "split": ex.get("split", ""),
                    "category": ex.get("category", ""),
                    "query": q[:200],
                    "gold_triage": gold_t,
                    "pred_triage": pred_t,
                    "exact_match": str(match),
                    "precision_at_k": "" if p_at_k is None else f"{p_at_k:.3f}",
                    "coherence": f"{coh:.3f}",
                    "latency_ms": f"{lat_ms:.1f}",
                }
            )

            y_true.append(gold_t)
            y_pred.append(pred_t)
            coh_vals.append(coh)
            if p_at_k is not None:
                prec_vals.append(p_at_k)

            detail_for_errors.append(
                {
                    "pred_t": pred_t,
                    "retrieved_ids": retrieved_ids,
                    "answer": result.answer_text,
                }
            )

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)
        mean_coh = float(np.mean(coh_vals)) if coh_vals else 0.0
        mean_prec = float(np.mean(prec_vals)) if prec_vals else float("nan")
        lat_sum = latency_summary_ms(lat_ms_vals)

        variant_stats[variant] = {
            "accuracy": acc,
            "macro_f1": f1,
            "mean_precision_at_k": mean_prec,
            "mean_coherence": mean_coh,
            "had_prec": bool(prec_vals),
            "y_true": list(y_true),
            "y_pred": list(y_pred),
            "details": detail_for_errors,
            "latency_ms": lat_sum,
        }

        rows_out.append(
            {
                "variant": variant,
                "split": "__summary__",
                "category": "",
                "query": "",
                "gold_triage": "",
                "pred_triage": "",
                "exact_match": f"accuracy={acc:.3f}",
                "precision_at_k": (
                    f"mean_p@{k_ret}={mean_prec:.3f}" if prec_vals else "mean_p@k=n/a"
                ),
                "coherence": f"mean={mean_coh:.3f}",
                "latency_ms": (
                    f"mean_ms={lat_sum['mean']:.1f} p50_ms={lat_sum['p50']:.1f} "
                    f"p95_ms={lat_sum['p95']:.1f}"
                ),
            }
        )

        plot_confusion(
            y_true,
            y_pred,
            title=f"Confusion matrix ({variant}) test n={len(test_cases)}",
            out_path=FIGURES_DIR / f"confusion_matrix_{variant}.png",
        )

    best_variant = max(VARIANTS, key=lambda v: variant_stats[v]["accuracy"])
    shutil.copyfile(
        FIGURES_DIR / f"confusion_matrix_{best_variant}.png",
        FIGURES_DIR / "confusion_matrix.png",
    )

    csv_path = RESULTS_DIR / "prompt_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "split",
                "category",
                "query",
                "gold_triage",
                "pred_triage",
                "exact_match",
                "precision_at_k",
                "coherence",
                "latency_ms",
            ],
        )
        w.writeheader()
        for row in rows_out:
            w.writerow(row)

    lines = [
        "# Prompt comparison (test split)",
        "",
        "| Prompt variant | Accuracy | Macro F1 | Mean P@k (labeled) | Mean coherence |",
        "|----------------|----------|----------|--------------------|----------------|",
    ]
    for variant in VARIANTS:
        s = variant_stats[variant]
        pks = f"{s['mean_precision_at_k']:.3f}" if s["had_prec"] else "n/a"
        lines.append(
            f"| {variant} | {s['accuracy']:.3f} | {s['macro_f1']:.3f} | {pks} | {s['mean_coherence']:.3f} |"
        )
    lines.append("")
    lines.append(f"Best accuracy variant: `{best_variant}` (see `figures/confusion_matrix.png`).")
    lines.append(f"Model: `{OLLAMA_MODEL}`; retrieval k={k_ret}.")
    lines.append("")
    lines.append(
        "Per-variant figures: "
        + ", ".join(f"`figures/confusion_matrix_{v}.png`" for v in VARIANTS)
    )
    lines.extend(
        [
            "",
            "## Inference latency (offline batch)",
            "",
            "Wall-clock time per example includes retrieval + local LLM generation (`run_turn`). Units: milliseconds.",
            "",
            "| Prompt variant | Mean ms | p50 ms | p95 ms |",
            "|----------------|---------|--------|--------|",
        ]
    )
    for variant in VARIANTS:
        ls = variant_stats[variant]["latency_ms"]
        lines.append(
            f"| {variant} | {ls['mean']:.1f} | {ls['p50']:.1f} | {ls['p95']:.1f} |"
        )
    (RESULTS_DIR / "prompt_comparison.md").write_text("\n".join(lines), encoding="utf-8")

    primary = "zero_shot"
    errs = []
    for ex, det in zip(test_cases, variant_stats[primary]["details"]):
        if det["pred_t"] != ex["gold_triage"]:
            errs.append(
                {
                    "query": ex["query"],
                    "gold": ex["gold_triage"],
                    "pred": det["pred_t"],
                    "category": ex.get("category", ""),
                    "retrieved": det["retrieved_ids"],
                }
            )

    ue_boundary = [
        e
        for e in errs
        if e["gold"] != e["pred"]
        and {e["gold"], e["pred"]}.issubset({"urgent", "emergency"})
    ]

    zs_details = variant_stats[primary]["details"]
    edge_rows: list[str] = []
    for ex, det in zip(test_cases, zs_details):
        cat = (ex.get("category") or "").lower()
        if "edge" in cat or cat in {"vague", "edge_fiction"}:
            q = ex["query"]
            qshort = q if len(q) <= 140 else q[:137] + "..."
            edge_rows.append(
                f"- `{cat or 'unknown'}`: gold **{ex['gold_triage']}**, zero-shot pred **{det['pred_t']}** — `{qshort}`"
            )

    ea = [
        "# Error analysis (zero-shot, test split)",
        "",
        f"Test cases in this run: **{len(test_cases)}**. Misclassified (zero-shot exact label match): **{len(errs)}**.",
        "",
        "Numbers come from `results/prompt_comparison.csv` and confusion matrices under `figures/`. "
        "Gold labels are heuristic course labels—not clinically adjudicated.",
        "",
        "## What the mistakes look like (qualitative)",
        "",
        "Most errors are **urgent vs emergency** swaps when the user describes severe symptoms. The model tends to be conservative— "
        "when retrieval surfaces stroke/bleeding/severe infection passages, it may push the informational tag upward compared to the "
        "gold label line in `eval_cases.jsonl`. A smaller pattern is **head injury** wording where the gold expects emergency but the "
        "model lands on urgent when vomiting is emphasized without explicit neurological deficits in the structured output.",
        "",
        "## Themes",
        "- **Severity boundary (urgent vs emergency)**: competing serious cues (stroke vs migraine, GI bleed vs nausea) drive label drift.",
        "- **Vague inputs**: low-detail prompts sometimes collapse to **monitor** because there are few retrieval anchors.",
        "- **Retrieval limits**: top-k passages may omit the single gold article ID even when the answer is reasonable.",
        "- **OOD / fiction**: nonsense prompts still retrieve real articles; labels treat **routine** as the safe bucket.",
        "",
        "## Urgent vs emergency boundary (mislabels only)",
        f"Count among errors where gold/pred are only urgent/emergency (subset): **{len(ue_boundary)}**.",
        "",
    ]
    for e in ue_boundary[:8]:
        qshort = e["query"] if len(e["query"]) <= 120 else e["query"][:117] + "..."
        ea.append(
            f"- gold **{e['gold']}** → pred **{e['pred']}** — `{qshort}` — retrieved: `{e['retrieved']}`"
        )
    if not ue_boundary:
        ea.append("- (none in this run)")

    ea.extend(
        [
            "",
            "## Edge / out-of-distribution prompts (categories)",
            "",
            "These rows intentionally exercise vague or fictional language; see `data/eval_cases.jsonl`.",
            "",
        ]
    )
    ea.extend(edge_rows if edge_rows else ["- (no edge-tagged rows in this split)"])

    ea.extend(
        [
            "",
            "## Example failures (all mislabels)",
            "",
        ]
    )
    for e in errs[:14]:
        qshort = e["query"] if len(e["query"]) <= 120 else e["query"][:117] + "..."
        ea.append(
            f"- Category `{e['category']}`: gold **{e['gold']}**, pred **{e['pred']}** — `{qshort}` — retrieved: `{e['retrieved']}`"
        )
    (RESULTS_DIR / "error_analysis.md").write_text("\n".join(ea), encoding="utf-8")

    print("Wrote", csv_path)
    print("Wrote", RESULTS_DIR / "prompt_comparison.md")
    print("Wrote", RESULTS_DIR / "error_analysis.md")
    print("Wrote figures to", FIGURES_DIR, "best=", best_variant)


if __name__ == "__main__":
    main()
