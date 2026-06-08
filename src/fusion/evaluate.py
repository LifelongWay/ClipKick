"""
F3 — Evaluation harness.

Scores a highlights CSV (start,end,event_type,score) against the ground-truth
annotation JSON using asymmetric tolerance matching, and reports Precision /
Recall / F1 per event type plus an overall (type-agnostic) detection figure.

Build/inspect this before tuning any fusion model — it is what makes A/B/E
comparable on the same yardstick.

Run from project root:
    python src/fusion/evaluate.py --pred results/fusion/highlights/<match>.csv --match <id>
"""

import argparse
import os

import pandas as pd

try:
    from . import common
except ImportError:
    import common


def _load_preds(pred_csv):
    if not os.path.exists(pred_csv):
        return []
    df = pd.read_csv(pred_csv)
    if df.empty:
        return []
    return [{"start": float(r.start), "end": float(getattr(r, "end", r.start)),
             "event_type": str(r.event_type), "score": float(getattr(r, "score", 0.0))}
            for r in df.itertuples()]


def evaluate(preds, match_id, ann_dir=common.ANN_DIR, verbose=True):
    """Return a dict of metrics; optionally print a per-type table."""
    truth = common.load_truth(match_id, ann_dir=ann_dir)
    report = {"match_id": match_id, "n_pred": len(preds), "n_truth": len(truth), "per_type": {}}

    if verbose:
        print(f"\n=== {match_id} ===  preds={len(preds)}  truth={len(truth)}")
        print(f"{'type':<10}{'P':>7}{'R':>7}{'F1':>7}{'tp':>5}{'fp':>5}{'fn':>5}")

    for etype in common.EVENT_TYPES:
        p_t = [p for p in preds if p["event_type"] == etype]
        g_t = [g for g in truth if g["type"] == etype]
        if not p_t and not g_t:
            continue
        tp, fp, fn = common.match_predictions(p_t, g_t, type_aware=True)
        p, r, f = common.prf(tp, fp, fn)
        report["per_type"][etype] = {"precision": p, "recall": r, "f1": f,
                                     "tp": tp, "fp": fp, "fn": fn}
        if verbose:
            print(f"{etype:<10}{p:>7.2f}{r:>7.2f}{f:>7.2f}{tp:>5}{fp:>5}{fn:>5}")

    # Overall, type-agnostic (did we find the moment at all?)
    tp, fp, fn = common.match_predictions(preds, truth, type_aware=False)
    p, r, f = common.prf(tp, fp, fn)
    report["overall"] = {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}
    if verbose:
        print(f"{'OVERALL':<10}{p:>7.2f}{r:>7.2f}{f:>7.2f}{tp:>5}{fp:>5}{fn:>5}")
    return report


def evaluate_csv(pred_csv, match_id=None, ann_dir=common.ANN_DIR, verbose=True):
    if match_id is None:
        match_id = os.path.splitext(os.path.basename(pred_csv))[0]
    return evaluate(_load_preds(pred_csv), match_id, ann_dir=ann_dir, verbose=verbose)


def main():
    parser = argparse.ArgumentParser(description="F3 — evaluate highlights vs ground truth")
    parser.add_argument("--pred", required=True, help="highlights CSV")
    parser.add_argument("--match", default=None, help="match id (defaults to CSV basename)")
    parser.add_argument("--ann-dir", default=common.ANN_DIR)
    args = parser.parse_args()
    evaluate_csv(args.pred, match_id=args.match, ann_dir=args.ann_dir)


if __name__ == "__main__":
    main()
