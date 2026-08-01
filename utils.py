"""Shared utilities: calibration metrics, calibrator fitting, and downstream metrics."""
import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


# ── Calibration metrics ───────────────────────────────────────────────────────

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs <= hi) if i == n_bins - 1 else (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(probs[mask].mean() - labels[mask].mean())
    return float(ece)


def compute_mce(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs <= hi) if i == n_bins - 1 else (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        mce = max(mce, abs(probs[mask].mean() - labels[mask].mean()))
    return float(mce)


def compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels.astype(float)) ** 2))


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """Return (bin_centers, avg_accuracy, bin_counts) for a reliability diagram."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, counts = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs <= hi) if i == n_bins - 1 else (probs >= lo) & (probs < hi)
        centers.append((lo + hi) / 2)
        if mask.sum() == 0:
            accs.append(np.nan)
            counts.append(0)
        else:
            accs.append(float(labels[mask].mean()))
            counts.append(int(mask.sum()))
    return np.array(centers), np.array(accs), np.array(counts)


def bootstrap_ece_ci(df: pd.DataFrame, n_bootstrap: int = 1000,
                     n_bins: int = 15, seed: int = 42) -> tuple:
    """Bootstrap 95% CI for ECE by resampling queries."""
    rng = np.random.default_rng(seed)
    judged = df[df["qrel_label"] >= 0]
    query_ids = judged["query_id"].unique()
    ece_vals = []
    for _ in range(n_bootstrap):
        sampled_qids = rng.choice(query_ids, size=len(query_ids), replace=True)
        sample = pd.concat([judged[judged["query_id"] == qid] for qid in sampled_qids])
        if len(sample) < 5:
            continue
        ece_vals.append(compute_ece(sample["ce_prob"].values, sample["qrel_label"].values, n_bins))
    arr = np.array(ece_vals)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


# ── Calibrator fitting ────────────────────────────────────────────────────────

def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Find scalar T that minimizes NLL of logits/T against binary labels."""
    labels = labels.astype(float)

    def nll(T):
        p = np.clip(sigmoid(logits / T), 1e-7, 1 - 1e-7)
        return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

    result = minimize_scalar(nll, bounds=(0.01, 20.0), method="bounded")
    return float(result.x)


def fit_platt(logits: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=10_000)
    lr.fit(logits.reshape(-1, 1), labels.astype(int))
    return lr


def fit_isotonic(probs: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(probs, labels.astype(float))
    return ir


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    return sigmoid(logits / T).astype(np.float32)


def apply_platt(logits: np.ndarray, lr: LogisticRegression) -> np.ndarray:
    return lr.predict_proba(logits.reshape(-1, 1))[:, 1].astype(np.float32)


def apply_isotonic(probs: np.ndarray, ir: IsotonicRegression) -> np.ndarray:
    return ir.predict(probs).astype(np.float32)


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def dcg_at_k(rels: list, k: int) -> float:
    rels = rels[:k]
    return sum(r / np.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked_rels: list, ideal_rels: list, k: int) -> float:
    ideal = sorted(ideal_rels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return float("nan")
    return dcg_at_k(ranked_rels, k) / idcg


def compute_ndcg10(score_df: pd.DataFrame, qrels: dict, k: int = 10) -> float:
    """score_df must have columns: query_id, doc_id, score."""
    vals = []
    for qid, grp in score_df.groupby("query_id"):
        q_qrels = qrels.get(qid, {})
        if not q_qrels or sum(q_qrels.values()) == 0:
            continue
        top = grp.sort_values("score", ascending=False).head(k)
        ranked = [q_qrels.get(d, 0) for d in top["doc_id"]]
        ideal  = list(q_qrels.values())
        v = ndcg_at_k(ranked, ideal, k)
        if not np.isnan(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def compute_map(score_df: pd.DataFrame, qrels_bin: dict, k: int = 100) -> float:
    ap_vals = []
    for qid, grp in score_df.groupby("query_id"):
        relevant = {d for d, r in qrels_bin.get(qid, {}).items() if r >= 1}
        if not relevant:
            continue
        top = grp.sort_values("score", ascending=False).head(k)["doc_id"].tolist()
        hits, n_rel = [], 0
        for i, d in enumerate(top, 1):
            if d in relevant:
                n_rel += 1
                hits.append(n_rel / i)
        ap_vals.append(np.mean(hits) if hits else 0.0)
    return float(np.mean(ap_vals)) if ap_vals else float("nan")


def compute_recall_at_k(score_df: pd.DataFrame, qrels_bin: dict, k: int = 100) -> float:
    vals = []
    for qid, grp in score_df.groupby("query_id"):
        relevant = {d for d, r in qrels_bin.get(qid, {}).items() if r >= 1}
        if not relevant:
            continue
        top_k = set(grp.sort_values("score", ascending=False).head(k)["doc_id"])
        vals.append(len(relevant & top_k) / len(relevant))
    return float(np.mean(vals)) if vals else float("nan")


def compute_abstention_quality(score_df: pd.DataFrame, qrels_bin: dict,
                                threshold: float = 0.5) -> dict:
    """
    Accept a query's top-1 doc if p > threshold; otherwise abstain.
    Returns precision on the accepted set and accept/abstain rates.
    """
    accepted_correct = accepted_total = abstained = 0
    for qid, grp in score_df.groupby("query_id"):
        q_qrels = qrels_bin.get(qid, {})
        top1 = grp.sort_values("score", ascending=False).iloc[0]
        if top1["score"] > threshold:
            accepted_total += 1
            if q_qrels.get(top1["doc_id"], 0) >= 1:
                accepted_correct += 1
        else:
            abstained += 1
    total = accepted_total + abstained
    return {
        "precision_at_accept": accepted_correct / accepted_total if accepted_total else float("nan"),
        "accept_rate":  accepted_total / total if total else float("nan"),
        "abstain_rate": abstained       / total if total else float("nan"),
        "n_queries": total,
    }
