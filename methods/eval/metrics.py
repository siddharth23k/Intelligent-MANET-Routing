"""Route quality metrics and the statistics used to compare methods."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats


def route_metrics(graph: nx.Graph, path: Optional[Sequence[int]]) -> Dict[str, float]:
    """Model derived route quality.

    When the graph was weighted by the same reliabilities averaged here, this
    metric and the routing objective are the same function. Use path survival
    for an independent answer.
    """
    if not path or len(path) < 2:
        return {"avg_reliability": 0.0, "min_reliability": 0.0, "hop_count": 0}
    reliabilities = [graph[u][v]["reliability"] for u, v in zip(path[:-1], path[1:])]
    return {
        "avg_reliability": float(np.mean(reliabilities)),
        "min_reliability": float(np.min(reliabilities)),
        "hop_count": int(len(path) - 1),
    }


def paired_run_test(
    df: pd.DataFrame, value_col: str, baseline_col: str, group: str = "run_id"
) -> dict:
    """Paired comparison at the run level.

    Decisions inside one run share almost all of their topology, so the number
    of independent units is the number of runs. Wilcoxon is reported alongside
    the t test because at that sample size normality cannot be checked.
    """
    by_run = df.groupby(group)[[value_col, baseline_col]].mean()
    n_runs = int(len(by_run))
    out = {
        "n_runs": n_runs,
        "mean_delta": float((by_run[value_col] - by_run[baseline_col]).mean()),
    }
    if n_runs < 2:
        out.update(t_p_value=float("nan"), wilcoxon_p_value=float("nan"), cohens_d=float("nan"))
        return out

    a = by_run[value_col].to_numpy(dtype=float)
    b = by_run[baseline_col].to_numpy(dtype=float)
    diff = a - b

    try:
        out["t_p_value"] = float(stats.ttest_rel(a, b).pvalue)
    except Exception:
        out["t_p_value"] = float("nan")
    try:
        out["wilcoxon_p_value"] = (
            float("nan") if np.allclose(diff, 0.0) else float(stats.wilcoxon(a, b).pvalue)
        )
    except Exception:
        out["wilcoxon_p_value"] = float("nan")

    sd = float(np.std(diff, ddof=1))
    out["cohens_d"] = float(np.mean(diff) / sd) if sd > 0 else float("nan")
    return out


def win_loss_tie(df: pd.DataFrame, value_col: str, baseline_col: str, tol: float = 1e-12) -> dict:
    """How often the method actually differs from the baseline.

    A win rate of 1.0 with a loss rate of 0.0 usually means the metric is not
    independent of the optimiser, not that the model is strong.
    """
    diff = (df[value_col] - df[baseline_col]).to_numpy(dtype=float)
    n = max(1, len(diff))
    return {
        "n_decisions": int(len(diff)),
        "win_rate": float(np.sum(diff > tol) / n),
        "tie_rate": float(np.sum(np.abs(diff) <= tol) / n),
        "loss_rate": float(np.sum(diff < -tol) / n),
    }


def proportion_test(successes_a: int, n_a: int, successes_b: int, n_b: int) -> dict:
    """Two proportion z test, used for survival rate differences."""
    if n_a == 0 or n_b == 0:
        return {"delta": float("nan"), "p_value": float("nan")}
    pa, pb = successes_a / n_a, successes_b / n_b
    pooled = (successes_a + successes_b) / (n_a + n_b)
    se = float(np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b)))
    if se == 0:
        return {"delta": float(pa - pb), "p_value": float("nan")}
    z = (pa - pb) / se
    return {"delta": float(pa - pb), "p_value": float(2 * (1 - stats.norm.cdf(abs(z))))}
