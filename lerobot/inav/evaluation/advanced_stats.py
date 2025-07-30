
"""advanced_stats.py
Statistical comparison of action smoothness between two policies (e.g. IL vs RL).

Workflow implemented according to user instructions:
1. For every rollout/episode, compute the *mean* L2-norm of first-order action
   differences: Δa_t = a_{t+1} − a_t.
2. Collect the distribution of these values for two policies (100 episodes each
   by default).
3. Assess statistical assumptions:
      • Shapiro–Wilk normality test + Q-Q plot for each policy.
      • Levene’s test for homogeneity of variances.
4. Depending on the outcomes, apply
      • Student’s t-test  (normal + equal variances),
      • Welch’s t-test    (normal + unequal variances), or
      • Mann–Whitney U    (non-normal).

The script is **stand-alone executable** but every processing step is exposed
as a function so you can import/extend/test them individually.

Example
-------
$ python advanced_stats.py \
      --eval-pkl unique_episodes_random.pkl \
      --rl-dir  ../../../datasets/iss_docking_images/data/chunk-000 \
      --n-episodes 100

All plotting / verbose logging is controlled by CLI flags so they can be
silenced once debugging is finished.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# -----------------------------------------------------------------------------
# ------------------------------  IO utilities  --------------------------------
# -----------------------------------------------------------------------------


def load_il_episodes(pkl_path: Path, max_episodes: int | None = None) -> List[dict]:
    """Load IL episodes stored as a pickle list of dicts.

    Each dict is expected to contain keys **T** and **L** with shapes (N-1, 3)
    – thrust and torque – which are concatenated into a 6-D action vector.
    """
    import pickle  # local import to avoid cost when function not used

    pkl_path = Path(pkl_path).expanduser()
    if not pkl_path.exists():  # pragma: no cover – safeguard for CLI usage
        raise FileNotFoundError(pkl_path)

    with pkl_path.open("rb") as f:
        episodes = pickle.load(f)

    if max_episodes is not None:
        episodes = episodes[: max_episodes]

    return episodes


def load_rl_parquet_actions(parquet_file: Path) -> np.ndarray:
    """Return actions (N, 6) from a single parquet episode file."""
    df = pd.read_parquet(parquet_file)
    # Column may be named exactly "action" with list-like rows
    if "action" not in df:
        raise KeyError(f"Column 'action' not found in {parquet_file}")
    actions = np.stack(df["action"].to_numpy())  # shape (N, 6)
    return actions


def list_rl_parquet_files(rl_dir: Path, max_episodes: int | None = None) -> List[Path]:
    """Return sorted list of episode parquet files inside *rl_dir*."""
    files = sorted(Path(rl_dir).glob("episode_*.parquet"))
    if max_episodes is not None:
        files = files[: max_episodes]
    return files

# -----------------------------------------------------------------------------
# ---------------------------  Metric computation  ----------------------------
# -----------------------------------------------------------------------------


def mean_delta_l2(actions: np.ndarray) -> float:
    """Compute mean L2 norm of successive action differences for one episode."""
    if len(actions) < 2:  # not enough timesteps
        return np.nan
    delta = actions[1:] - actions[:-1]
    norms = np.linalg.norm(delta, axis=1)
    return float(norms.mean())


def compute_metrics_il(episodes: List[dict]) -> np.ndarray:
    """Compute smoothness metric for each IL episode."""
    vals: list[float] = []
    for ep in episodes:
        T = np.asarray(ep["T"])
        L = np.asarray(ep["L"])
        actions = np.concatenate([T, L], axis=1)  # (N-1, 6)
        vals.append(mean_delta_l2(actions))
    return np.asarray(vals, dtype=float)


def compute_metrics_rl(parquet_files: List[Path]) -> np.ndarray:
    """Compute smoothness metric for each RL parquet episode."""
    vals: list[float] = []
    for pf in parquet_files:
        try:
            actions = load_rl_parquet_actions(pf)
            vals.append(mean_delta_l2(actions))
        except Exception as exc:
            print(f"[WARN] Skipping {pf.name}: {exc}")
    return np.asarray(vals, dtype=float)

# -----------------------------------------------------------------------------
# -----------------------------  Statistics  ----------------------------------
# -----------------------------------------------------------------------------


def normality_and_variance_tests(
    data_a: np.ndarray,
    data_b: np.ndarray,
    alpha: float = 0.05,
    verbose: bool = True,
    plot: bool = True,
    label_a: str = "ACT",
    label_b: str = "RL",
) -> Tuple[str, dict]:
    """Run assumption checks and choose the appropriate significance test.

    Returns a tuple *(chosen_test_name, results_dict)* where *results_dict*
    contains the individual test statistics and p-values as well as the final
    selected test result.
    """

    results = {}

    # ----- Shapiro–Wilk -----
    sw_a = stats.shapiro(data_a)
    sw_b = stats.shapiro(data_b)
    results["shapiro_a"] = sw_a
    results["shapiro_b"] = sw_b
    normal_a = sw_a.pvalue > alpha
    normal_b = sw_b.pvalue > alpha

    if verbose:
        print("\nShapiro–Wilk normality test")
        print(f"  {label_a}: statistic={sw_a.statistic:.4f}, p={sw_a.pvalue:.4e}")
        print(f"  {label_b}: statistic={sw_b.statistic:.4f}, p={sw_b.pvalue:.4e}")

    if plot:
        import statsmodels.api as sm  # heavy import only when needed

        # Create two separate plots

        # --- First Q-Q Plot ---
        plt.figure(figsize=(6, 6))
        sm.qqplot(data_a, line="s")
        plt.xlabel("Theoretical Quantiles", fontsize=36)
        plt.ylabel("Sample Quantiles", fontsize=36, labelpad=15)
        plt.tick_params(axis='both', which='major', labelsize=36)
        plt.tight_layout()
        plt.show()

        # --- Second Q-Q Plot ---
        plt.figure(figsize=(6, 6))
        sm.qqplot(data_b, line="s")
        plt.xlabel("Theoretical Quantiles", fontsize=36)
        plt.ylabel("Sample Quantiles", fontsize=36, labelpad=15)
        plt.tick_params(axis='both', which='major', labelsize=36)
        plt.tight_layout()
        plt.show()


    # ----- Levene (equal variance) -----
    levene_res = stats.levene(data_a, data_b, center="median")
    results["levene"] = levene_res
    equal_var = levene_res.pvalue > alpha

    if verbose:
        print("\nLevene variance test")
        print(f"  statistic={levene_res.statistic:.4f}, p={levene_res.pvalue:.4e}")

    # ----- Decide on final test -----
    if normal_a and normal_b:
        if equal_var:
            test_name = "Student t-test"
            t_res = stats.ttest_ind(data_a, data_b, equal_var=True)
        else:
            test_name = "Welch t-test"
            t_res = stats.ttest_ind(data_a, data_b, equal_var=False)
        results["final_test"] = t_res
    else:
        test_name = "Mann–Whitney U"
        u_res = stats.mannwhitneyu(data_a, data_b, alternative="two-sided")
        results["final_test"] = u_res

    if verbose:
        print(f"\nSelected test: {test_name}")
        res = results["final_test"]
        print(f"  statistic={res.statistic:.4f}, p={res.pvalue:.4e}")

    return test_name, results

# -----------------------------------------------------------------------------
# -------------------------------  CLI  ---------------------------------------
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Statistical comparison of action smoothness between two policies")
    p.add_argument("--eval-pkl", type=Path, required=True, help="Path to IL evaluation pickle file")
    p.add_argument("--rl-dir", type=Path, required=True, help="Directory with RL parquet episode files")
    p.add_argument("--n-episodes", type=int, default=100, help="Maximum number of episodes to include from each policy (default 100)")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level for hypothesis tests")
    p.add_argument("--no-plot", action="store_true", help="Disable QQ and histogram plots")
    p.add_argument("--quiet", action="store_true", help="Suppress verbose logging")
    return p


def main(args: list[str] | None = None):  # noqa: D401 – simple main wrapper
    parser = build_arg_parser()
    cfg = parser.parse_args(args)

    verbose = not cfg.quiet
    plot = not cfg.no_plot

    # --------------- Load data ---------------
    il_eps = load_il_episodes(cfg.eval_pkl, max_episodes=cfg.n_episodes)
    rl_files = list_rl_parquet_files(cfg.rl_dir, max_episodes=cfg.n_episodes)

    if verbose:
        print(f"Loaded {len(il_eps)} IL episodes from {cfg.eval_pkl}")
        print(f"Loaded {len(rl_files)} RL episodes from {cfg.rl_dir}")

    il_metrics = compute_metrics_il(il_eps)
    rl_metrics = compute_metrics_rl(rl_files)

    # Handle NaNs (e.g. episodes too short)
    il_metrics = il_metrics[~np.isnan(il_metrics)]
    rl_metrics = rl_metrics[~np.isnan(rl_metrics)]

    if verbose:
        print("\nMetric summary (mean Δa L2 norm)")
        print(f"  IL: mean={il_metrics.mean():.6f}, std={il_metrics.std():.6f}, N={len(il_metrics)}")
        print(f"  RL: mean={rl_metrics.mean():.6f}, std={rl_metrics.std():.6f}, N={len(rl_metrics)}")

    # --------------- Plots ---------------
    if plot:
        plt.figure(figsize=(8, 4))
        plt.hist(il_metrics, bins=30, alpha=0.6, label="ACT", color="steelblue")
        plt.hist(rl_metrics, bins=30, alpha=0.6, label="RL", color="darkorange")
        plt.xlabel("Mean L2 norm of Δa per episode", fontsize=36)
        plt.ylabel("Count", fontsize=36)
        plt.legend(fontsize=36, loc="upper right")
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.tight_layout()
        plt.show()

    # --------------- Statistics ---------------
    normality_and_variance_tests(
        il_metrics,
        rl_metrics,
        alpha=cfg.alpha,
        verbose=verbose,
        plot=plot,
        label_a="IL",
        label_b="RL",
    )


if __name__ == "__main__":  # pragma: no cover
    main()

