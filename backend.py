# backend.py
"""
Backend utilities for TEQUILA equivalence testing.

This module provides:
- File/table parsing helper (`read_table_from_bytes`)
- Summary-statistic based TOST for t-tests (one-sample, pooled two-sample, Welch)
- Raw-data wrappers for the t-test variants
- Brunner-Munzel test (replaces Mann-Whitney U) with analytic SE
- Hodges-Lehmann tests (two-sample and one-sample/paired) with asymptotic inference
- Paired concordance test for paired/repeated-measures designs
- Two-sample proportions TOST (difference, odds ratio, risk ratio)

Docstrings in this file follow the NumPy documentation style.
"""

from typing import Tuple, Optional
import io
import math

import numpy as np
import pandas as pd

# Attempt to import required scipy functions; raise helpful errors if missing at call-time
try:
    from scipy.stats import t as t_dist
    from scipy.stats import norm
    from scipy.stats import gaussian_kde
except Exception:
    t_dist = None
    norm = None
    gaussian_kde = None


def _equivalence_ci_t(estimate: float, se: float, df: float, alpha: float):
    """Return the (1 - 2*alpha) confidence interval used with TOST."""
    if t_dist is None:
        raise RuntimeError("scipy is required for confidence interval calculations. Install scipy.")
    if se < 0 or not np.isfinite(se):
        raise ValueError("Standard error must be finite and non-negative.")
    if df <= 0 or not np.isfinite(df):
        raise ValueError("Degrees of freedom must be finite and positive.")
    q = float(t_dist.ppf(1.0 - alpha, df))
    return float(estimate - q * se), float(estimate + q * se)


# -------------------------
# Utility summarizer
# -------------------------
def summarize(arr_like) -> Tuple[float, float, int]:
    """
    Compute mean, sample standard deviation, and sample size for an array-like.

    NaN values are ignored.

    Parameters
    ----------
    arr_like : array-like
        Sequence of numeric values (will be cast to float).

    Returns
    -------
    mean : float
        Sample mean of non-NaN values.
    sd : float
        Sample standard deviation (ddof=1). If n <= 1, returns 0.0.
    n : int
        Number of non-NaN observations.

    Raises
    ------
    ValueError
        If there are no valid (non-NaN) observations.
    """
    a = np.asarray(arr_like, dtype=float)
    if a.ndim > 1:
        a = a.ravel()
    a = a[~np.isnan(a)]
    n = a.size
    if n == 0:
        raise ValueError("No valid (non-NaN) observations found.")
    mean = float(np.mean(a)) if n > 0 else float("nan")
    if n <= 1:
        sd = 0.0
    else:
        sd = float(np.std(a, ddof=1))
    return mean, sd, int(n)


# -------------------------
# TOST summary-stat functions (t-tests)
# -------------------------
def one_sample_tost_from_summary(mean: float, sd: float, n: int, mu0: float, lower: float, upper: float, alpha: float):
    """
    One-sample TOST using summary statistics (t-distribution).

    Parameters
    ----------
    mean : float
        Sample mean.
    sd : float
        Sample standard deviation (sample, ddof=1).
    n : int
        Sample size (must be > 1).
    mu0 : float
        Reference mean for the one-sample test (often 0).
    lower, upper : float
        Equivalence bounds for the mean difference (lower < upper).
    alpha : float
        Significance level (unused for calculation here but kept for API symmetry).

    Returns
    -------
    p_low : float
        One-sided p-value testing mean - mu0 > lower (upper bound of lower test).
    p_high : float
        One-sided p-value testing mean - mu0 < upper (upper test).
    details : dict
        Auxiliary computed values (se, df, t statistics, etc).

    Raises
    ------
    RuntimeError
        If required SciPy functionality is not available.
    ValueError
        If n <= 1.
    """
    if t_dist is None:
        raise RuntimeError("scipy is required for summary-stat calculations. Install scipy.")
    if n <= 1:
        raise ValueError("n must be > 1 for one-sample test.")
    se = sd / np.sqrt(n)
    df = n - 1
    diff = mean - mu0
    t_low = (diff - lower) / se
    p_low = 1.0 - float(t_dist.cdf(t_low, df))
    t_high = (diff - upper) / se
    p_high = float(t_dist.cdf(t_high, df))
    ci_lower, ci_upper = _equivalence_ci_t(diff, se, df, alpha)
    details = {
        "mean": float(mean),
        "sd": float(sd),
        "n": int(n),
        "se": float(se),
        "df": float(df),
        "mean_diff": float(diff),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_level": float(1.0 - 2.0 * alpha),
        "t_low": float(t_low),
        "t_high": float(t_high),
    }
    return float(p_low), float(p_high), details


def two_sample_pooled_tost_from_summary(m1: float, s1: float, n1: int, m2: float, s2: float, n2: int, lower: float, upper: float, alpha: float):
    """
    Two-sample pooled (equal-variance) TOST using summary statistics.

    Parameters
    ----------
    m1, s1, n1 : float, float, int
        Mean, sample sd, and sample size for group 1.
    m2, s2, n2 : float, float, int
        Mean, sample sd, and sample size for group 2.
    lower, upper : float
        Equivalence bounds for the difference (m1 - m2).
    alpha : float
        Significance level (kept for API symmetry).

    Returns
    -------
    p_low, p_high : float
        One-sided p-values for the two TOST comparisons.
    details : dict
        Auxiliary computed values: pooled variance, se, df, t-statistics, etc.

    Raises
    ------
    RuntimeError
        If SciPy t-distribution is unavailable.
    ValueError
        If n1 <= 1 or n2 <= 1.
    """
    if t_dist is None:
        raise RuntimeError("scipy is required for summary-stat calculations. Install scipy.")
    if n1 <= 1 or n2 <= 1:
        raise ValueError("n1 and n2 must be > 1 for pooled (equal-variance) two-sample test.")
    df = n1 + n2 - 2
    ss1 = (n1 - 1) * (s1 ** 2)
    ss2 = (n2 - 1) * (s2 ** 2)
    pooled_var = (ss1 + ss2) / df
    se = np.sqrt(pooled_var * (1.0 / n1 + 1.0 / n2))
    diff = m1 - m2
    t_low = (diff - lower) / se
    p_low = 1.0 - float(t_dist.cdf(t_low, df))
    t_high = (diff - upper) / se
    p_high = float(t_dist.cdf(t_high, df))
    ci_lower, ci_upper = _equivalence_ci_t(diff, se, df, alpha)
    details = {
        "mean1": float(m1),
        "sd1": float(s1),
        "n1": int(n1),
        "mean2": float(m2),
        "sd2": float(s2),
        "n2": int(n2),
        "pooled_var": float(pooled_var),
        "se": float(se),
        "df": float(df),
        "mean_diff": float(diff),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_level": float(1.0 - 2.0 * alpha),
        "t_low": float(t_low),
        "t_high": float(t_high),
    }
    return float(p_low), float(p_high), details


def two_sample_welch_tost_from_summary(m1: float, s1: float, n1: int, m2: float, s2: float, n2: int, lower: float, upper: float, alpha: float):
    """
    Two-sample Welch TOST using summary statistics (unequal variances).

    Parameters
    ----------
    m1, s1, n1 : float, float, int
        Mean, sample sd, and sample size for group 1.
    m2, s2, n2 : float, float, int
        Mean, sample sd, and sample size for group 2.
    lower, upper : float
        Equivalence bounds for the difference (m1 - m2).
    alpha : float
        Significance level (kept for API symmetry).

    Returns
    -------
    p_low, p_high : float
        One-sided p-values for the two TOST comparisons.
    details : dict
        Auxiliary computed values including Welch df estimate, se, etc.

    Raises
    ------
    RuntimeError
        If SciPy t-distribution is unavailable.
    ValueError
        If n1 <= 1 or n2 <= 1.
    """
    if t_dist is None:
        raise RuntimeError("scipy is required for summary-stat calculations. Install scipy.")
    if n1 <= 1 or n2 <= 1:
        raise ValueError("n1 and n2 must be > 1 for Welch two-sample test.")
    se = np.sqrt((s1 ** 2) / n1 + (s2 ** 2) / n2)
    num = ((s1 ** 2) / n1 + (s2 ** 2) / n2) ** 2
    den = 0.0
    if n1 > 1:
        den += (s1 ** 4) / ((n1 ** 2) * (n1 - 1))
    if n2 > 1:
        den += (s2 ** 4) / ((n2 ** 2) * (n2 - 1))
    df = float(num / den) if den > 0 else float("nan")
    diff = m1 - m2
    t_low = (diff - lower) / se
    p_low = 1.0 - float(t_dist.cdf(t_low, df))
    t_high = (diff - upper) / se
    p_high = float(t_dist.cdf(t_high, df))
    ci_lower, ci_upper = _equivalence_ci_t(diff, se, df, alpha)
    details = {
        "mean1": float(m1),
        "sd1": float(s1),
        "n1": int(n1),
        "mean2": float(m2),
        "sd2": float(s2),
        "n2": int(n2),
        "se": float(se),
        "df": float(df),
        "mean_diff": float(diff),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_level": float(1.0 - 2.0 * alpha),
        "t_low": float(t_low),
        "t_high": float(t_high),
    }
    return float(p_low), float(p_high), details


# -------------------------
# Raw-data convenience wrappers (t-tests)
# -------------------------
def one_sample_tost(raw_array, mu0, lower, upper, alpha):
    """
    One-sample TOST wrapper that accepts raw data.

    Parameters
    ----------
    raw_array : array-like
        Raw sample values (NaNs ignored).
    mu0, lower, upper, alpha : float
        See `one_sample_tost_from_summary`.

    Returns
    -------
    tuple
        (p_low, p_high, details) from `one_sample_tost_from_summary`.
    """
    mean, sd, n = summarize(raw_array)
    return one_sample_tost_from_summary(mean, sd, n, mu0, lower, upper, alpha)


def two_sample_pooled_tost(a, b, lower, upper, alpha):
    """
    Two-sample pooled TOST wrapper that accepts raw data arrays for groups a and b.

    Parameters
    ----------
    a, b : array-like
        Raw data for groups 1 and 2 (NaNs ignored).
    lower, upper, alpha : float
        See `two_sample_pooled_tost_from_summary`.

    Returns
    -------
    tuple
        (p_low, p_high, details)
    """
    m1, sd1, n1 = summarize(a)
    m2, sd2, n2 = summarize(b)
    return two_sample_pooled_tost_from_summary(m1, sd1, n1, m2, sd2, n2, lower, upper, alpha)


def two_sample_welch_tost(a, b, lower, upper, alpha):
    """
    Two-sample Welch TOST wrapper that accepts raw data arrays for groups a and b.

    See `two_sample_welch_tost_from_summary` for parameter meanings.
    """
    m1, sd1, n1 = summarize(a)
    m2, sd2, n2 = summarize(b)
    return two_sample_welch_tost_from_summary(m1, sd1, n1, m2, sd2, n2, lower, upper, alpha)



# =============================================================================
# NONPARAMETRIC TESTS - TWO SAMPLE
# =============================================================================

def _satterthwaite_df(s1, s2, n1, n2):
    """Compute Satterthwaite-Welch degrees of freedom from variance components."""
    num = (s1 + s2) ** 2
    den = s1 ** 2 / (n1 - 1) + s2 ** 2 / (n2 - 1)
    if den == 0:
        return 1000.0
    return float(num / den)


def brunner_munzel_tost(a, b, lower, upper, alpha=0.05):
    """
    Brunner-Munzel TOST for two independent groups.

    Tests the relative effect θ = P(X > Y) + 0.5·P(X = Y) using
    rank-based analytic variance and Satterthwaite-Welch df.

    Reference: brunner_munzel.R (TOSTER package).

    Parameters
    ----------
    a, b : array-like
        Two groups of observations (NaNs ignored).
    lower, upper : float
        Equivalence bounds on θ (probability scale, in [0, 1]).
    alpha : float
        Significance level.

    Returns
    -------
    p_low, p_high : float
        One-sided TOST p-values.
    details : dict
        theta_hat, se, df, t-statistics, CI, warnings.
    """
    if t_dist is None:
        raise RuntimeError("scipy is required for Brunner-Munzel test.")

    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if x.size == 0 or y.size == 0:
        raise ValueError("Not enough observations for Brunner-Munzel test.")

    n_x = float(x.size)
    n_y = float(y.size)
    N = n_x + n_y

    # Combined ranks
    combined = np.concatenate([x, y])
    rxy = np.argsort(np.argsort(combined)).astype(float) + 1.0  # rank of combined
    # Scipy rankdata handles ties better
    from scipy.stats import rankdata
    rxy = rankdata(combined, method='average')
    rx = rankdata(x, method='average')
    ry = rankdata(y, method='average')

    # Placement values
    pl2 = (1.0 / n_y) * (rxy[:int(n_x)] - rx)  # for x observations
    pl1 = (1.0 / n_x) * (rxy[int(n_x):] - ry)  # for y observations

    # Relative effect estimate: P(X > Y) + 0.5*P(X == Y)
    pd = float(np.mean(pl2))
    # Clamp extreme values
    if pd == 1.0:
        pd = 0.9999
    if pd == 0.0:
        pd = 0.0001

    # Variance components (Brunner & Munzel 2000)
    s1 = float(np.var(pl2, ddof=1)) / n_x
    s2 = float(np.var(pl1, ddof=1)) / n_y

    V = N * (s1 + s2)
    if V == 0:
        V = N * 0.5 / (n_x * n_y) ** 2

    std_err = float(np.sqrt(V / N))

    # Satterthwaite-Welch df
    df_sw = _satterthwaite_df(s1, s2, n_x, n_y)

    # TOST: two one-sided tests
    t_low = float(np.sqrt(N)) * (pd - lower) / float(np.sqrt(V))
    t_high = float(np.sqrt(N)) * (pd - upper) / float(np.sqrt(V))

    p_low = 1.0 - float(t_dist.cdf(t_low, df_sw))   # H0: θ ≤ lower
    p_high = float(t_dist.cdf(t_high, df_sw))         # H0: θ ≥ upper

    # Confidence interval: (1 - 2α) CI for TOST
    ci_lower = pd - float(t_dist.ppf(1.0 - alpha, df_sw)) / float(np.sqrt(N)) * float(np.sqrt(V))
    ci_upper = pd + float(t_dist.ppf(1.0 - alpha, df_sw)) / float(np.sqrt(N)) * float(np.sqrt(V))
    ci_lower = max(0.0, ci_lower)
    ci_upper = min(1.0, ci_upper)

    # Warnings
    warnings = []
    if n_x < 30 or n_y < 30:
        warnings.append(
            "Warning: Sample size is small (n < 30 in at least one group). "
            "The asymptotic approximation may yield liberal p-values. "
            "Consider collecting more data or interpreting results cautiously."
        )

    # TODO: Add permutation test option (test_method="perm") for small samples.
    # Reference: brunner_munzel.R perm_loop function and related permutation code.

    details = {
        "theta_hat": pd,
        "se": std_err,
        "df": df_sw,
        "t_low": t_low,
        "t_high": t_high,
        "n1": int(n_x),
        "n2": int(n_y),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "warnings": warnings,
    }
    return float(p_low), float(p_high), details


def _kde_at_zero(diffs):
    """Estimate density at zero via kernel density estimation."""
    if gaussian_kde is None:
        raise RuntimeError("scipy.stats.gaussian_kde is required.")
    if len(diffs) < 2:
        return None
    try:
        kde = gaussian_kde(diffs)
        return float(kde(0.0)[0])
    except Exception:
        return None


def hodges_lehmann_two_sample_tost(a, b, lower, upper, alpha=0.05):
    """
    Hodges-Lehmann two-sample TOST (asymptotic).

    Tests location shift between two independent groups using the HL2 estimator
    (median of all pairwise differences Y_j - X_i).

    Reference: hodges_lehmann.R (TOSTER package).

    Parameters
    ----------
    a, b : array-like
        Two groups of observations (NaNs ignored).
    lower, upper : float
        Equivalence bounds in raw data units.
    alpha : float
        Significance level.

    Returns
    -------
    p_low, p_high : float
        One-sided TOST p-values.
    details : dict
        HL estimate, SE, z-statistics, CI, warnings.
    """
    if norm is None:
        raise RuntimeError("scipy is required for Hodges-Lehmann test.")

    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if x.size < 2 or y.size < 2:
        raise ValueError("Need at least 2 observations in each group.")

    m = float(x.size)
    n = float(y.size)
    N = m + n

    # HL2 estimator: median of all pairwise differences (y_j - x_i)
    # Sign convention: x - y to match t-test convention (group1 - group2)
    diffs = np.subtract.outer(x, y).ravel()
    hl_estimate = float(np.median(diffs))

    # Asymptotic SE via kernel density estimation
    # Use within-sample pairwise differences for KDE
    diff_x = np.subtract.outer(x, x)
    diff_x = diff_x[np.tril_indices_from(diff_x, k=-1)]
    diff_y = np.subtract.outer(y, y)
    diff_y = diff_y[np.tril_indices_from(diff_y, k=-1)]
    all_diffs = np.concatenate([diff_x, diff_y])

    h0 = _kde_at_zero(all_diffs)
    if h0 is None or h0 <= 0:
        # Fallback: use all pairwise diffs for KDE
        h0 = _kde_at_zero(diffs)
        if h0 is None or h0 <= 0:
            h0 = 0.001  # prevent division by zero

    lam = m / N
    se = 1.0 / (float(np.sqrt(12.0 * lam * (1.0 - lam) * N)) * h0)

    # TOST z-tests
    z_low = (hl_estimate - lower) / se
    z_high = (hl_estimate - upper) / se

    p_low = 1.0 - float(norm.cdf(z_low))   # H0: shift ≤ lower
    p_high = float(norm.cdf(z_high))         # H0: shift ≥ upper

    # Confidence interval (1 - 2α)
    z_crit = float(norm.ppf(1.0 - alpha))
    ci_lower = hl_estimate - z_crit * se
    ci_upper = hl_estimate + z_crit * se

    warnings = []
    if x.size < 30 or y.size < 30:
        warnings.append(
            "Warning: Sample size is small (n < 30 in at least one group). "
            "The asymptotic approximation may be unreliable. "
            "Consider collecting more data or interpreting results cautiously."
        )

    # TODO: Add permutation test option (R parameter) for exact/randomization inference.
    # Reference: hodges_lehmann.R two-sample permutation section with scale estimators S1/S2.

    details = {
        "hl_estimate": hl_estimate,
        "se": se,
        "z_low": z_low,
        "z_high": z_high,
        "n1": int(m),
        "n2": int(n),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "warnings": warnings,
    }
    return float(p_low), float(p_high), details


# =============================================================================
# NONPARAMETRIC TESTS - PAIRED
# =============================================================================

def paired_brunner_munzel_tost(x, y, lower, upper, alpha=0.05):
    """
    Paired Brunner-Munzel TOST for repeated-measures designs.

    Tests the relative effect θ = P(X > Y) + 0.5·P(X = Y) using
    rank-based analytic variance with df = n - 1.

    Reference: brunner_munzel.R (TOSTER package), paired section.

    Parameters
    ----------
    x, y : array-like
        Paired observations. Must have the same length. NaN pairs are removed
        (complete-case: both x[i] and y[i] must be non-missing).
    lower, upper : float
        Equivalence bounds on θ (probability scale, in [0, 1]).
    alpha : float
        Significance level.

    Returns
    -------
    p_low, p_high : float
        One-sided TOST p-values.
    details : dict
        theta_hat, se, df, t_low, t_high, n, ci_lower, ci_upper, warnings.
    """
    if t_dist is None:
        raise RuntimeError("scipy is required for paired Brunner-Munzel test.")
    from scipy.stats import rankdata

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if x.size != y.size:
        raise ValueError("Paired Brunner-Munzel test requires equal-length inputs.")

    # Complete-case: drop pairs where either is NaN
    ok = ~(np.isnan(x) | np.isnan(y))
    x = x[ok]
    y = y[ok]
    n = x.size
    if n < 2:
        raise ValueError("Paired Brunner-Munzel test requires at least 2 complete pairs.")

    N = 2 * n

    # Data setup (R convention: y first, then x)
    all_data = np.concatenate([y, x])
    xinverse = np.concatenate([x, y])

    # Ranking
    rx = rankdata(all_data, method='average')
    rxinverse = rankdata(xinverse, method='average')
    rx1 = rx[:n]         # Ranks of y in combined
    rx2 = rx[n:]         # Ranks of x in combined
    rix1 = rankdata(y, method='average')   # Internal ranks of y
    rix2 = rankdata(x, method='average')   # Internal ranks of x

    # Placement values and relative effect
    BM1 = (1.0 / n) * (rx1 - rix1)
    BM2 = (1.0 / n) * (rx2 - rix2)
    BM3 = BM1 - BM2
    pd = float(np.mean(BM2))  # Relative effect estimate θ̂

    # Variance and standard error (analytic, rank-based)
    m = float(np.mean(BM3))
    v = (np.sum(BM3 ** 2) - n * m ** 2) / (n - 1)
    if v == 0:
        v = 1.0 / n

    std_err = np.sqrt(v / n)

    # Degrees of freedom: n - 1 (NOT Satterthwaite)
    df = n - 1

    # TOST: two one-sided tests
    # Note: uses sqrt(n) not sqrt(N) (key difference from independent case)
    t_low = float(np.sqrt(n)) * (pd - lower) / float(np.sqrt(v))
    t_high = float(np.sqrt(n)) * (pd - upper) / float(np.sqrt(v))

    p_low = 1.0 - float(t_dist.cdf(t_low, df))    # H0: θ ≤ lower
    p_high = float(t_dist.cdf(t_high, df))          # H0: θ ≥ upper

    # Confidence interval: (1 - 2α) CI for equivalence testing
    ci_lower = pd - float(t_dist.ppf(1.0 - alpha, df)) * float(np.sqrt(v / n))
    ci_upper = pd + float(t_dist.ppf(1.0 - alpha, df)) * float(np.sqrt(v / n))
    ci_lower = max(0.0, ci_lower)
    ci_upper = min(1.0, ci_upper)

    # Warnings
    warnings = []
    if n < 30:
        warnings.append(
            "Warning: Sample size is small (n < 30 pairs). The asymptotic approximation may yield liberal "
            "p-values. Consider collecting more data or interpreting results cautiously."
        )

    # TODO: Add permutation test option (test_method="perm") for small samples.
    # Reference: brunner_munzel.R paired permutation section.
    # TODO: Add logit transformation option (test_method="logit") for range-preserving CIs.
    # Reference: brunner_munzel.R paired logit section.

    details = {
        "theta_hat": pd,
        "se": float(std_err),
        "df": df,
        "t_low": t_low,
        "t_high": t_high,
        "n": n,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "warnings": warnings,
    }
    return float(p_low), float(p_high), details


# =============================================================================
# HIDDEN TESTS (kept for future reactivation)
# These functions are not currently exposed in the UI but remain functional.
# =============================================================================

def hodges_lehmann_one_sample_tost(d, lower, upper, alpha=0.05):
    """
    Hodges-Lehmann one-sample/paired TOST (asymptotic).

    For paired data, tests whether the pseudomedian of differences equals a
    specified value. The HL1 estimator is the median of Walsh averages.

    Reference: hodges_lehmann.R (TOSTER package).

    Parameters
    ----------
    d : array-like
        Single column of differences (for paired) or single sample.
    lower, upper : float
        Equivalence bounds in raw data units.
    alpha : float
        Significance level.

    Returns
    -------
    p_low, p_high : float
        One-sided TOST p-values.
    details : dict
        Pseudomedian estimate, SE, z-statistics, CI, warnings.
    """
    if norm is None:
        raise RuntimeError("scipy is required for Hodges-Lehmann test.")

    d = np.asarray(d, dtype=float).ravel()
    d = d[~np.isnan(d)]
    if d.size < 2:
        raise ValueError("Need at least 2 observations.")

    n = float(d.size)

    # HL1 estimator: median of Walsh averages (i <= j)
    pairs = np.add.outer(d, d) / 2.0
    walsh = pairs[np.triu_indices_from(pairs)]
    hl_estimate = float(np.median(walsh))

    # Asymptotic SE via kernel density estimation on within-sample diffs
    diff_mat = np.subtract.outer(d, d)
    diffs = diff_mat[np.tril_indices_from(diff_mat, k=-1)]

    h0 = _kde_at_zero(diffs)
    if h0 is None or h0 <= 0:
        h0 = 0.001  # prevent division by zero

    se = 1.0 / (float(np.sqrt(12.0 * n)) * h0)

    # TOST z-tests
    z_low = (hl_estimate - lower) / se
    z_high = (hl_estimate - upper) / se

    p_low = 1.0 - float(norm.cdf(z_low))
    p_high = float(norm.cdf(z_high))

    # Confidence interval (1 - 2α)
    z_crit = float(norm.ppf(1.0 - alpha))
    ci_lower = hl_estimate - z_crit * se
    ci_upper = hl_estimate + z_crit * se

    warnings = []
    if d.size < 30:
        warnings.append(
            "Warning: Sample size is small (n < 30). "
            "The asymptotic approximation may be unreliable. "
            "Consider collecting more data or interpreting results cautiously."
        )

    details = {
        "pseudomedian": hl_estimate,
        "se": se,
        "z_low": z_low,
        "z_high": z_high,
        "n": int(n),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "warnings": warnings,
    }
    return float(p_low), float(p_high), details


def paired_concordance_tost(d, lower, upper, alpha=0.05):
    """
    Paired concordance TOST for paired/repeated-measures data.

    Tests the concordance probability P(X_after > X_before) + 0.5·P(X_after = X_before).
    This is a probability-based inference method for paired designs, analogous to
    the Brunner-Munzel relative effect but for paired data.

    Reference: perm_ses_test.R (TOSTER package).

    Parameters
    ----------
    d : array-like
        Single column of differences (after - before).
    lower, upper : float
        Equivalence bounds on probability scale [0, 1], centered around 0.5.
    alpha : float
        Significance level.

    Returns
    -------
    p_low, p_high : float
        One-sided TOST p-values.
    details : dict
        Concordance probability, SE, z-statistics, CI, warnings.
    """
    if norm is None:
        raise RuntimeError("scipy is required for paired concordance test.")

    d = np.asarray(d, dtype=float).ravel()
    d = d[~np.isnan(d)]
    if d.size < 1:
        raise ValueError("Not enough observations for paired concordance test.")

    n = float(d.size)

    # Concordance probability
    concordant = float(np.sum(d > 0))
    ties = float(np.sum(d == 0))
    c_prob = (concordant + 0.5 * ties) / n

    # Standard error
    se = float(np.sqrt(c_prob * (1.0 - c_prob) / n))
    if se == 0.0:
        se = 1e-10  # prevent division by zero

    # TOST z-tests
    z_low = (c_prob - lower) / se
    z_high = (c_prob - upper) / se

    p_low = 1.0 - float(norm.cdf(z_low))
    p_high = float(norm.cdf(z_high))

    # Confidence interval (1 - 2α)
    z_crit = float(norm.ppf(1.0 - alpha))
    ci_lower = max(0.0, c_prob - z_crit * se)
    ci_upper = min(1.0, c_prob + z_crit * se)

    warnings = []
    if d.size < 30:
        warnings.append(
            "Warning: Sample size is small (n < 30). "
            "Consider collecting more data or interpreting results cautiously."
        )

    # TODO: Add permutation test option for exact inference with small samples.
    # Reference: perm_ses_test.R permutation implementation.

    details = {
        "concordance_probability": c_prob,
        "se": se,
        "z_low": z_low,
        "z_high": z_high,
        "n": int(n),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "warnings": warnings,
    }
    return float(p_low), float(p_high), details


# =============================================================================
# PROPORTIONS
# =============================================================================

def proportions_tost_difference(x1, n1, x2, n2, lower, upper, alpha):
    """
    Wald z-test TOST for difference in proportions (p1 - p2).

    Reference: two_proportions.R (TOSTER package).

    Parameters
    ----------
    x1, n1 : int
        Successes and total in group 1.
    x2, n2 : int
        Successes and total in group 2.
    lower, upper : float
        Equivalence bounds on p1 - p2.
    alpha : float
        Significance level.

    Returns
    -------
    p_low, p_high : float
        One-sided TOST p-values.
    details : dict
    """
    if norm is None:
        raise RuntimeError("scipy is required for proportions test.")
    x1 = float(x1); x2 = float(x2)
    n1 = int(n1); n2 = int(n2)
    if n1 <= 0 or n2 <= 0:
        raise ValueError("Sample sizes must be positive.")
    p1 = x1 / n1
    p2 = x2 / n2
    diff = p1 - p2
    se = float(np.sqrt(max(0.0, p1 * (1 - p1) / n1) + max(0.0, p2 * (1 - p2) / n2)))

    if se == 0.0:
        z_low = float("inf") if (diff - lower) > 0 else float("-inf")
        z_high = float("inf") if (diff - upper) > 0 else float("-inf")
    else:
        z_low = (diff - lower) / se
        z_high = (diff - upper) / se

    p_low = 1.0 - float(norm.cdf(z_low))
    p_high = float(norm.cdf(z_high))

    # CI (1-2α)
    z_crit = float(norm.ppf(1.0 - alpha))
    ci_lower = diff - z_crit * se if se > 0 else diff
    ci_upper = diff + z_crit * se if se > 0 else diff

    warnings = []
    if n1 < 30 or n2 < 30:
        warnings.append("Warning: Small sample size detected. The Wald approximation may be inaccurate.")
    if min(x1, n1 - x1, x2, n2 - x2) < 5:
        warnings.append("Warning: Cell count < 5 detected. The Wald approximation may be unreliable for sparse tables.")
    if p1 in (0.0, 1.0) or p2 in (0.0, 1.0):
        warnings.append("Warning: Observed proportion is 0 or 1. Standard errors may be undefined or unreliable.")

    details = {
        "effect_size": "difference",
        "estimate": diff,
        "p1": p1, "p2": p2,
        "x1": x1, "n1": n1, "x2": x2, "n2": n2,
        "se": se,
        "z_low": float(z_low) if np.isfinite(z_low) else None,
        "z_high": float(z_high) if np.isfinite(z_high) else None,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "warnings": warnings,
    }
    return float(p_low), float(p_high), details


def proportions_tost_odds_ratio(x1, n1, x2, n2, lower, upper, alpha):
    """
    Wald z-test TOST for odds ratio between two proportions.

    Reference: two_proportions.R (TOSTER package).

    Parameters
    ----------
    x1, n1, x2, n2 : int
        Successes and totals for each group.
    lower, upper : float
        Equivalence bounds on OR (e.g., [0.8, 1.25]).
    alpha : float
        Significance level.

    Returns
    -------
    p_low, p_high : float
        One-sided TOST p-values.
    details : dict
    """
    if norm is None:
        raise RuntimeError("scipy is required for proportions test.")
    x1 = float(x1); x2 = float(x2)
    n1 = int(n1); n2 = int(n2)
    if n1 <= 0 or n2 <= 0:
        raise ValueError("Sample sizes must be positive.")
    p1 = x1 / n1
    p2 = x2 / n2
    q1 = 1.0 - p1
    q2 = 1.0 - p2

    OR = (p1 / q1) / (p2 / q2) if q1 > 0 and q2 > 0 and p2 > 0 else float('inf')

    # SE of log(OR) using Fleiss formula with +0.5 correction
    se_log = float(np.sqrt(
        1.0 / (n1 * p1 + 0.5) + 1.0 / (n1 * q1 + 0.5) +
        1.0 / (n2 * p2 + 0.5) + 1.0 / (n2 * q2 + 0.5)
    ))

    log_or = float(np.log(OR)) if OR > 0 and np.isfinite(OR) else 0.0
    log_lower = float(np.log(lower)) if lower > 0 else float('-inf')
    log_upper = float(np.log(upper)) if upper > 0 else float('inf')

    z_low = (log_or - log_lower) / se_log if se_log > 0 else float('inf')
    z_high = (log_or - log_upper) / se_log if se_log > 0 else float('-inf')

    p_low = 1.0 - float(norm.cdf(z_low))
    p_high = float(norm.cdf(z_high))

    # CI on log scale, then exponentiate
    z_crit = float(norm.ppf(1.0 - alpha))
    ci_lower = float(np.exp(log_or - z_crit * se_log))
    ci_upper = float(np.exp(log_or + z_crit * se_log))

    warnings = []
    if n1 < 30 or n2 < 30:
        warnings.append("Warning: Small sample size detected. The Wald approximation may be inaccurate.")
    if min(x1, n1 - x1, x2, n2 - x2) < 5:
        warnings.append("Warning: Cell count < 5 detected. The Wald approximation may be unreliable for sparse tables.")
    if p1 in (0.0, 1.0) or p2 in (0.0, 1.0):
        warnings.append("Warning: Observed proportion is 0 or 1. Standard errors may be undefined or unreliable.")

    details = {
        "effect_size": "odds_ratio",
        "estimate": OR,
        "p1": p1, "p2": p2,
        "x1": x1, "n1": n1, "x2": x2, "n2": n2,
        "se_log": se_log,
        "z_low": float(z_low) if np.isfinite(z_low) else None,
        "z_high": float(z_high) if np.isfinite(z_high) else None,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "warnings": warnings,
    }
    return float(p_low), float(p_high), details


def proportions_tost_risk_ratio(x1, n1, x2, n2, lower, upper, alpha):
    """
    Wald z-test TOST for risk ratio between two proportions.

    Reference: two_proportions.R (TOSTER package).

    Parameters
    ----------
    x1, n1, x2, n2 : int
        Successes and totals for each group.
    lower, upper : float
        Equivalence bounds on RR (e.g., [0.8, 1.25]).
    alpha : float
        Significance level.

    Returns
    -------
    p_low, p_high : float
        One-sided TOST p-values.
    details : dict
    """
    if norm is None:
        raise RuntimeError("scipy is required for proportions test.")
    x1 = float(x1); x2 = float(x2)
    n1 = int(n1); n2 = int(n2)
    if n1 <= 0 or n2 <= 0:
        raise ValueError("Sample sizes must be positive.")
    p1 = x1 / n1
    p2 = x2 / n2
    q1 = 1.0 - p1
    q2 = 1.0 - p2

    RR = p1 / p2 if p2 > 0 else float('inf')

    # SE of log(RR)
    se_log = float(np.sqrt(q1 / (n1 * p1) + q2 / (n2 * p2))) if p1 > 0 and p2 > 0 else float('inf')

    log_rr = float(np.log(RR)) if RR > 0 and np.isfinite(RR) else 0.0
    log_lower = float(np.log(lower)) if lower > 0 else float('-inf')
    log_upper = float(np.log(upper)) if upper > 0 else float('inf')

    z_low = (log_rr - log_lower) / se_log if se_log > 0 and np.isfinite(se_log) else float('inf')
    z_high = (log_rr - log_upper) / se_log if se_log > 0 and np.isfinite(se_log) else float('-inf')

    p_low = 1.0 - float(norm.cdf(z_low))
    p_high = float(norm.cdf(z_high))

    # CI: RR * exp(±z * SE)
    z_crit = float(norm.ppf(1.0 - alpha))
    if np.isfinite(se_log) and np.isfinite(log_rr):
        ci_lower = RR * float(np.exp(-z_crit * se_log))
        ci_upper = RR * float(np.exp(z_crit * se_log))
    else:
        ci_lower = 0.0
        ci_upper = float('inf')

    warnings = []
    if n1 < 30 or n2 < 30:
        warnings.append("Warning: Small sample size detected. The Wald approximation may be inaccurate.")
    if min(x1, n1 - x1, x2, n2 - x2) < 5:
        warnings.append("Warning: Cell count < 5 detected. The Wald approximation may be unreliable for sparse tables.")
    if p1 in (0.0, 1.0) or p2 in (0.0, 1.0):
        warnings.append("Warning: Observed proportion is 0 or 1. Standard errors may be undefined or unreliable.")

    details = {
        "effect_size": "risk_ratio",
        "estimate": RR,
        "p1": p1, "p2": p2,
        "x1": x1, "n1": n1, "x2": x2, "n2": n2,
        "se_log": se_log,
        "z_low": float(z_low) if np.isfinite(z_low) else None,
        "z_high": float(z_high) if np.isfinite(z_high) else None,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "warnings": warnings,
    }
    return float(p_low), float(p_high), details

def proportions_wald_test_difference(x1, n1, x2, n2, pooled: bool = False, alternative: str = "two-sided") -> float:
    """
    Wald z-test for the difference in two independent proportions.

    This is the companion to `proportions_tost_difference`, which uses
    the (unpooled) Wald standard error for the risk difference (p1 - p2).
    """
    if norm is None:
        raise RuntimeError("scipy is required for proportions test.")

    x1 = float(x1); x2 = float(x2)
    n1 = int(n1); n2 = int(n2)
    if n1 <= 0 or n2 <= 0:
        raise ValueError("Sample sizes must be positive.")
    if x1 < 0 or x2 < 0 or x1 > n1 or x2 > n2:
        raise ValueError("Counts must satisfy 0 <= x <= n.")

    p1 = x1 / n1
    p2 = x2 / n2
    diff = p1 - p2

    if pooled:
        p_pool = (x1 + x2) / (n1 + n2)
        se = float(np.sqrt(max(0.0, p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))))
    else:
        se = float(np.sqrt(max(0.0, p1 * (1.0 - p1) / n1) + max(0.0, p2 * (1.0 - p2) / n2)))

    if se == 0.0:
        z = float("inf") if diff > 0 else (float("-inf") if diff < 0 else 0.0)
    else:
        z = diff / se

    alt = (alternative or "two-sided").lower()
    if alt == "two-sided":
        return float(2.0 * (1.0 - norm.cdf(abs(z))))
    if alt == "greater":
        return float(1.0 - norm.cdf(z))
    if alt == "less":
        return float(norm.cdf(z))
    raise ValueError("alternative must be one of {'two-sided','greater','less'}")