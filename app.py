# app.py — Streamlit UI (imports helpers + io_utils)
import re
import traceback
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st
import backend as be

# optional scipy for conventional two-sided p-values
try:
    from scipy import stats
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from helpers import describe_tost, update_parsed_counts
import io_utils as io_utils  # provides parse_table_from_text, read_table_from_bytes wrappers

st.set_page_config(page_title="TEACUP — Equivalence tests", layout="wide")
st.title("TEACUP — Equivalence testing (TOST)")

# Sidebar
with st.sidebar:
    st.header("Test selection")

    TEST_GROUPS = {
        "T-tests": [
            "Welch t-test (two-sample)",
            "Pooled t-test (two-sample)",
            "One-sample t-test",
        ],
        "Nonparametric (two-sample)": [
            "Brunner-Munzel test",
            # "Hodges-Lehmann (two-sample)",  # Hidden for now
        ],
        "Nonparametric (paired)": [
            "Paired Brunner-Munzel test",
            # "Paired concordance test",      # Hidden for now
            # "Hodges-Lehmann (paired)",       # Hidden for now
        ],
        "Proportions": [
            "Two-sample proportions",
        ],
    }

    group = st.selectbox("Category", list(TEST_GROUPS.keys()), index=0)
    test_choice = st.selectbox("Test", TEST_GROUPS[group], index=0)

    st.markdown("---")
    header_override = st.selectbox("Header detection override", ["Auto-detect", "Force header", "Force no header"])    
    st.markdown("---")
    st.caption("**Links**")
    st.link_button("Code and instructions on GitHub", "https://github.com/jtmff/teacup")
    st.link_button("Buy me a coffee ☕", "https://buymeacoffee.com/teacup_equivtest")
    st.markdown("---")
    st.caption("Contact")
    st.write("Email: jakub.tomek@dpag.ox.ac.uk")


left_col, right_col = st.columns([2, 1])

# canonical session keys
for k, default in {
    "df_uploaded": None,
    "parsed_counts": None,
    "paste_text": "",
    "df_from_paste": None,
    "summary_inputs": None,
    "input_mode": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = default

TEST_DESCRIPTIONS = {
    "Welch t-test (two-sample)": (
        "Two-sample Welch t-test — tests difference in group means for two independent groups "
        "without assuming equal variances. Recommended for most parametric cases."
    ),
    "Pooled t-test (two-sample)": (
        "Two-sample pooled t-test — tests difference in group means for two independent groups.\n\n"
        "**Note:** The pooled (equal variance) t-test assumes equal variances between groups, which is "
        "rarely justified in practice. The Welch t-test is recommended as it performs well regardless "
        "of whether variances are equal."
    ),
    "One-sample t-test": (
        "One-sample t-test — tests whether the sample mean (or mean difference) differs from a "
        "reference value (mu0). Data may be provided as a single column of differences, or "
        "two columns with the same number of rows, interpreted as values before and after treatment (in a paired design)."
    ),
    "Brunner-Munzel test": (
        "Brunner-Munzel test — a nonparametric test comparing two independent groups based on the "
        "relative effect (stochastic superiority). Tests the probability that a randomly drawn "
        "observation from group 1 exceeds one from group 2. Equivalence bounds are on the probability "
        "scale [0, 1], centered around 0.5 (no effect). For example, bounds of [0.35, 0.65] test "
        "whether the probability falls within 15 percentage points of 0.5."
    ),
    # "Hodges-Lehmann (two-sample)": (  # Hidden for now
    #     "Hodges-Lehmann test (two-sample) — a robust nonparametric test for location shift between "
    #     "two independent groups. Estimates the median of all pairwise differences between groups. "
    #     "More robust to outliers than the t-test. Equivalence bounds are in the original data units "
    #     "(e.g., [-0.5, 0.5] tests whether the location shift is within ±0.5 units)."
    #     "The Mann-Whitney U test is used to calculate p-value for differences."
    # ),
    "Paired Brunner-Munzel test": (
        "Paired Brunner-Munzel test — a nonparametric test for paired/repeated-measures "
        "data based on the relative effect (stochastic superiority). Tests the probability "
        "that a randomly sampled value from one condition exceeds one from the other, "
        "accounting for the paired design in variance estimation. Equivalence bounds are "
        "on the probability scale [0, 1], centered around 0.5 (no effect). "
        "Requires two columns of equal length representing paired observations."
    ),
    # "Paired concordance test": (  # Hidden for now
    #     "Paired concordance test — a nonparametric test for paired/repeated measures data based on "
    #     "the probability of improvement. Tests whether the probability that a post-treatment value "
    #     "exceeds its paired pre-treatment value differs from chance (0.5). Equivalence bounds are "
    #     "on the probability scale [0, 1]. Data may be provided as a single column of differences, or "
    #     "two columns with the same number of rows, interpreted as values before and after treatment (in a paired design)."
    # ),
    # "Hodges-Lehmann (paired)": (  # Hidden for now
    #     "Hodges-Lehmann test (paired/one-sample) — a robust nonparametric test for the pseudomedian "
    #     "of paired differences or a single sample. The pseudomedian (median of Walsh averages) is a "
    #     "robust location estimator. Equivalence bounds are in the original data units. Data may be provided as a single column of differences, or "
    #     "two columns with the same number of rows, interpreted as values before and after treatment (in a paired design)."
    #     "The Wilcoxon signed-rank test is used to calculate p-value for differences."
    # ),
    "Two-sample proportions": (
        "Two-sample proportions test — compares proportions between two independent groups using a "
        "Wald z-test. Can test equivalence with regards to difference in proportions, odds ratio, or risk ratio. Requires "
        "a 2×2 contingency table of counts. For difference: bounds are in proportion units (e.g., "
        "[-0.1, 0.1] = 10 percentage points). For OR/RR: bounds are ratios (e.g., [0.8, 1.25]). "
        "The same Wald z-test is also used to calculate the p-value for differences. "
    ),
}

UPLOAD_HINTS = {
    "Welch t-test (two-sample)": "Upload CSV/XLS/XLSX with two columns (group1, group2) of numeric values.",
    "Pooled t-test (two-sample)": "Upload CSV/XLS/XLSX with two columns (group1, group2) of numeric values.",
    "One-sample t-test": "Upload CSV/XLS/XLSX with one column OR two columns (col2 - col1).",
    "Brunner-Munzel test": "Upload CSV/XLS/XLSX with two columns (group1, group2) of numeric values.",
    # "Hodges-Lehmann (two-sample)": "Upload CSV/XLS/XLSX with two columns (group1, group2) of numeric values.",  # Hidden
    "Paired Brunner-Munzel test": "Upload CSV/XLS/XLSX with two equal-length columns of paired numeric values.",
    # "Paired concordance test": "Upload CSV/XLS/XLSX with one column of differences OR two columns (col2 - col1).",  # Hidden
    # "Hodges-Lehmann (paired)": "Upload CSV/XLS/XLSX with one column of differences OR two columns (col2 - col1).",  # Hidden
    "Two-sample proportions": "Upload CSV/XLS/XLSX with a 2×2 contingency table of counts.",
}

# Protect against huge uploads (simple safeguard)
MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB

# Tests that require two independent columns
TWO_SAMPLE_TESTS = {
    "Welch t-test (two-sample)",
    "Pooled t-test (two-sample)",
    "Brunner-Munzel test",
    # "Hodges-Lehmann (two-sample)",  # Hidden
}

# Tests that accept a single column (or two columns differenced)
ONE_SAMPLE_TESTS = {
    "One-sample t-test",
    # "Paired concordance test",    # Hidden
    # "Hodges-Lehmann (paired)",    # Hidden
}

# Tests that require two equal-length paired columns
PAIRED_TWO_COL_TESTS = {
    "Paired Brunner-Munzel test",
}

def one_sample_vector_from_df(df: pd.DataFrame) -> np.ndarray:
    """
    Accepts either:
      - 1 column: uses that as the one-sample vector
      - >=2 columns: computes differences as (2nd - 1st) using first two columns
    Returns a 1D numpy array with NaNs removed.
    """
    if df is None or df.shape[1] < 1:
        raise ValueError("Input requires at least one column.")

    # Two-column mode => interpret as paired values and convert to differences
    if df.shape[1] >= 2:
        df2 = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce")
        a = df2.iloc[:, 0].to_numpy()
        b = df2.iloc[:, 1].to_numpy()
        diffs = b - a  # (2nd - 1st)
        diffs = diffs[~np.isnan(diffs)]
        return diffs.astype(float, copy=False)

    # One-column mode
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
    s = s[~np.isnan(s)]
    return s.astype(float, copy=False)

def ci_heading(alpha: float, details: Optional[dict] = None) -> str:
    """Return the CI heading that corresponds to the TOST alpha."""
    try:
        level = float((details or {}).get("ci_level", 1.0 - 2.0 * alpha))
    except Exception:
        level = 1.0 - 2.0 * alpha
    pct = 100.0 * level
    if abs(pct - round(pct)) < 1e-9:
        pct_text = f"{int(round(pct))}%"
    else:
        pct_text = f"{pct:.1f}%"
    return f"{pct_text} confidence interval"


def ci_estimate_label(test_choice: str, details: Optional[dict] = None, effect_size_option: Optional[str] = None) -> str:
    """Human-readable effect estimate label for the CI output."""
    if test_choice == "One-sample t-test":
        return "Mean difference"
    if test_choice in ("Welch t-test (two-sample)", "Pooled t-test (two-sample)"):
        return "Mean difference (group 1 - group 2)"
    if test_choice in ("Brunner-Munzel test", "Paired Brunner-Munzel test"):
        return "Relative effect (theta)"
    if test_choice == "Two-sample proportions":
        if effect_size_option == "Difference in proportions":
            return "Difference in proportions (group 1 - group 2)"
        if effect_size_option == "Odds ratio":
            return "Odds ratio"
        if effect_size_option == "Risk ratio":
            return "Risk ratio"
        effect = (details or {}).get("effect_size", "estimate")
        return str(effect).replace("_", " ").title()
    return "Estimate"


def fmt_ci_value(value, digits: int = 6) -> str:
    """Format finite and infinite CI values safely for Streamlit output."""
    try:
        v = float(value)
    except Exception:
        return "NA"
    if np.isposinf(v):
        return "inf"
    if np.isneginf(v):
        return "-inf"
    if not np.isfinite(v):
        return "NA"
    return f"{v:.{digits}g}"


def render_equivalence_ci(details: Optional[dict], alpha: float, test_choice: str, effect_size_option: Optional[str] = None):
    """Render the TOST-equivalent CI if the backend returned it."""
    if not isinstance(details, dict):
        st.write("Confidence interval not available for this result.")
        return
    ci_low = details.get("ci_lower")
    ci_high = details.get("ci_upper")
    if ci_low is None or ci_high is None:
        st.write("Confidence interval not available for this result.")
        return
    label = ci_estimate_label(test_choice, details, effect_size_option)
    st.markdown(f"### {ci_heading(alpha, details)}")
    st.write(f"{label}: **[{fmt_ci_value(ci_low)}, {fmt_ci_value(ci_high)}]**")

# --- generate callback: safe to modify session_state inside this function ---
def generate_sample_data():
    rng = np.random.default_rng()
    def gen_norm(mean, sd, n):
        return rng.normal(loc=mean, scale=sd, size=n)

    if test_choice in TWO_SAMPLE_TESTS:
        n1 = rng.integers(8, 13)
        n2 = rng.integers(8, 13)
        a = gen_norm(2.0, 0.35, n1)
        b = gen_norm(2.0, 0.35, n2)
        rows = []
        maxn = max(n1, n2)
        for i in range(maxn):
            v1 = f"{a[i]:.6g}" if i < n1 else ""
            v2 = f"{b[i]:.6g}" if i < n2 else ""
            rows.append(f"{v1},{v2}")
        generated_text = "\n".join(rows)
    elif test_choice in PAIRED_TWO_COL_TESTS:
        n = rng.integers(8, 13)
        a = gen_norm(2.0, 0.35, n)
        b = gen_norm(2.0, 0.35, n)  # Same n for paired
        rows = [f"{a[i]:.6g},{b[i]:.6g}" for i in range(n)]
        generated_text = "\n".join(rows)
    elif test_choice in ONE_SAMPLE_TESTS:
        n = rng.integers(8, 13)
        diffs = gen_norm(0.0, 0.35, n)
        rows = [f"{diffs[i]:.6g}" for i in range(n)]
        generated_text = "\n".join(rows)
    elif test_choice == "Two-sample proportions":
        generated_text = "success,failure\n10,990\n50,950"
    else:
        n1 = 10
        n2 = 10
        a = gen_norm(2.0, 0.35, n1)
        b = gen_norm(2.0, 0.35, n2)
        rows = [f"{a[i]:.6g},{b[i]:.6g}" for i in range(10)]
        generated_text = "\n".join(rows)

    # write generated CSV into text area via session state
    st.session_state["paste_text"] = generated_text
    # parse immediately and update parsed df & counts
    try:
        df_parsed, hdr = io_utils.parse_table_from_text(generated_text, header_override=header_override)
        st.session_state["df_from_paste"] = df_parsed.copy()
        st.session_state["parsed_counts"] = update_parsed_counts(df_parsed, test_choice)
    except Exception:
        st.session_state["df_from_paste"] = None
        st.session_state["parsed_counts"] = None

# UI left column
with left_col:
    st.subheader(test_choice)
    st.markdown(TEST_DESCRIPTIONS.get(test_choice, ""))

    summary_allowed_tests = {
        "Welch t-test (two-sample)",
        "Pooled t-test (two-sample)",
        "One-sample t-test",
    }
    summary_allowed = test_choice in summary_allowed_tests

    input_modes = ["Upload file", "Paste table (text box)"]
    if summary_allowed:
        input_modes.append("Summary statistics (manual)")

    if st.session_state.get("input_mode") not in input_modes:
        st.session_state["input_mode"] = input_modes[0]

    input_mode = st.radio("Input mode", input_modes, horizontal=True, key="input_mode")

    if input_mode == "Upload file":
        hint = UPLOAD_HINTS.get(test_choice, "Upload CSV / XLS / XLSX.")
        uploaded_file = st.file_uploader(hint, type=["csv", "xls", "xlsx"], key="uploader_main")
        if uploaded_file is not None:
            try:
                uploaded_file.seek(0)
                content = uploaded_file.read()
                if len(content) > MAX_UPLOAD_BYTES:
                    st.error(
                        f"Uploaded file too large (> {MAX_UPLOAD_BYTES/(1024*1024):.1f} MB). "
                        "Please upload a smaller file."
                    )
                    st.session_state["df_uploaded"] = None
                    st.session_state["parsed_counts"] = None
                else:
                    df_new, hdr = io_utils.read_table_from_bytes(
                        content,
                        getattr(uploaded_file, "name", "uploaded_file"),
                        header_override
                    )
                    st.session_state["df_uploaded"] = df_new.copy()
                    st.session_state["parsed_counts"] = update_parsed_counts(df_new, test_choice)
            except Exception as e:
                st.error(f"Error reading uploaded file: {e}")
                st.session_state["df_uploaded"] = None
                st.session_state["parsed_counts"] = None
                if show_debug:
                    with st.expander("Traceback (debug)", expanded=False):
                        st.code(traceback.format_exc())

    elif input_mode == "Paste table (text box)":
        st.markdown(
            "Paste a table into the box below (tab- or comma-separated). "
            "The table will be parsed immediately and used when you click **Run equivalence test**."
        )

        # text_area binds to st.session_state["paste_text"]
        st.text_area(
            "Paste table here (tab or comma separated).",
            key="paste_text",
            height=240,
        )

        st.button("Generate sample data", on_click=generate_sample_data)

        # If there is pasted text, parse and update parsed state
        current_text = st.session_state.get("paste_text", "")
        if isinstance(current_text, str) and current_text.strip():
            try:
                df_parsed, hdr = io_utils.parse_table_from_text(current_text, header_override=header_override)
                st.session_state["df_from_paste"] = df_parsed.copy()
                st.session_state["parsed_counts"] = update_parsed_counts(df_parsed, test_choice)
            except Exception:
                st.session_state["df_from_paste"] = None
                st.session_state["parsed_counts"] = None

    else:
        # Summary statistics branch
        st.markdown("Enter summary statistics manually.")
        prev = st.session_state.get("summary_inputs") or {}
        if test_choice in ("Welch t-test (two-sample)", "Pooled t-test (two-sample)"):
            st.markdown("**Group 1**")
            m1 = st.number_input("Mean (group 1)", value=float(prev.get("m1", 0.0)))
            sd1 = st.number_input("SD (group 1)", min_value=0.0, value=float(prev.get("sd1", 1.0)))
            n1 = st.number_input("n (group 1)", min_value=2, value=int(prev.get("n1", 20)))
            st.markdown("**Group 2**")
            m2 = st.number_input("Mean (group 2)", value=float(prev.get("m2", 0.0)))
            sd2 = st.number_input("SD (group 2)", min_value=0.0, value=float(prev.get("sd2", 1.0)))
            n2 = st.number_input("n (group 2)", min_value=2, value=int(prev.get("n2", 20)))
            st.session_state["summary_inputs"] = {
                "m1": m1, "sd1": sd1, "n1": int(n1),
                "m2": m2, "sd2": sd2, "n2": int(n2),
            }
            st.session_state["parsed_counts"] = {"group1": int(n1), "group2": int(n2)}
        elif test_choice == "One-sample t-test":
            m = st.number_input("Sample mean", value=float(prev.get("m", 0.0)))
            sd = st.number_input("Sample SD", min_value=0.0, value=float(prev.get("sd", 1.0)))
            n = st.number_input("n", min_value=2, value=int(prev.get("n", 20)))
            st.session_state["summary_inputs"] = {"m": m, "sd": sd, "n": int(n)}
            st.session_state["parsed_counts"] = {"values": int(n)}

# RIGHT COLUMN: test parameters and Run logic
with right_col:
    st.subheader("Test parameters")

    # Probability-scale tests
    if test_choice in ("Brunner-Munzel test", "Paired Brunner-Munzel test"):
        alpha = st.number_input("Alpha level", min_value=1e-6, max_value=0.5, value=0.05, step=0.01, format="%.3f")
        lower = st.number_input("Lower equivalence boundary (L, θ in [0,1])", min_value=0.0, max_value=1.0, value=0.35, format="%.6f")
        upper = st.number_input("Upper equivalence boundary (U, θ in [0,1])", min_value=0.0, max_value=1.0, value=0.65, format="%.6f")
    elif test_choice == "Two-sample proportions":
        alpha = st.number_input("Alpha level", min_value=1e-6, max_value=0.5, value=0.05, step=0.01, format="%.3f")
        effect_size_option = st.selectbox(
            "Effect size measure",
            ["Difference in proportions", "Odds ratio", "Risk ratio"],
            index=0,
        )
        if effect_size_option == "Difference in proportions":
            lower = st.number_input("Lower equivalence boundary (L)", value=-0.1, format="%.6f")
            upper = st.number_input("Upper equivalence boundary (U)", value=0.1, format="%.6f")
        else:
            lower = st.number_input("Lower equivalence boundary (L)", value=0.8, format="%.6f")
            upper = st.number_input("Upper equivalence boundary (U)", value=1.25, format="%.6f")
    else:
        # Location-based tests (t-tests, Hodges-Lehmann)
        alpha = st.number_input("Alpha level", min_value=1e-6, max_value=0.5, value=0.05, step=0.01, format="%.3f")
        lower = st.number_input("Lower equivalence boundary (L)", value=-0.5, format="%.6f")
        upper = st.number_input("Upper equivalence boundary (U)", value=0.5, format="%.6f")

    if test_choice == "One-sample t-test":
        mu0 = st.number_input("Reference value (mu0)", value=0.0, format="%.6f")
    else:
        mu0 = 0.0

    run_button = st.button("Run equivalence test")

results_placeholder = st.empty()

try:
    if run_button:
        with results_placeholder.container():
            if lower >= upper:
                raise ValueError("Lower equivalence boundary must be smaller than upper boundary.")
            if not (0 < alpha < 1):
                raise ValueError("Alpha must be between 0 and 1.")

            df_use: Optional[pd.DataFrame] = None
            used_mode = None
            if input_mode == "Upload file":
                df_use = st.session_state.get("df_uploaded")
                used_mode = "upload"
                if df_use is None:
                    raise ValueError("No uploaded file available. Upload a file or switch input mode.")
            elif input_mode == "Paste table (text box)":
                df_use = st.session_state.get("df_from_paste")
                used_mode = "paste"
                if df_use is None:
                    raise ValueError("No pasted data found. Paste a table into the textbox and click Run.")
            else:
                used_mode = "summary"
                summ = st.session_state.get("summary_inputs")
                if summ is None:
                    raise ValueError("No summary statistics provided. Enter required inputs in the left pane.")

            pc = st.session_state.get("parsed_counts")
            # explicit handling for parsed_counts (avoid ambiguous DataFrame truth checks)
            if pc is not None:
                try:
                    if isinstance(pc, dict):
                        st.write("Parsed counts — " + ", ".join([f"{k}: {v}" for k, v in pc.items()]))
                    elif isinstance(pc, pd.DataFrame):
                        st.write(f"Parsed counts dataframe: {pc.shape[0]} rows, {pc.shape[1]} cols.")
                    else:
                        st.write("Parsed counts available.")
                except Exception:
                    st.write("Parsed counts available (unable to format).")

            # run selected test (backend calls)
            p_diff = None
            p_low = p_high = None
            details = {}
            test_warnings = []

            # initialize data holders for summary creation
            x = y = None
            paired_a = paired_b = None
            summ_inputs = None

            if used_mode in ("upload", "paste"):
                # ---- Parametric two-sample tests ----
                if test_choice == "Pooled t-test (two-sample)":
                    if df_use.shape[1] < 2:
                        raise ValueError("Two-sample tests require two columns (group1, group2).")
                    df_proc = df_use.iloc[:, :2].apply(pd.to_numeric, errors="coerce")
                    x = df_proc.iloc[:, 0].dropna().values
                    y = df_proc.iloc[:, 1].dropna().values
                    p_low, p_high, details = be.two_sample_pooled_tost(x, y, lower, upper, alpha)
                    if _HAS_SCIPY:
                        _, pval = stats.ttest_ind(x, y, equal_var=True, nan_policy="omit")
                        p_diff = float(pval)

                elif test_choice == "Welch t-test (two-sample)":
                    if df_use.shape[1] < 2:
                        raise ValueError("Two-sample tests require two columns (group1, group2).")
                    df_proc = df_use.iloc[:, :2].apply(pd.to_numeric, errors="coerce")
                    x = df_proc.iloc[:, 0].dropna().values
                    y = df_proc.iloc[:, 1].dropna().values
                    p_low, p_high, details = be.two_sample_welch_tost(x, y, lower, upper, alpha)
                    if _HAS_SCIPY:
                        _, pval = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
                        p_diff = float(pval)

                elif test_choice == "One-sample t-test":
                    # 1 col: values; 2 cols: (col2 - col1)
                    x = one_sample_vector_from_df(df_use)
                    if x.size < 2:
                        raise ValueError("One-sample t-test requires at least 2 non-missing observations.")
                    p_low, p_high, details = be.one_sample_tost(x, mu0, lower, upper, alpha)
                    if _HAS_SCIPY:
                        _, pval = stats.ttest_1samp(x, popmean=mu0, nan_policy="omit")
                        p_diff = float(pval)

                # ---- Nonparametric two-sample tests ----
                elif test_choice == "Brunner-Munzel test":
                    if df_use.shape[1] < 2:
                        raise ValueError("Brunner-Munzel test requires two columns (group1, group2).")
                    df_proc = df_use.iloc[:, :2].apply(pd.to_numeric, errors="coerce")
                    x = df_proc.iloc[:, 0].dropna().values
                    y = df_proc.iloc[:, 1].dropna().values
                    p_low, p_high, details = be.brunner_munzel_tost(x, y, lower, upper, alpha)
                    test_warnings = details.get("warnings", [])
                    if _HAS_SCIPY:
                        try:
                            _, pval = stats.mannwhitneyu(x, y, alternative="two-sided")
                        except TypeError:
                            _, pval = stats.mannwhitneyu(x, y)
                        p_diff = float(pval)

                # elif test_choice == "Hodges-Lehmann (two-sample)":  # Hidden for now
                #     if df_use.shape[1] < 2:
                #         raise ValueError("Two-sample tests require two columns (group1, group2).")
                #     df_proc = df_use.iloc[:, :2].apply(pd.to_numeric, errors="coerce")
                #     x = df_proc.iloc[:, 0].dropna().values
                #     y = df_proc.iloc[:, 1].dropna().values
                #     p_low, p_high, details = be.hodges_lehmann_two_sample_tost(x, y, lower, upper, alpha)
                #     test_warnings = details.get("warnings", [])
                #     if _HAS_SCIPY:
                #         try:
                #             _, pval = stats.mannwhitneyu(x, y, alternative="two-sided")
                #         except TypeError:
                #             _, pval = stats.mannwhitneyu(x, y)
                #         p_diff = float(pval)

                # ---- Nonparametric paired tests ----
                elif test_choice == "Paired Brunner-Munzel test":
                    if df_use.shape[1] < 2:
                        raise ValueError("Paired Brunner-Munzel test requires two columns of paired observations.")
                    df_proc = df_use.iloc[:, :2].apply(pd.to_numeric, errors="coerce")
                    # Complete-case: drop rows where either value is missing
                    df_proc = df_proc.dropna()
                    if df_proc.shape[0] < 2:
                        raise ValueError("Paired Brunner-Munzel test requires at least 2 complete pairs.")
                    x = df_proc.iloc[:, 0].values
                    y = df_proc.iloc[:, 1].values
                    if len(x) != len(y):
                        raise ValueError("Paired Brunner-Munzel test requires equal-length columns.")
                    p_low, p_high, details = be.paired_brunner_munzel_tost(x, y, lower, upper, alpha)
                    test_warnings = details.get("warnings", [])
                    # Two-sided test: use Mann-Whitney as a reference p-value
                    if _HAS_SCIPY:
                        try:
                            _, pval = stats.mannwhitneyu(x, y, alternative="two-sided")
                        except TypeError:
                            _, pval = stats.mannwhitneyu(x, y)
                        p_diff = float(pval)

                # elif test_choice == "Paired concordance test":  # Hidden for now
                #     diffs = one_sample_vector_from_df(df_use)
                #     x = diffs
                #     if diffs.size < 1:
                #         raise ValueError("Paired concordance test requires at least 1 non-missing difference.")
                #     p_low, p_high, details = be.paired_concordance_tost(diffs, lower, upper, alpha)
                #     test_warnings = details.get("warnings", [])

                # elif test_choice == "Hodges-Lehmann (paired)":  # Hidden for now
                #     diffs = one_sample_vector_from_df(df_use)
                #     x = diffs
                #     if diffs.size < 1:
                #         raise ValueError("Hodges-Lehmann (paired) requires at least 1 non-missing difference.")
                #     p_low, p_high, details = be.hodges_lehmann_one_sample_tost(diffs, lower, upper, alpha)
                #     test_warnings = details.get("warnings", [])

                # ---- Proportions ----
                elif test_choice == "Two-sample proportions":
                    if pc is None or not isinstance(pc, dict) or not all(k in pc for k in ("x1", "n1", "x2", "n2")):
                        raise ValueError(
                            "Two-sample proportions test requires a 2×2 contingency table of counts (see upload/paste hint)."
                        )
                    x1 = int(pc["x1"]); n1_p = int(pc["n1"])
                    x2 = int(pc["x2"]); n2_p = int(pc["n2"])
                    summ_inputs = {"x1": x1, "n1": n1_p, "x2": x2, "n2": n2_p}

                    if effect_size_option == "Difference in proportions":
                        p_low, p_high, details = be.proportions_tost_difference(x1, n1_p, x2, n2_p, lower, upper, alpha)
                    elif effect_size_option == "Odds ratio":
                        p_low, p_high, details = be.proportions_tost_odds_ratio(x1, n1_p, x2, n2_p, lower, upper, alpha)
                    else:
                        p_low, p_high, details = be.proportions_tost_risk_ratio(x1, n1_p, x2, n2_p, lower, upper, alpha)

                    test_warnings = details.get("warnings", [])
		    
	            # Test for differences
                    p_diff = float(be.proportions_wald_test_difference(x1, n1_p, x2, n2_p, pooled=False, alternative="two-sided"))


                else:
                    raise ValueError("Selected test not implemented.")
            else:
                # summary-mode (only t-tests supported)
                if test_choice == "Pooled t-test (two-sample)":
                    summ = st.session_state.get("summary_inputs", {})
                    summ_inputs = summ
                    p_low, p_high, details = be.two_sample_pooled_tost_from_summary(
                        summ["m1"], summ["sd1"], summ["n1"],
                        summ["m2"], summ["sd2"], summ["n2"],
                        lower, upper, alpha
                    )
                    if _HAS_SCIPY:
                        res = stats.ttest_ind_from_stats(
                            mean1=summ["m1"], std1=summ["sd1"], nobs1=summ["n1"],
                            mean2=summ["m2"], std2=summ["sd2"], nobs2=summ["n2"],
                            equal_var=True
                        )
                        p_diff = float(res.pvalue)
                elif test_choice == "Welch t-test (two-sample)":
                    summ = st.session_state.get("summary_inputs", {})
                    summ_inputs = summ
                    p_low, p_high, details = be.two_sample_welch_tost_from_summary(
                        summ["m1"], summ["sd1"], summ["n1"],
                        summ["m2"], summ["sd2"], summ["n2"],
                        lower, upper, alpha
                    )
                    if _HAS_SCIPY:
                        res = stats.ttest_ind_from_stats(
                            mean1=summ["m1"], std1=summ["sd1"], nobs1=summ["n1"],
                            mean2=summ["m2"], std2=summ["sd2"], nobs2=summ["n2"],
                            equal_var=False
                        )
                        p_diff = float(res.pvalue)
                elif test_choice == "One-sample t-test":
                    summ = st.session_state.get("summary_inputs", {})
                    summ_inputs = summ
                    p_low, p_high, details = be.one_sample_tost_from_summary(
                        summ["m"], summ["sd"], summ["n"],
                        mu0, lower, upper, alpha
                    )
                    try:
                        se = summ["sd"] / (summ["n"] ** 0.5)
                        tstat = (summ["m"] - mu0) / se
                        df_ = summ["n"] - 1
                        if _HAS_SCIPY:
                            p_diff = float(2.0 * stats.t.sf(abs(tstat), df_))
                    except Exception:
                        p_diff = None
                else:
                    raise ValueError("Summary-mode is not available for this test; upload/paste raw data instead.")

            equiv = (p_low < alpha) and (p_high < alpha)

            st.markdown("## Results")

            # Show test-specific warnings
            for w in test_warnings:
                st.warning(w)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### One-sided TOST p-values")
                st.write(f"p(lower) = **{p_low:.6g}**")
                st.write(f"p(upper) = **{p_high:.6g}**")
                render_equivalence_ci(details, alpha, test_choice, globals().get("effect_size_option"))
                st.markdown("### Difference test (two-sided)")
                if p_diff is None:
                    if not _HAS_SCIPY:
                        st.warning("scipy not available — cannot compute conventional two-sided p-value (install scipy to enable).")
                    else:
                        st.write("Could not compute difference p-value for this input.")
                else:
                    st.write(f"p (difference, two-sided) = **{p_diff:.6g}**")
            with c2:
                st.markdown("### Decision")
                if equiv:
                    st.success(f"Equivalent at α = {alpha}")
                else:
                    st.error(f"Not significantly equivalent at α = {alpha}")

            # Plain-language summary (describe-style)
            summary_inputs_for_helper = summ_inputs if ('summ_inputs' in locals() and summ_inputs is not None) else st.session_state.get("summary_inputs")
            summary_text = describe_tost(
                test_choice=test_choice,
                p_low=p_low,
                p_high=p_high,
                p_diff=p_diff,
                lower=lower,
                upper=upper,
                alpha=alpha,
                x=x,
                y=y,
                paired_a=paired_a,
                paired_b=paired_b,
                summ=summary_inputs_for_helper,
                mu0=mu0,
                digits=3,
            )

            st.markdown("### Plain-language summary")
            st.write(summary_text)

            st.markdown("### Data preview (first 10 rows or parsed counts)")
            if used_mode in ("upload", "paste"):
                if test_choice == "Two-sample proportions":
                    # only print parsed_counts if it's the expected dict
                    if pc is not None and isinstance(pc, dict):
                        st.write(pc)
                    else:
                        st.write("Parsed counts not available in the expected form; previewing raw data.")
                        df_to_show = st.session_state.get("df_from_paste") or st.session_state.get("df_uploaded")
                        if df_to_show is not None:
                            st.dataframe(df_to_show.head(10))
                        else:
                            st.write("No data available to preview.")
                else:
                    # choose the df to preview safely
                    _df_from_paste = st.session_state.get("df_from_paste")
                    _df_uploaded = st.session_state.get("df_uploaded")
                    df_to_show = _df_from_paste if _df_from_paste is not None else _df_uploaded
                    if df_to_show is not None:
                        st.dataframe(df_to_show.head(10))
                    else:
                        st.write("No data available to preview.")
            else:
                st.write("Summary-mode — see inputs")

            st.markdown("### Diagnostic details")
            # Filter out non-serializable items for JSON display
            display_details = {k: v for k, v in details.items() if k != "warnings"}
            st.json(display_details, expanded=False)

except Exception as e:
    if show_debug:
        st.error(f"Error: {e}")
        st.subheader("Traceback (debug)")
        st.code(traceback.format_exc())
    else:
        st.error(f"Error: {e}")
