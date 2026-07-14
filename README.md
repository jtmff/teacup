# TEACUP — Equivalence testing (TOST)

TEACUP is a small Streamlit web app for running **equivalence tests** using the **Two One-Sided Tests (TOST)** approach. It’s designed to make equivalence testing easy from a GUI: upload (or paste) your data, choose a test, set equivalence bounds, and run.

## Live app

➡️ **Open the app:** https://teacup-equivtest.streamlit.app/

---
## What this app does

For a chosen test, the app runs:

- **Equivalence test (TOST):** two one-sided tests against your equivalence bounds *(L, U)*.
- (When available) a **two-sided “difference” test** as a reference (NHST).

The online app enables you to:
- select the test type,
- enter bounds and alpha,
- provide data (upload / paste / summary stats where supported),
- view results and a plain-language interpretation.

---

## Tests available in the GUI

### T-tests (mean / mean difference)
Use these when your outcome is approximately continuous and you’re comfortable with mean-based comparison.

- **Welch t-test (two-sample)** — two independent groups, does *not* assume equal variances.
- **Pooled t-test (two-sample)** — two independent groups, assumes equal variances.
- **One-sample t-test** — one column (values or differences), or two columns interpreted as paired values and converted to differences *(col2 − col1)*.

**Equivalence bounds:** in the original data units (e.g., `[-0.5, 0.5]`).

### Nonparametric (two-sample)
- **Brunner–Munzel test** — compares two independent groups using a probability-scale “relative effect” (stochastic superiority).

**Equivalence bounds:** on the probability scale **[0, 1]**, typically centered around **0.5** (no effect), e.g. `[0.35, 0.65]`.

### Nonparametric (paired)
- **Paired Brunner–Munzel test** — paired / repeated-measures version on the same probability scale.

**Equivalence bounds:** on **[0, 1]**, centered around **0.5**.

### Proportions (2×2 tables)
- **Two-sample proportions** — equivalence testing for proportions using a Wald z-test.

Choose one effect-size parameterization:
- **Difference in proportions**
- **Odds ratio (OR)**
- **Risk ratio (RR)**

**Equivalence bounds:** depend on the effect size:
- Difference: bounds in proportion units (e.g., `[-0.10, 0.10]`)
- OR/RR: bounds as ratios (e.g., `[0.8, 1.25]`)

---

## Input formats

The app supports three input modes (depending on the selected test):

1. **Upload file** (`.csv`, `.xls`, `.xlsx`)
   (The folder 'sample files' has examples for one- and two-sample tests. They do not produce particularly interesting results, they just serve as an illustration of how the input format looks.)
2. **Paste table** (comma- or tab-separated; you can use the button "Generate sample data" underneath the textbox to generate random data that illustrate the input format. )
3. **Summary statistics (manual)** *(t-tests only)*  
   - Two-sample: mean, SD, n for each group  
   - One-sample: mean, SD, n


### Expected shapes (quick guide)

- **Two-sample tests:** 2 columns (group 1, group 2)
- **Paired Brunner–Munzel:** 2 columns of equal length (paired observations)
- **One-sample t-test:**  
  - 1 column: values (or already-computed differences), **or**  
  - 2 columns: paired values interpreted as differences *(col2 − col1)*
- **Two-sample proportions:** a **2×2** contingency table of counts (e.g., successes/failures by group)

---

## Run locally

### 1) Install dependencies

Create and activate a virtual environment, then install dependencies:

pip install streamlit numpy pandas scipy openpyxl

###  2) Download or clone the repository
git clone https://github.com/jtmff/teacup.git

Or download the repository as a ZIP file and extract it.

### 3) Start the application

Open a terminal and navigate to the directory containing app.py:

cd path/to/project

Then launch the Streamlit app using:

streamlit run app.py

Streamlit will start a local server and display a URL in the terminal (typically http://localhost:8501). Open this URL in your web browser to use the application (it often starts a browser on its own).