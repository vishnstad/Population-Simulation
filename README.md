# Hierarchical Multi-Agent Population Simulation (PopSim)
### Calibrated Distribution-Valued Cluster Agents for Held-Out Survey Prediction

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 174 Passed](https://img.shields.io/badge/tests-174%20passed-brightgreen.svg)]()
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Final Year B.Tech Project — School of Artificial Intelligence, Amrita Vishwa Vidyapeetham**  
**Team No.:** B17  
**Team Members:**  
- **Vishal S** (CB.SC.U4AIE23160) — *Data Ingestion & Clustering Lead*
- **Nitin Krishna V** (CB.SC.U4AIE23156) — *Agent Design Lead*
- **Gaurav Mahesh** (CB.SC.U4AIE23176) — *Simulation & Integration Lead*
- **Manogna Challa** (CB.SC.U4AIE23175) — *Analytics & Visualization Lead*

**Faculty Mentors:** Dr. Snigdhatanu Acharya (Mentor) | Dr. Premjith B. (Co-Mentor)

---

## 1. Project Overview

Simulating human populations using Large Language Models traditionally prompts one LLM agent per individual persona. This suffers from two fundamental flaws:
1. **Computational Explosion:** Simulating thousands of agents is cost-prohibitive.
2. **Variance Collapse (F1/F2 Failure):** Prompted personas collapse within-group response variance and regress toward national averages.

**PopSim** resolves this by partitioning the population into $K = 56$ demographic clusters (`age_band × degree × sex`) and prompting the LLM once per cluster with a **stat card** containing demographic marginals and empirical answers to a set of *anchor* survey questions. The raw elicited distributions are calibrated using an anchor-fitted **Level/Deviation Decomposition**, predicting full response distributions for unasked questions.

### Mathematical Formulation
$$\text{pred\_cdf}(c) = \underbrace{\text{level\_cdf}}_{\text{national topline}} + s \cdot \underbrace{P_r}_{\text{subspace projection}} \Big(\underbrace{\text{raw\_cdf}(c) - \sum_c w_c \cdot \text{raw\_cdf}(c)}_{\text{subgroup deviation}}\Big)$$

- **$\text{level\_cdf}$:** Baseline item level (observed topline or population LLM call).
- **$P_r$:** Projection onto top principal directions of anchor variation (filters out isotropic noise).
- **$s$:** Scaling factor fitted strictly on out-of-fold anchors ($F = 3$ cross-fitting).

---

## 2. Review 1 Headline Results

Tested on **pooled GSS 2010–2022** ($n = 18{,}772$ respondents) across 39 held-out target survey items:

| Metric / Subset | Achieved $W_1$ | Baseline $B0a$ | Noise Floor | Performance |
|---|---|---|---|---|
| **All 39 Held-Out Targets** | **0.0602** | 0.0807 | 0.0273 | **PASS ($-25.3\%$ error reduction, pre-registered $\le -20\%$)** |
| **Top-Quartile Heterogeneity** | **0.0787** | 0.1139 | 0.0402 | **PASS ($-30.9\%$ error reduction)** |
| **Leakage-Resistant Subset** | **0.0565** | 0.0757 | 0.0324 | **PASS ($-25.4\%$ error reduction)** |
| **Uncalibrated Model ($B4$)** | 0.1926 | 0.0807 | 0.0273 | **Calibration cuts error by $69\%$** |

- **True Population Variance:** Recovered variance ratio = **$1.009$** (target band $[0.8, 1.2]$ $\implies$ Objective 2 passed).
- **90% Interval Coverage:** **$0.899$** (matches nominal $0.90$).
- **Subgroup Ordering Correlation:** Spearman $\rho = \mathbf{+0.879}$ across tolerance batteries.
- **Model Scaling:** Monotonic improvement: 3B ($-15.0\%$) $\to$ 8B ($-21.4\%$) $\to$ 14B ($-25.3\%$).
- **Zero Dollar Compute:** Executed entirely on free-tier APIs via custom SQLite prompt caching and provider failover.

---

## 3. Repository Architecture

```
Population-Simulation/
├── popsim/                      # Core package (CLI, data adapters, calibration, evaluation)
│   ├── agents/                  # Stat-card builder & prompt schemas
│   ├── aggregate/               # ACS 2024 PUMS raking (IPF) & bootstrap CIs
│   ├── calibration/             # Level/structure decomposition & subspace projection
│   ├── clustering/              # Supervised demographic partitioning (K = 56)
│   ├── data/                    # GSS & GFS survey microdata adapters
│   ├── evalx/                   # Wasserstein-1, ECE, & split-half noise gates
│   ├── llm/                     # Multi-provider router (Mistral/Groq/Gemini), quota guards, SQLite cache
│   └── cli.py                   # Unified CLI entrypoint (`popsim`)
├── tests/                       # Comprehensive test suite (174 passing tests)
├── configs/                     # Partition & run configurations (gss_main.yaml)
├── codebooks/                   # Harmonized item metadata & frozen splits
├── app/                         # Interactive Streamlit dashboard with mandatory Honesty Box
├── scenarios/                   # Policy simulation specifications
├── scripts/                     # Automated gate verification & diagnostic scripts
├── pipeline_modules/            # Standalone modular pipeline scripts (M1–M10)
│   ├── M3_statcards/            # Fold isolation & stat-card generator
│   ├── M4_elicitation/          # Distributional elicitation runners
│   ├── M5_calibration/          # Calibration algorithms
│   ├── M6_aggregation/          # Census raking
│   ├── M7_router/               # Oracle router (TF-IDF + cosine)
│   ├── M9_evaluation/           # Benchmark metrics
│   ├── M10_app/                 # Streamlit UI
│   ├── benchmark.py             # Pipeline benchmarking script
│   └── simulate_region.py       # Regional simulation CLI
├── preprocessing/               # Raw survey data ingestion & cluster tree generation
├── presentations/               # Department presentation decks
│   ├── First_Review_Presentation.tex   # LaTeX Beamer deck (15 slides)
│   └── First_Review_Presentation.pptx  # PowerPoint deck (15 slides)
├── docs/                        # Complete technical specifications & logs
│   ├── B17_build_checklist.md   # Step-by-step build & gate specifications
│   ├── B17_architecture_spec.md # System architectural specification
│   ├── PREREGISTRATION.md       # Pre-registered pass marks & freeze logs
│   └── RESULTS.md               # Draft research paper manuscript
├── pyproject.toml               # Package configuration
└── requirements.txt             # Python dependencies
```

---

## 4. Getting Started

### Installation
```bash
git clone https://github.com/vishnstad/Population-Simulation.git
cd Population-Simulation

# Create and activate environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### Running the Test Suite (174 Tests)
```bash
pytest tests/ -v
```

### Running the Pipeline & Verification Gates
```bash
# Run automated verification gates (Layers 0 to 5)
bash scripts/gates.sh

# Or run via the CLI
popsim run --config configs/gss_main.yaml

# Generate benchmark HTML report
popsim report
```

### Launching the Interactive Web Dashboard
```bash
streamlit run app/app.py
```
*The dashboard includes the Oracle Router (serving empirical survey cross-tabs for known questions), calibrated cluster predictions for unasked scenarios, and the **Mandatory Honesty Box** showing nearest benchmarked item distance.*

---

## 5. Team Responsibilities

- **Vishal S:** Data Ingestion & Harmonization (M1), Cluster Tree Partitioning (M2), Layer 0 Published Topline Verification.
- **Nitin Krishna V:** Stat Card Generator (M3), Distributional Elicitation (M4), API Router & SQLite Caching.
- **Gaurav Mahesh:** Level/Deviation Calibration Layer (M5), ACS 2024 Census Raking (M6), Layer 2 Permutation Testing.
- **Manogna Challa:** Evaluation Harness (M9), Streamlit App & Honesty Box (M10), Layer 5 Memorization Probing.

---

## 6. Citation & References

```bibtex
@article{popsim2026,
  title={Hierarchical Multi-Agent Population Simulation: Calibrated Distribution-Valued Cluster Agents},
  author={Vishal S and Nitin Krishna V and Gaurav Mahesh and Manogna Challa},
  journal={Amrita Vishwa Vidyapeetham B.Tech Thesis},
  year={2026}
}
```

Key academic foundations: Argyle et al. (*Political Analysis*, 2023), Bisbee et al. (*Political Analysis*, 2024), Santurkar et al. (*ICML OpinionQA*, 2023), Park et al. (*ACM UIST*, 2023), Li et al. (*ACL*, 2024), Pi et al. (*arXiv*, 2025).

---

## 7. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
