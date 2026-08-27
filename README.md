# 🎓 Student Economic Pressure Map

### A Responsible, White-Box Decision-Support Platform for Analyzing Socioeconomic Strain Across Student Cohorts

[![Python](https://img.shields.io/badge/Python-3.10.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-%E2%89%A50.100.0-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Deployed on Render](https://img.shields.io/badge/Backend-Render-46E3B7?logo=render&logoColor=white)](https://render.com/)
[![GitHub Pages](https://img.shields.io/badge/Frontend-GitHub%20Pages-222222?logo=githubpages&logoColor=white)](https://pages.github.com/)
[![Explainability](https://img.shields.io/badge/XAI-100%25%20White--Box-14B8A6)]()
[![Privacy](https://img.shields.io/badge/Data%20Policy-Zero%20Retention-F59E0B)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()

---

## 📑 Table of Contents

- [1. Executive Overview](#1-executive-overview)
- [2. Interactive System Architecture & High-Level Workflow](#2-interactive-system-architecture--high-level-workflow)
- [3. The Model Development Journey (Research & Methodology)](#3-the-model-development-journey-research--methodology)
  - [Phase 1: The Unsupervised Clustering Exploration & Its Abandonment](#phase-1-the-unsupervised-clustering-exploration--its-abandonment)
  - [Phase 2: Mathematical Formulation of the Economic Pressure Index (EPI)](#phase-2-mathematical-formulation-of-the-economic-pressure-index-epi)
  - [Phase 3: Statistical Predictor Analysis & Feature Reduction](#phase-3-statistical-predictor-analysis--feature-reduction)
  - [Phase 4: Multi-Model Algorithmic Tournament](#phase-4-multi-model-algorithmic-tournament)
- [4. Production Decision Logic, Leaf Registry & Explainability (XAI)](#4-production-decision-logic-leaf-registry--explainability-xai)
- [5. Backend Architecture & API Contract Specifications](#5-backend-architecture--api-contract-specifications)
- [6. Frontend Engineering & User Experience](#6-frontend-engineering--user-experience)
- [7. Cloud Deployment & Cold-Start Latency Mitigation Runbook](#7-cloud-deployment--cold-start-latency-mitigation-runbook)
- [8. Local Setup & Reproduction Guide](#8-local-setup--reproduction-guide)
- [9. Ethics, Privacy & Responsible AI Statement](#9-ethics-privacy--responsible-ai-statement)

---

## 1. Executive Overview

Socioeconomic pressure among students is a classic **unobserved latent construct** — nobody can directly measure "financial stress," so it has to be inferred indirectly from proxy behaviors and attitudes. Two engineering risks follow from that:

1. **Fabricated categories.** If you force a discretization onto a phenomenon before checking whether it actually clusters, you risk manufacturing socioeconomic "tiers" that have no empirical basis.
2. **Black-box gatekeeping.** In education and student-welfare contexts, an opaque ensemble model that silently flags a student as "high risk" — with no explanation a counselor or the student can inspect — erodes institutional trust and is difficult to challenge or audit.

**Student Economic Pressure Map** was built to avoid both traps. The project:

- **Empirically tested** whether students form natural clusters before assuming they do (they don't — see [Phase 1](#phase-1-the-unsupervised-clustering-exploration--its-abandonment)).
- Built a continuous, mathematically defined **Economic Pressure Index (EPI)** instead of an arbitrary label.
- Reduced nine candidate predictors down to four statistically justified survey questions.
- Selected a **regularized, depth-4 Decision Tree** over higher-scoring ensemble models specifically because it offers exact, traceable, leaf-level explanations — a deliberate accuracy-for-transparency trade-off.
- Shipped that model behind a **stateless FastAPI backend** and a **zero-dependency vanilla frontend**, with every prediction accompanied by a plain-language rule trace and an explicit uncertainty flag.

The result is a decision-*support* tool, not a decision-*maker* — every output is traceable back to an exact terminal node in a small, auditable tree.

### Live Deployment

| Component | Link |
|---|---|
| 🌐 Frontend (GitHub Pages) | `https://saleemshahzad08.github.io/Student-Economic-Pressure-Map/` |
| ⚙️ Backend API root | `https://student-economic-pressure-map.onrender.com` |
| 📑 Interactive API Docs (Swagger UI) | `https://student-economic-pressure-map.onrender.com/docs` |
| 💻 Source Repository | `https://github.com/saleemshahzad08/Student-Economic-Pressure-Map` |

> ⚠️ The backend runs on Render's free tier and spins down after 15 minutes of inactivity. The first request after idle time may take 30–50 seconds while the container cold-starts — see [§7](#7-cloud-deployment--cold-start-latency-mitigation-runbook) for the mitigation strategy.

---

## 2. Interactive System Architecture & High-Level Workflow

### 2.1 End-to-End Request Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                         CLIENT (GitHub Pages)                         │
│                                                                        │
│   [Page Load] ──► DOMContentLoaded ──► GET /health  (silent pre-warm) │
│                                             │                          │
│   [User fills 4-question form]             │                          │
│         │                                  ▼                          │
│         └──► form submit ──► JSON payload {                           │
│                internet_cost, device_access,                          │
│                work_reason, education_improvement }                   │
└───────────────────────────────┬────────────────────────────────────────┘
                                 │  POST /predict
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     BACKEND (FastAPI on Render)                       │
│                                                                        │
│  1. Pydantic v2 Literal validation ──► reject out-of-vocab (HTTP 422) │
│  2. DataFrame construction from validated payload                     │
│  3. ColumnTransformer.transform(X)      (preprocessing)               │
│  4. DecisionTreeClassifier.predict_proba(X)  →  [P_Low, P_Mod, P_High]│
│  5. argmax(P)  →  predicted_category                                  │
│  6. classifier.apply(X)  →  leaf_id  →  LEAF_RULES[leaf_id]           │
│  7. margin = P↓[0] − P↓[1]                                             │
│     is_uncertain = (margin ≤ 0.05) OR (P↓[0] < 0.45)                  │
│  8. Assemble PredictionResponse (Pydantic-validated JSON)              │
└───────────────────────────────┬────────────────────────────────────────┘
                                 │  200 OK JSON
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         CLIENT (renderResults)                        │
│                                                                        │
│  • Animated probability bars for Low / Moderate / High                │
│  • Colored tone badge (.tone-low / .tone-mod / .tone-high)            │
│  • Plain-language rule-trace text in a collapsible accordion          │
│  • Amber "Split-Decision Boundary Alert" if is_uncertain === true     │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 Repository Directory Tree

```
Student-Economic-Pressure-Map/
├── index.html                              # Main frontend view (survey + results)
├── style.css                               # Design system, layout, responsive rules
├── app.js                                  # Client state machine, pre-warm, API calls
├── methodology.html                        # Public-facing research methodology page
├── assets/
│   └── favicon.svg
├── backend/
│   ├── main.py                             # FastAPI app, inference engine, leaf registry
│   ├── schemas.py                          # Pydantic v2 request/response contracts
│   ├── requirements.txt                    # Pinned dependency manifest
│   ├── student_economic_pressure_pipeline.joblib   # Serialized preprocessing + model
│   └── .gitignore                          # venv / cache exclusions
└── README.md                               # You are here
```

---

## 3. The Model Development Journey (Research & Methodology)

This section documents the empirical research record — including the paths that were **tried and rejected** — because the abandonment of unsupervised clustering is itself the project's central scientific finding.

### Phase 1: The Unsupervised Clustering Exploration & Its Abandonment

**Research question:** Do students naturally partition into discrete, unobserved socioeconomic clusters, or does economic pressure vary continuously?

Because the survey data consists almost entirely of nominal and ordinal categorical variables, standard Euclidean-distance clustering (plain K-Means) is theoretically unsound without a mixed-type embedding. Three configurations were tested:

| Exploratory Framework | Clustering Technique | Peak Silhouette Score (s) |
|---|---|---|
| Gower's Distance Matrix | K-Medoids / PAM | < 0.120 |
| One-Hot Encoding Space | K-Means (Euclidean baseline) | < 0.145 |
| **FAMD Embedding** | **Agglomerative Hierarchical** | **0.185** |

The silhouette coefficient is defined per-point as:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s \in [-1, 1]$$

where $a(i)$ is the mean intra-cluster distance and $b(i)$ the mean nearest-cluster distance. Even the best configuration (FAMD → Agglomerative) peaked at $s \approx 0.185$ — far below the threshold needed to claim well-separated natural groupings.

> **Core Empirical Finding:** Student economic pressure exists along an unbroken, unimodal continuum rather than within segregated, naturally occurring social groups.

Rather than manipulating hyperparameters until a visually plausible cluster boundary emerged, clustering was deliberately abandoned in favor of building an interpretable **continuous** construct — a decision grounded in responsible-AI practice rather than in algorithmic failure.

### Phase 2: Mathematical Formulation of the Economic Pressure Index (EPI)

Six survey indicators directly measuring financial strain, each rated $X_{i,k} \in \{0,1,2,3\}$, were aggregated per respondent $i$:

| # | Indicator | Measures |
|---|---|---|
| 1 | `financial_focus` | Frequency of concern about personal/academic finances |
| 2 | `job_worry` | Frequency of career-related economic anxiety |
| 3 | `financial_choices` | Budgetary restriction on necessary academic expenditures |
| 4 | `cost_opportunity` | Academic trade-offs made to accommodate living expenses |
| 5 | `family_financial_impact` | Financial dependencies / family socioeconomic burden |
| 6 | `financial_career_pressure` | Career trajectory compelled by immediate financial need |

With a maximum raw score of 18 ($6 \times 3$), the continuous index is normalized to a 0–100 scale:

$$\text{EPI}_i = \left(\frac{\sum_{k=1}^{6} X_{i,k}}{18}\right) \times 100$$

**Descriptive statistics ($N = 514$):**

| Metric | Value | Metric | Value |
|---|---|---|---|
| Mean ($\mu$) | 60.07% | Q1 | 44.44% |
| Std. Dev. ($\sigma$) | 20.77% | Median (Q2) | 61.11% |
| Min / Max | 0.00% / 100.00% | Q3 | 72.22% |
| Skewness | −0.282 (mild, unimodal) | — | — |

**Quantile stratification** ($q = 3$) converts the continuous EPI into three analytical — not naturally occurring — cohorts:

$$
y_i =
\begin{cases}
\text{Low} \ (0), & \text{EPI}_i \le 55.56\% \quad (N=225,\ 43.77\%) \\
\text{Moderate} \ (1), & 55.56\% < \text{EPI}_i \le 72.22\% \quad (N=163,\ 31.71\%) \\
\text{High} \ (2), & \text{EPI}_i > 72.22\% \quad (N=126,\ 24.51\%)
\end{cases}
$$

### Phase 3: Statistical Predictor Analysis & Feature Reduction

Nine candidate predictors (outside the EPI formulation) were tested for association with the target tier via $\chi^2$ independence tests, Cramér's V effect size, and 5-fold cross-validated Permutation Importance:

| Predictor | $\chi^2$ | p-value | Cramér's V | Permutation Importance (ΔF1) |
|---|---|---|---|---|
| `internet_cost` | 74.81 | $4.21\times10^{-14}$ | 0.2698 | **+0.0720 ± 0.0377** |
| `device_access` | 69.42 | $5.37\times10^{-13}$ | 0.2599 | **+0.0372 ± 0.0244** |
| `education_improvement` | 56.00 | $2.91\times10^{-10}$ | 0.2334 | +0.0079 ± 0.0338 |
| `salary_importance` | 37.23 | $1.59\times10^{-6}$ | 0.1903 | +0.0169 ± 0.0209 |
| `work_reason` | 35.21 | $2.45\times10^{-5}$ | 0.1851 | +0.0195 ± 0.0190 |
| `job_security` | 32.51 | $1.30\times10^{-5}$ | 0.1778 | +0.0018 ± 0.0259 |
| `salary_vs_degree` | 28.81 | $6.61\times10^{-5}$ | 0.1674 | +0.0074 ± 0.0166 |
| `work_status` | 9.40 | 0.1521 | 0.0956 | −0.0035 ± 0.0249 |
| `career_confidence` | 4.06 | 0.6690 | 0.0628 | +0.0054 ± 0.0159 |

A 7-experiment feature-ablation tournament (Stratified 5-Fold CV, Macro F1) followed:

| Exp. | Predictor Subset | Mean Macro F1 | σ |
|---|---|---|---|
| A | All 9 raw features (baseline) | 0.5022 | ±0.0426 |
| B | Top 3 permutation (`internet`, `device`, `work_reason`) | 0.5193 | ±0.0313 |
| C | Top 5 permutation (+ `salary_importance`, `work_status`) | 0.5061 | ±0.0385 |
| D | Top 7 Cramér's V features | 0.4730 | ±0.0323 |
| E | `internet_cost`, `device_access`, `education_improvement` | 0.4949 | ±0.0292 |
| **F** | **Winning 4-feature set** (`internet`, `device`, `work_reason`, `edu_imp`) | **0.5114** | **±0.0309** |
| G | 4-feature set + `salary_importance` | 0.4894 | ±0.0312 |

**Experiment F** was selected: it isolates four orthogonal, statistically justified predictors, keeps cross-validated performance stable, and — critically — reduces the student-facing survey to just 4 questions, minimizing survey fatigue without meaningfully sacrificing signal.

### Phase 4: Multi-Model Algorithmic Tournament

Using the winning 4 features, seven algorithm families were benchmarked with `RandomizedSearchCV` (Stratified 5-fold, Macro F1, 50 iterations):

| Algorithm | Best Configuration | Best Macro F1 |
|---|---|---|
| Logistic Regression | C=4.45, L1, solver=saga, class_weight=balanced | 0.4807 |
| K-Nearest Neighbors | k=28, Manhattan distance, uniform weights | 0.4878 |
| Linear SVC | kernel=linear, C=5.20, class_weight=balanced | 0.4880 |
| Gradient Boosting | 62 estimators, lr=0.0767, depth=3, subsample=0.79 | 0.4949 |
| XGBoost | 159 estimators, depth=4, lr=0.0603, α=0.116, λ=1.60 | 0.5141 |
| Random Forest | 171 estimators, depth=7, max_features=sqrt, bootstrap=False | 0.5201 |
| **Decision Tree** | criterion=log_loss, depth=4, min_split=12, min_leaf=5 | **0.5255** |

**Why not the highest scorer?** Random Forest and XGBoost were competitive numerically, but both are effectively un-inspectable ensembles. In a student-welfare context, an opaque "high risk" label with no accompanying rationale is a trust and accountability liability. The Decision Tree matched ensemble-level accuracy while providing exact, per-prediction path tracing — an intentional accuracy-for-transparency trade-off.

A focused grid search then regularized the final production tree to guarantee statistically meaningful leaves:

- **Splitting criterion:** `entropy`
- **Maximum depth:** 4
- **Minimum samples per leaf:** 25 (guarantees statistical power in every terminal node)
- **Final cross-validated Macro F1:** **0.5361 ± 0.0354**

**Held-out classification performance by tier:**

| Target Tier | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Low Economic Pressure | 0.61 | 0.72 | 0.66 | 225 |
| Moderate Economic Pressure | 0.47 | 0.42 | 0.45 | 163 |
| High Economic Pressure | 0.50 | 0.40 | 0.45 | 126 |
| **Macro Average** | **0.53** | **0.52** | **0.52** | N = 514 |
| Weighted Average | 0.54 | 0.55 | 0.54 | — |

---

## 4. Production Decision Logic, Leaf Registry & Explainability (XAI)

### 4.1 100% White-Box Explainability

Because the production model is a shallow, regularized `DecisionTreeClassifier`, every prediction can be traced to an **exact** terminal node — no approximation, no post-hoc surrogate explainer (SHAP/LIME) needed:

```python
input_df_transformed = preprocessor.transform(input_df)
leaf_id = int(classifier.apply(input_df_transformed)[0])
rule_trace = LEAF_RULES.get(leaf_id, fallback_message)
```

`classifier.apply()` returns the index of the exact leaf a given sample fell into, which is used as a direct key into a hand-verified, human-readable rule registry.

### 4.2 The Verified 12-Leaf Node Registry

The regularized tree partitions the 4-dimensional input space into exactly 12 valid terminal leaves:

| Leaf ID | Dominant Category | Confidence | Plain-Language Rule Trace |
|---|---|---|---|
| 4 | Low | 86.0% | Low internet cost burden (≤ A little) with minimal device access strain (≤ Moderately). |
| 5 | Low | 78.0% | Low internet cost burden (≤ A little) with emerging device access strain (≥ Moderately). |
| 7 | Low | 69.0% | Moderate internet cost burden with low device access strain (≤ A little). |
| 8 | Low | 52.0% | Moderate internet cost burden with moderate-to-high device access strain (> A little). |
| 10 | Low | 52.0% | Low-to-moderate internet cost burden with low device strain, prioritizing institutional financial stability. |
| 11 | Moderate | 50.0% | Low-to-moderate internet cost burden with elevated device strain, prioritizing institutional financial stability. |
| 15 | Moderate | 48.0% | High internet cost burden (≥ Moderately) with low-to-moderate device strain (≤ Moderately). |
| 16 | Moderate | 45.0% | High internet cost burden with elevated device strain, primary work motivation = personal expenses. |
| 17 | High (uncertain) | 35.0% | High internet cost burden with severe device strain (A lot), no institutional financial priority — high contextual uncertainty (Low 33% / Mod 33% / High 35%). |
| 20 | Moderate | 45.0% | High internet cost burden (A lot), moderate device strain, non-working cohort with financial stability priority. |
| 21 | Moderate | 52.0% | High internet cost burden (A lot), moderate device strain, working cohort with financial stability priority. |
| 22 | High | 66.1% | High internet cost burden (A lot) combined with compound device strain (A lot) and financial stability needs. |

### 4.3 Dual-Condition Decision Uncertainty Engine

Given the sorted (descending) probability vector $P^{\downarrow} = [p_0^{\downarrow}, p_1^{\downarrow}, p_2^{\downarrow}]$:

$$\text{margin} = p_0^{\downarrow} - p_1^{\downarrow}$$

$$\text{is\_uncertain} = (\text{margin} \le 0.05) \lor \ (p\_0^{\downarrow} < 0.45)$$

A prediction is flagged as a **Split-Decision Boundary** whenever the top two class probabilities are within a 5-point margin of each other, *or* the leading probability itself fails to clear 45% — i.e. even an uncontested "winner" that's still weak gets flagged. Both the displayed percentages and the boolean flag are derived from the same rounded values in `main.py`, so the UI can never show numbers that visually contradict the uncertainty badge.

---

## 5. Backend Architecture & API Contract Specifications

### 5.1 Stateless Privacy Governance

The backend performs **pure in-memory computation**: no database, no disk writes of request data, no session storage, no IP logging, and no per-user tracking. Every submitted vector exists only for the duration of the request and is garbage-collected immediately after the response is sent.

### 5.2 Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Lightweight liveness probe; also used for client-side and external pre-warming pings |
| `POST` | `/predict` | Primary inference route — accepts a `StudentProfileRequest`, returns a `PredictionResponse` |
| `GET` | `/` | Basic welcome/test route |

### 5.3 Data Schemas & Type Contracts

Strict Pydantic v2 `Literal` typing intercepts any out-of-vocabulary category **before** it ever reaches the ML pipeline, returning an HTTP 422 rather than allowing silent coercion or a pipeline crash.

```python
class StudentProfileRequest(BaseModel):
    internet_cost: Literal["Not at all", "A little", "Moderately", "A lot"]
    device_access: Literal["Not at all", "A little", "Moderately", "A lot"]
    work_reason: Literal[
        "Education Cost", "Family Support", "Future savings / Other",
        "Personal expenses", "Not Applicable"
    ]
    education_improvement: Literal[
        "Better career guidance", "Better learning resources",
        "Greater financial stability", "More flexible study arrangements"
    ]

class ProbabilityDistribution(BaseModel):
    Low: float = Field(..., ge=0.0, le=1.0)
    Moderate: float = Field(..., ge=0.0, le=1.0)
    High: float = Field(..., ge=0.0, le=1.0)

class PredictionResponse(BaseModel):
    predicted_category: Literal["Low", "Moderate", "High"]
    probabilities: ProbabilityDistribution
    rule_trace: str
    is_uncertain: bool
```

### 5.4 Sample Payloads

<details>
<summary><strong>✅ Standard high-certainty prediction</strong></summary>

**Request** — `POST /predict`
```json
{
  "internet_cost": "Not at all",
  "device_access": "A little",
  "work_reason": "Not Applicable",
  "education_improvement": "Better learning resources"
}
```

**Response** — `200 OK`
```json
{
  "predicted_category": "Low",
  "probabilities": {
    "Low": 0.86,
    "Moderate": 0.10,
    "High": 0.04
  },
  "rule_trace": "Low internet cost burden (<= A little) with minimal device access strain (<= Moderately). Associated with Low Economic Pressure cohort (86.0%).",
  "is_uncertain": false
}
```
</details>

<details>
<summary><strong>⚠️ Borderline / high-uncertainty prediction</strong></summary>

**Request** — `POST /predict`
```json
{
  "internet_cost": "A lot",
  "device_access": "A lot",
  "work_reason": "Personal expenses",
  "education_improvement": "Better career guidance"
}
```

**Response** — `200 OK`
```json
{
  "predicted_category": "High",
  "probabilities": {
    "Low": 0.33,
    "Moderate": 0.33,
    "High": 0.34
  },
  "rule_trace": "High internet cost burden (>= Moderately) with severe device access strain (A lot), with no prioritization of institutional financial stability. High Contextual Uncertainty cohort across Low (33%), Moderate (33%), and High (35%).",
  "is_uncertain": true
}
```
</details>

<details>
<summary><strong>❌ Validation error (out-of-vocabulary input)</strong></summary>

**Request** — `POST /predict`
```json
{
  "internet_cost": "Extremely",
  "device_access": "A little",
  "work_reason": "Not Applicable",
  "education_improvement": "Better learning resources"
}
```

**Response** — `422 Unprocessable Entity`
```json
{
  "detail": [
    {
      "type": "literal_error",
      "loc": ["body", "internet_cost"],
      "msg": "Input should be 'Not at all', 'A little', 'Moderately' or 'A lot'",
      "input": "Extremely"
    }
  ]
}
```
</details>

### 5.5 CORS & Middleware

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "https://saleemshahzad08.github.io"
    ],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"]
)
```

Any pipeline execution failure is caught and returned as a generic HTTP 500 JSON payload — the exception message is included for developer visibility, but no filesystem paths or internal stack traces are leaked to the client.

---

## 6. Frontend Engineering & User Experience

### 6.1 Architecture

A deliberately **zero-dependency vanilla stack** — HTML5, CSS3, ES6+ JavaScript, no bundler, no framework:

- `index.html` — the interactive survey tool and results display
- `style.css` — a unified dark-slate design system with semantic tier colors
- `app.js` — the client-side state machine and API layer
- `methodology.html` — a companion documentation page presenting the research foundations directly to educators and students

This choice eliminates supply-chain risk, gives near-instant load times, and keeps long-term maintenance trivial for a small research prototype.

### 6.2 Design System

| Role | Color |
|---|---|
| Base surfaces | `#0F172A`, `#1E293B` (dark slate) |
| Low Pressure accent | `#14B8A6` (teal) |
| Moderate Pressure accent | `#F59E0B` (amber) |
| High Pressure accent | `#EF4444` (crimson) |

### 6.3 Key UX Features

- **Governance banner** — a non-dismissible top-level notice clarifying the tool is derived from a 514-student survey sample and is explicitly non-diagnostic.
- **Responsive 2-column survey form**, collapsing to a single column at ≤ 768px, mirroring the backend's exact `Literal` vocab so no client-side value can ever fail validation.
- **Cold-start-aware loading state** — a CSS ring spinner, with a secondary "connecting to secure backend" notice that only appears after 2.5 seconds, so a normal (warm) response never shows unnecessary alarm text.
- **Animated probability bars** for Low / Moderate / High with cubic-bezier width transitions.
- **Tone-coded prediction badge** (`.tone-low` / `.tone-mod` / `.tone-high`).
- **Collapsible decision-logic accordion** (`<details>`) surfacing the raw leaf-trace explanation string.
- **Split-Decision Boundary Alert** — an amber warning block rendered only when `is_uncertain === true`, explaining the margin/threshold logic in plain language.
- **Dedicated `methodology.html`** page — the clustering-abandonment story, the EPI formula, feature-selection statistics, and the leaf registry, written for a non-technical academic audience.

### 6.4 Client-Side State Machine (`app.js`)

| Layer | Responsibility |
|---|---|
| DOM Cache Registry | Queries all interactive elements once at load, avoiding repeated tree traversals |
| `preWarmBackend()` | Fires a silent `GET /health` on `DOMContentLoaded` to start Render's cold boot while the user is still reading the form |
| `performServerTask()` | Wraps `fetch()` with an `AbortController` (60s timeout), defensive `response.ok` / JSON error parsing, and try/catch/finally cleanup |
| `renderResults()` | Converts float probabilities to rounded percentages, updates bar widths and labels, toggles the uncertainty badge, injects the rule trace, and smooth-scrolls into view |
| `toggleLoadingState()` | Centralized button/spinner/timer lifecycle management, including the 2500ms cold-start notice timer |

### 6.5 Cross-System Contract Alignment

| Interaction | Frontend Node | Backend Field | Type |
|---|---|---|---|
| Pre-warm | `DOMContentLoaded` | `GET /health` | HTTP GET |
| Request | `#internet-cost` | `internet_cost` | categorical `str` |
| Request | `#device-strain` | `device_access` | categorical `str` |
| Request | `#work-motivation` | `work_reason` | categorical `str` |
| Request | `#institutional-improvement` | `education_improvement` | categorical `str` |
| Response | `#predicted-category` | `predicted_category` | `"Low"｜"Moderate"｜"High"` |
| Response | `#prob-*-val` / fills | `probabilities` | `dict[str, float]` |
| Response | `#rule-trace-text` | `rule_trace` | `str` |
| Response | `#uncertainty-badge` | `is_uncertain` | `bool` |

---

## 7. Cloud Deployment & Cold-Start Latency Mitigation Runbook

### 7.1 Decoupled Production Architecture

**Frontend — GitHub Pages**
- Enabled under repository settings, serving the root (`/`) of the `main` branch, SSL enforced.

**Backend — Render Web Service (deployed via Dashboard UI)**

| Setting | Production Value |
|---|---|
| Service Type | Web Service |
| Repository | `saleemshahzad08/Student-Economic-Pressure-Map` |
| Branch | `main` |
| Root Directory | `backend` |
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Instance Plan | Free Tier |
| Environment Variable | `PYTHON_VERSION=3.10.12` |

### 7.2 CORS Whitelisting

The production frontend origin `https://saleemshahzad08.github.io` is explicitly whitelisted in `CORSMiddleware`, alongside local development ports (`localhost:5500`, `127.0.0.1:5500`) for Live-Server-based testing.

### 7.3 Version Harmonization

Runtime dependencies are pinned to match the exact model-training environment, preventing `InconsistentVersionWarning` on unpickling and eliminating silent runtime drift:

```
fastapi>=0.100.0,<1.0.0
uvicorn[standard]>=0.22.0,<1.0.0
scikit-learn>=1.6.1,<1.7.0
pandas>=2.0.0,<3.0.0
joblib>=1.3.0,<2.0.0
pydantic>=2.0.0,<3.0.0
numpy>=1.24.0,<3.0.0
```

### 7.4 Two-Tier Cold-Start Elimination Strategy

Render's free tier spins a web service down after 15 minutes of inactivity, producing a 30–50 second cold-start delay on the next request. Two complementary layers address this:

1. **Client-side pre-warming** — an asynchronous background `GET /health` request fires the instant the page's `DOMContentLoaded` event fires, so the container begins waking up while the user is still reading the four survey questions, well before they click submit.
2. **External scheduled keep-alive** — an automated uptime scheduler (cron-job.org / UptimeRobot) pings `GET /health` every 10 minutes around the clock, keeping the container warm and memory-resident even between user sessions.

---

## 8. Local Setup & Reproduction Guide

```bash
# 1. Clone the repository
git clone https://github.com/saleemshahzad08/Student-Economic-Pressure-Map.git
cd Student-Economic-Pressure-Map

# 2. Set up the backend
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# 3. Run the API locally with hot-reload
uvicorn main:app --reload --port 8000
# → API docs available at http://127.0.0.1:8000/docs

# 4. Serve the frontend
cd ..
# Open index.html with VS Code "Live Server" on port 5500
# (already whitelisted in CORS: http://localhost:5500 / http://127.0.0.1:5500)
```

> If testing locally, update `API_BASE_URL` in `app.js` to `http://localhost:8000` (or `127.0.0.1:8000`) instead of the production Render URL.

---

## 9. Ethics, Privacy & Responsible AI Statement

- **Non-punitive, non-diagnostic by design.** This tool produces exploratory, aggregate-pattern estimates derived from a 514-respondent survey sample. It is **not** a clinical, psychological, or individualized financial diagnosis, and it must not be used as the sole basis for financial-aid decisions, academic probation, or any other high-stakes determination about a specific student.
- **Zero data retention.** No submitted survey response is logged, stored on disk, cached, or transmitted to a third-party database. Computation is purely in-memory and request-scoped.
- **Consent & framing.** The tool is intended for institutional decision-support and pattern discovery — for example, helping a student-affairs office understand aggregate cohort trends — not for surveilling or labeling individual students without their knowledge and consent.
- **Full-margin transparency.** Every prediction ships with its raw probability distribution, its exact decision-tree rule trace, and an explicit uncertainty flag, so a human reviewer can always see *why* the model produced a given estimate and how confident it actually was.
- **Recommended use.** As one input among several in a broader, human-reviewed support process — never as an automated gatekeeping mechanism.

---

<p align="center">
Built as an open-source responsible-AI prototype for exploratory pattern discovery and institutional decision support.
</p>
