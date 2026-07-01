# Decision Memory Engine (DME) — Clinical Advisory System

> An AI-powered clinical decision-support prototype that combines a trained ML risk model with
> case-based reasoning and an LLM advisory layer — presented as an interactive Streamlit dashboard.

![Type](https://img.shields.io/badge/type-research%20prototype-blueviolet)
![Language](https://img.shields.io/badge/python-3.10%2B-3776ab)
![UI](https://img.shields.io/badge/ui-Streamlit-ff4b4b)
![ML](https://img.shields.io/badge/ML-scikit--learn%20RandomForest-f7931e)
![LLM](https://img.shields.io/badge/LLM-Groq%20Llama%203.3%2070B-000000)

---

## Overview

The **Decision Memory Engine** is a clinical advisory system that helps reason about patient
risk and outcomes. It is built around the idea of a *decision memory* — every decision the
system makes is logged, later linked to an outcome, and fed back to improve future predictions
(a lightweight case-based-reasoning + continual-learning loop).

The system fuses three complementary techniques:
1. **Statistical ML** — a RandomForest classifier trained on 100 MIMIC-IV demo patients for
   outcome prediction and risk scoring.
2. **Case-Based Reasoning (CBR)** — retrieval of historically similar cases from the decision
   memory to contextualize new ones.
3. **LLM advisory** — Groq-hosted Llama 3.3 70B for natural-language summaries, differential
   suggestions, missing-data detection, safety/bias analysis, and an interactive assistant.

> This project is the engineering artefact behind the CST3391 final-year project
> ([`cst3391-individual-project`](https://github.com/Alakazam-boop/cst3391-individual-project))
> and the CST3350 poster.

## Features — the five-tab dashboard
| Tab | Capability |
|-----|-----------|
| 1. **Clinical Analysis** | Enter vitals → risk score + ML outcome prediction |
| 2. **Deep Analysis** | LLM-generated summary, differentials, and missing-data flags |
| 3. **Decision Memory** | Charts of historical cases + model version history |
| 4. **Safety & Bias** | Automated data-quality checks + AI safety/bias analysis |
| 5. **AI Assistant** | Free-form chat with full patient context |

## Architecture

```
        ┌───────────────────────── Streamlit UI (app.py) ─────────────────────────┐
        │  Tab1 Clinical · Tab2 Deep · Tab3 Memory · Tab4 Safety · Tab5 Assistant  │
        └───────┬───────────────────────┬───────────────────────────┬─────────────┘
                │                        │                           │
      ┌─────────▼─────────┐   ┌──────────▼──────────┐     ┌──────────▼──────────┐
      │ learning_engine   │   │ outcome_linker      │     │  Groq LLM API       │
      │ RandomForest +    │   │ links decisions →   │     │  Llama 3.3 70B      │
      │ CBR index         │   │ outcomes            │     │  (via env API key)  │
      └─────────┬─────────┘   └──────────┬──────────┘     └─────────────────────┘
                │                         │
      ┌─────────▼─────────────────────────▼─────────┐
      │  decision_logger  →  SQLite (database/dme.db) │
      └───────────────────────────────────────────────┘
```

## Technology Stack
| Component | Technology |
|-----------|-----------|
| UI | Streamlit |
| ML | scikit-learn (RandomForest), NumPy, pandas |
| Reasoning | Custom case-based-reasoning index over the decision log |
| LLM | Groq API (Llama 3.3 70B) — **key supplied via environment variable, never committed** |
| Storage | SQLite (`database/dme.db`) |
| Plotting | Matplotlib |
| Data | MIMIC-IV demo cohort (100 patients) |

## Repository Structure
```
.
├── app.py                    # Streamlit dashboard (single-file, 5 tabs)
├── reset.py                  # Reset the decision-memory database
├── modules/
│   ├── learning_engine.py    # train_classifier · predict_outcome · build_cbr_index
│   ├── outcome_linker.py     # link completed decision traces to outcomes
│   ├── decision_logger.py    # persist every decision to SQLite
│   ├── feature_pipeline.py   # feature engineering
│   └── setup_db.py           # schema bootstrap
├── evaluation/
│   ├── evaluate.py           # model evaluation
│   └── results/              # confusion matrix, cross-val & model comparison plots
├── models/                   # serialized model artefacts (.pkl)
└── data/                     # raw / processed / synthetic datasets
```

## Getting Started
```bash
# 1. Clone
git clone https://github.com/Alakazam-boop/dme-project.git
cd dme-project

# 2. Install dependencies
pip install streamlit scikit-learn pandas numpy matplotlib groq

# 3. Provide the LLM API key via environment (NEVER hard-code it)
export GROQ_API_KEY="your-groq-key"      # Windows: set GROQ_API_KEY=...

# 4. Run
streamlit run app.py
```
The AI features degrade gracefully if `GROQ_API_KEY` is unset — the ML prediction and
decision-memory features still work; only the LLM tabs are disabled.

## Model Evaluation
`evaluation/evaluate.py` produces a confusion matrix, cross-validation comparison, and a
model-comparison chart (see `evaluation/results/`).

## Security & Data Notes
- 🔑 The Groq API key is read from `GROQ_API_KEY` at runtime — no secrets are stored in the repo.
- 🏥 Uses the **public MIMIC-IV demo** subset only; no real/identifiable patient data.
- ⚠️ Research prototype — **not** a certified medical device and not for clinical use.

---
_Final-year research project. © Simaak Sayed._
