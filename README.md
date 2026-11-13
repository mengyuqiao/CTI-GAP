# Uncovering Vulnerabilities of LLM-Assisted Cyber Threat Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation for the paper **"Uncovering Vulnerabilities of LLM-Assisted Cyber Threat Intelligence"** 

> Anonymous repository. Will be migrated to a permanent GitHub repo with DOI via Zenodo upon acceptance.

---

## Overview

This repository provides full reproduction code for our large-scale empirical study of LLM failure modes in Cyber Threat Intelligence (CTI) reasoning. We identify **three domain-specific vulnerability categories**:

| Category | Description |
|---|---|
| **Spurious Correlation** | LLMs over-attribute based on co-occurrence rather than causal evidence (5 subtypes) |
| **Contradictory Knowledge** | Conflicting CTI sources destabilize model reasoning (6 subtypes) |
| **Constrained Generalization** | LLMs fail on emerging/zero-day threats and novel distributions (4 subtypes) |
---

## Repository Structure

```text
.
├── main.py                         # CLI and stage orchestration
├── config.py                       # Configuration, provenance and readiness checks
├── config.yaml                     # Configuration template
├── configs/
│   ├── reproduction.yaml           # Alternative study configuration template
│   └── subtype_policies.yaml       # Subtype-specific prompt policies
├── dataset_loader.py               # Manifest-based acquisition and task conversion
├── identifier_normalization.py     # Task-specific canonical identifier visibility
├── model_client.py                 # Configured LiteLLM inference and output validation
├── failure_identification.py       # RQ1: hard rules and human-calibrated strata
├── taxonomy.py                     # RQ2: human-seeded discovery and refinement
├── model_initialization.py         # Model-seeded initialization ablation
├── categorization.py               # RQ3: two-round classification and human review
├── annotation_app.py               # Annotation service and Streamlit interface
├── experiments.py                  # Baselines, ablations, stability and defenses
├── research_backends.py            # Retrieval, reranking, verification and adapters
├── generative_probabilities.py     # Complete bounded autoregressive distributions
├── generation_context.py           # Immutable generation inputs for later review
├── sampling.py                     # Matched sampling through condition projections
├── evaluation.py                   # Task, agreement, audit and stability metrics
├── schemas.py                      # Validated scientific records and contracts
├── storage.py                      # SQLite records and content-addressed artifacts
├── prompts/
│   ├── task_evaluation.jinja2       # CTI task evaluation
│   ├── taxonomy_initialization.jinja2
│   ├── taxonomy_classification.jinja2
│   ├── agent_round_one.jinja2       # Independent categorization
│   ├── agent_round_two.jinja2       # Peer-informed reconsideration
│   ├── agent_single_reconsideration.jinja2
│   └── defense_revision.jinja2      # Evidence-constrained answer revision
├── tests/                          # Offline numerical, regression and integration tests
├── requirements.txt                # Base runtime and test dependencies
├── LICENSE
├── .gitignore
└── README.md
```

---

## Installation

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

### API Keys

Set the following environment variables for the models you plan to use:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
export XAI_API_KEY="your-xai-key"           # For Grok
export NVD_API_KEY="your-nvd-key"           # Optional, increases rate limit
export VT_API_KEY="your-virustotal-key"     # For VirusTotal enrichment
export OTX_API_KEY="your-otx-key"           # For AlienVault OTX
```

---

## Quick Start

### 1. Prepare a Research Configuration

```bash
cp config.yaml study.yaml
.venv/bin/python main.py --help
```
### 2. Execute the Pipeline

```bash
.venv/bin/python main.py --config study.yaml prepare
.venv/bin/python main.py --config study.yaml evaluate
.venv/bin/python main.py --config study.yaml identify
.venv/bin/python main.py --config study.yaml discover-taxonomy
.venv/bin/python main.py --config study.yaml categorize
.venv/bin/python main.py --config study.yaml report

.venv/bin/python main.py --config study.yaml resume discover-taxonomy
.venv/bin/python main.py --config study.yaml resume categorize
```

### Human Annotation Interface

```bash
.venv/bin/python -m streamlit run annotation_app.py
```

The web UI requires an OIDC-capable Streamlit installation and deployment authentication. Before launching, configure these environment variables:

| Variable | Value |
|---|---|
| `CTI_ANNOTATION_DATABASE` | The experiment's SQLite database path |
| `CTI_ANNOTATION_ARTIFACT_ROOT` | The corresponding artifact directory |
| `CTI_ANNOTATION_IDENTITY_MAP` | A JSON object mapping authenticated subject IDs to assigned annotator IDs |


---

## Evaluated Models

### General-Purpose LLMs

| Abbreviation | Model |
|---|---|
| G5 | GPT-5 |
| Go4 | GPT-o4 mini |
| CLD | Claude Sonnet 4 |
| GEM | Gemini 2.5 |
| LL70 | Llama-3.1-70B-Instruct |
| MIX | Mixtral-8x7B-Instruct-v0.1 |
| QWN | Qwen2.5-14B-Instruct |
| GRK | Grok-2 |

### Cybersecurity-Specialized Models

| Abbreviation | Model |
|---|---|
| FSC | Foundation-Sec-8B |
| CB0 | Cyber-Zero |
| ZYS | ZySec-AI SecurityLLM |
| LLY | Lily-Cybersecurity-7B-v0.2 |
| CBS | CyberBase-13B |
| SPT | SecGPT (Clouditera) |
| DHT | DeepHat-V1-7B |

---

## CTI Stages & Tasks

| Stage | Tasks | Primary Metrics |
|---|---|---|
| ❶ Contextualization | Affected Systems, Attack Infrastructure, Vulnerability Linking, Malware Family Mapping, IOC Normalization, Threat Report Alignment, Event Timeline, Graph Population, Source Reliability | F1, Acc, BLEU, AUC |
| ❷ Attribution | Threat Actor Linking, TTP Extraction, Campaign Attribution, Infrastructure Reuse, Language/Style Profiling, False Flag Detection, Evidence Weighting, Relation Graph | F1, Acc, BLEU |
| ❸ Prediction | Exploit Likelihood, Impact Forecast, Target Sector Prediction, Campaign Escalation | AUC, BLEU, Acc |
| ❹ Mitigation | Patch Recommendation, YARA Rule Generation, Response Summarization, Mitigation-TTP Mapping, Defensive Playbook, Countermeasure Ranking, Incident Ticket | F1, BLEU, Acc, NDCG |

---

## Vulnerability Taxonomy

```
Spurious Correlation (SC)
├── 1.1 Co-mention bias from raw threat incidents
├── 1.2 Exploitation bias from deliberately reused IoCs
├── 1.3 Confounding factors (explicit/implicit entity correlation)
├── 1.4 Skewed source representation
└── 1.5 Hierarchical metadata from attack chains

Contradictory Knowledge (CK)
├── 2.1 Temporal contradiction (outdated vs. recent evidence)
├── 2.2 Conflicting reports of attack contexts/dependencies
├── 2.3 Semantic conflict (naming/taxonomy divergence)
├── 2.4 Divergent data structures across platforms
├── 2.5 Misaligned knowledge and security standards
└── 2.6 Counteracting CTI generation and LLM alignment

Constrained Generalization (CG)
├── 3.1 Distributional bias
├── 3.2 Unseen patterns from emerging threats
├── 3.3 Overfitted reasoning
└── 3.4 Environmental unawareness
```


---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
