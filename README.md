# TrustGeoAgent

A LangGraph-based geospatial modeling prototype that compares spatial interpolation models, evaluates prediction intervals, and produces readable reports with an audit trail.

## Overview

Spatial predictions are easier to assess when model performance, uncertainty, and review decisions are visible together. TrustGeoAgent brings these steps into one workflow: compare candidate models, evaluate the selected model, inspect residual spatial patterns, and record a review decision.

The project includes a Streamlit dashboard and a command-line interface. It supports a reproducible synthetic spatial benchmark and CSV-based observations, including a bundled air-quality dataset. Optional LLM calls generate model-selection explanations and evaluation reports; deterministic reports are available without an API key.

## Features

- Cross-validation of three spatial interpolation candidates: inverse distance weighting (IDW), locally weighted regression (GWR-lite), and Gaussian process regression with an RBF kernel.
- Model selection by cross-validated root mean squared error (RMSE).
- Split-conformal prediction intervals with empirical coverage and mean interval-width reporting.
- Moran's I diagnostics for spatial patterns in prediction residuals.
- A bounded feedback loop that can adjust interval scaling once following automated review.
- Optional interactive terminal review, with decisions appended to a JSON Lines audit log.
- An interactive Leaflet map showing observations, predictions, and intervals over an OpenStreetMap basemap.
- A Streamlit dashboard with a map, evaluation report, cross-validation scores, and audit-log viewer.

## Workflow

1. Load synthetic observations or a CSV containing `lat`, `lon`, and `value`.
2. Split the observations into training, calibration, and test sets.
3. Compare the candidate models using four-fold cross-validation on the training set.
4. Fit the selected model and calibrate prediction intervals.
5. Calculate test RMSE, empirical coverage, interval width, and residual spatial diagnostics.
6. Run automated review or request terminal input. If requested, perform one interval-scaling adjustment and repeat evaluation and review.
7. Generate the evaluation report and export the interactive map.

LangGraph coordinates these stages. Model selection is numerical; the LLM provides supporting text rather than choosing or generating models.

## Installation

Use a Python virtual environment, then install the dependencies:

```bash
git clone https://github.com/HabibaSajid321/explainable-geo-agents.git
cd explainable-geo-agents
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
python -m pip install streamlit
```

Streamlit is installed separately here because the current `requirements.txt` does not include it.

## Run the dashboard

```bash
streamlit run app_ui.py
```

Choose the bundled dataset, synthetic benchmark, or a custom CSV, then select Run Agent Pipeline. The dashboard displays the selected model, evaluation metrics, map, report, and audit log. Review in the dashboard is currently automated; interactive human review is available through the command-line interface.

## Run from the command line

```bash
# Synthetic benchmark with automated review
python run_demo.py

# Synthetic benchmark with interactive terminal review
python run_demo.py --interactive

# Bundled observations
python run_demo.py --data data/real_data.csv

# Custom observations and split seed
python run_demo.py --data path/to/observations.csv --seed 42
```

Run these commands from the repository root. The pipeline exports `spatial_trust_map.html` and appends review records to `audit_log.jsonl`.

## Data format

CSV input must contain numeric `lat`, `lon`, and `value` columns:

```csv
lat,lon,value
31.5204,74.3587,42.5
33.6844,73.0479,28.1
24.8607,67.0011,51.0
```

These rows illustrate the format; they are not measured observations. Before using a dataset, check missing-value codes, coordinate validity, units, duplicate locations, and measurement times. The current loader does not perform these checks automatically.

### Fetch air-quality observations

The repository includes an OpenAQ fetch script. It requires a separate OpenAQ API key:

```bash
# macOS / Linux
export OPENAQ_API_KEY="your-openaq-key"

# Windows PowerShell
$env:OPENAQ_API_KEY="your-openaq-key"
```

```bash
python data/fetch_openaq.py --country PK --parameter pm25 --limit 200
```

The script writes `data/real_data.csv` by default. Its current implementation takes the first latest reading returned for each location; verify that the reading belongs to the requested pollutant before treating the output as a consistent dataset. The exported CSV does not retain sensor IDs, units, or timestamps.

## Optional LLM configuration

Without an LLM key, model comparison, evaluation, review, and map export still run, using deterministic report text.

The current LLM client reads `GROQ_API_KEY` and defaults to Groq's OpenAI-compatible endpoint:

```bash
# macOS / Linux
export GROQ_API_KEY="your-groq-key"

# Windows PowerShell
$env:GROQ_API_KEY="your-groq-key"
```

Optional environment variables:

| Variable | Purpose | Current default |
| --- | --- | --- |
| `GROQ_API_KEY` | Bearer token for LLM requests | Unset; deterministic fallback |
| `OPENAI_BASE_URL` | OpenAI-compatible API base URL | `https://api.groq.com/openai/v1` |
| `OPENAI_MODEL` | Model identifier | `openai/gpt-oss-20b` |

Endpoint and model availability depend on the configured provider. OpenAQ credentials are independent of LLM credentials. The map's OpenStreetMap tile URL contains no API key, but map tiles and Leaflet assets require internet access.

## Repository structure

| Path | Purpose |
| --- | --- |
| `app_ui.py` | Streamlit dashboard |
| `run_demo.py` | Command-line entry point |
| `graph.py` | LangGraph state and workflow routing |
| `agents/` | Model comparison, evaluation, review, reporting, and LLM client |
| `models/` | Interpolation models, conformal wrapper, and spatial diagnostics |
| `data/` | Synthetic generator, CSV loader, OpenAQ fetcher, sample data, and map exporter |

## Current limitations

This is an experimental prototype. Its results require interpretation alongside the data and evaluation design.

- Candidates are selected from a fixed library. The workflow does not generate model code or perform evolutionary model discovery.
- Data splitting and cross-validation are random rather than spatially blocked. Scores should not be treated as evidence of generalization to unseen geographic regions.
- The current feedback loop adjusts interval scaling using test-set coverage and re-evaluates on the same set. Those adjusted results are exploratory; an independent test set is needed for a final assessment.
- The conformal wrapper uses constant-width intervals. Width-based map flags do not establish location-specific uncertainty hotspots, and ties can flag all points.
- Formal split-conformal coverage relies on assumptions such as exchangeability and an appropriate calibration quantile. Spatial dependence and the current scaling adjustment mean the reported target should not be presented as an unconditional guarantee.
- The dashboard coverage slider is not yet connected to graph evaluation; the pipeline currently uses the default 90% target. Map popup text also assumes 90% coverage.
- The bundled CSV needs data-quality review, including the negative `-999` reading and unusually large values. These are currently passed directly into modeling.
- Moran's I significance handling needs further validation, including small-sample cases. A nonsignificant result does not establish spatial independence.
- Synthetic coordinates are generated on a unit square; their placement on the geographic basemap does not represent real observation locations.

## Development priorities

Connect coverage configuration through the graph, validate input data and pollutant metadata, separate feedback tuning from final testing, and improve spatial evaluation and interval diagnostics. Further improvements include dashboard review controls, per-run output isolation, and automated regression checks.

## Related work

GeoEvolve is relevant background on multi-agent geospatial model discovery. TrustGeoAgent explores a narrower workflow centered on predefined model comparison, evaluation, reporting, and review.

Luo, P., Lou, X., Zheng, Y., Zheng, Z., & Ermon, S. (2025). [GeoEvolve: Automating Geospatial Model Discovery via Multi-Agent Large Language Models](https://arxiv.org/abs/2509.21593).
