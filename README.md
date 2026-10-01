# TrustGeoAgent

A human-in-the-loop, explainable multi-agent pipeline for geospatial model
selection and uncertainty quantification.

## Why this project

Peng Luo's [GeoEvolve](https://arxiv.org/abs/2509.21593) introduced a
multi-agent LLM framework that couples evolutionary code search with a
geospatial-knowledge RAG module (GeoKnowRAG) to automatically discover
spatial models, evaluated on kriging interpolation and geospatial conformal
prediction. It's fully autonomous — there's no point where a domain expert
reviews *why* a model was chosen or *where* it's least trustworthy before
the result is used.

TrustGeoAgent is a scoped, working prototype of the piece I think is
missing: a **human-review and audit layer** bolted onto a GeoEvolve-style
discovery pipeline, aimed squarely at the "Trustworthy GeoAI" side of
that agenda — explainability and uncertainty quantification a non-ML domain
expert can actually act on, not just a metric in a paper.

## Architecture

```
Discovery Agent  ->  Evaluation Agent  ->  Explainability Agent  ->  Human Review Agent
 (select/CV        (fit + split-        (plain-language           (approve / reject /
  candidate         conformal            trust report)             flag, w/ audit log)
  model)            intervals)
```

- **Discovery Agent** — cross-validates a small library of candidate spatial
  models (IDW, GWR-lite, kriging via GP) and picks the best by RMSE, with an
  LLM-generated rationale. Stands in for GeoEvolve's code-evolver + RAG step;
  the natural next iteration is to replace "select from library" with
  "generate and mutate candidates."
- **Evaluation Agent** — fits the selected model, calibrates a
  **split-conformal** interval (distribution-free coverage guarantee) on a
  held-out calibration set, and reports test RMSE + empirical coverage.
  This mirrors GeoEvolve's own use of geospatial conformal prediction as an
  uncertainty benchmark.
- **Explainability Agent** — turns the numbers into a short trust report:
  what was chosen, how well-calibrated the intervals actually are, and
  which test points are flagged as high-uncertainty.
- **Human Review Agent** — the gate GeoEvolve doesn't have. Approves,
  rejects, or flags the run based on whether empirical coverage matches the
  target (or takes interactive y/n/comment input), and appends every
  decision to `audit_log.jsonl`. Same approval-workflow pattern as the
  human-in-the-loop steps in my SOC-assistant and bid-automation projects,
  applied here to model trust instead of security remediation.

Built with **LangGraph** for orchestration, matching the agent architecture
I already use in production.

## Run it

```bash
pip install -r requirements.txt
python run_demo.py                 # batch mode: auto-approve/flag
python run_demo.py --interactive   # pause for human y/n/comment
```

On the bundled synthetic benchmark, the pipeline auto-selects a
GP-based kriging model but **flags it for review** because empirical
coverage (82.2%) falls short of the 90% target — exactly the kind of
silent miscalibration a fully autonomous pipeline wouldn't surface.

## Extending to real data

The current benchmark is a synthetic Gaussian-process field (zero external
downloads, reproducible, standard for testing interpolation methods). To
point this at a real dataset instead — e.g. soil-moisture stations or
similar structured spatial data — replace `data/synthetic.py`'s
`make_spatial_field()` with a loader that returns `(coords, values)`, and
everything downstream (model selection, conformal calibration, review)
works unchanged.

## LLM configuration

Discovery and explainability agents call an OpenAI-compatible endpoint if
`OPENAI_API_KEY` is set (also works against a local Ollama/vLLM server via
`OPENAI_BASE_URL`); otherwise they fall back to deterministic templates so
the pipeline always runs end-to-end without a key.

## Reference

Luo, P., Lou, X., Zheng, Y., Zheng, Z., & Ermon, S. (2025). *GeoEvolve:
Automating Geospatial Model Discovery via Multi-Agent Large Language
Models.* arXiv:2509.21593.
