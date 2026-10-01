"""
Discovery Agent
---------------
Analogous to GeoEvolve's code evolver + GeoKnowRAG: instead of open-ended
code generation, this scoped version selects and configures the best
candidate from a small library of known spatial models via cross-validated
RMSE, then asks an LLM to justify the pick in domain terms. Swap in an
actual code-generation loop later to move from "select" to "evolve."
"""
import numpy as np
from sklearn.model_selection import KFold

from models.spatial_models import CANDIDATE_MODELS
from agents.llm_client import call_llm

DOMAIN_NOTES = {
    "idw": "Deterministic, no training data assumptions, but ignores directional "
           "anisotropy and provides no native uncertainty estimate.",
    "gwr_lite": "Captures local non-stationarity (varying relationships across "
                "space) but is sensitive to bandwidth choice and can be unstable "
                "with sparse local samples.",
    "kriging_gp": "Models spatial covariance directly and yields calibrated "
                  "predictive variance, but is more compute-heavy and assumes a "
                  "reasonably well-specified covariance structure.",
}


def _cv_rmse(model_cls, X, y, k=4, seed=0):
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    errs = []
    for tr_idx, va_idx in kf.split(X):
        m = model_cls().fit(X[tr_idx], y[tr_idx])
        preds = m.predict(X[va_idx])
        errs.append(np.sqrt(np.mean((preds - y[va_idx]) ** 2)))
    return float(np.mean(errs))


def run_discovery(X_train, y_train):
    """Cross-validate every candidate model and pick the best by RMSE."""
    scores = {name: _cv_rmse(cls, X_train, y_train) for name, cls in CANDIDATE_MODELS.items()}
    best_name = min(scores, key=scores.get)

    scoreboard = "\n".join(f"  - {name}: CV-RMSE={rmse:.4f}" for name, rmse in scores.items())
    fallback = (
        f"Selected '{best_name}' (lowest cross-validated RMSE among candidates).\n"
        f"{scoreboard}\n"
        f"Note: {DOMAIN_NOTES[best_name]}"
    )
    rationale = call_llm(
        system_prompt=(
            "You are a geospatial modeling assistant. Given cross-validation "
            "scores for candidate spatial interpolation models, explain in 3-4 "
            "sentences why the top-scoring model was selected and one risk to "
            "watch for. Be concrete and technical."
        ),
        user_prompt=f"Scores (lower RMSE is better):\n{scoreboard}\n\nSelected: {best_name}",
        fallback=fallback,
    )

    return {
        "scores": scores,
        "selected_model": best_name,
        "model_cls": CANDIDATE_MODELS[best_name],
        "rationale": rationale,
    }
