"""
Explainability Agent
---------------------
This is the "Trustworthy GeoAI" piece: turns the Evaluation Agent's
numbers into a plain-language trust report a domain expert (not an ML
engineer) can act on -- what was chosen, how well-calibrated it is, and
specifically where in space the model is least reliable.
"""
from agents.llm_client import call_llm


def explain_results(discovery_out, eval_out):
    n_flagged = len(eval_out["high_uncertainty_idx"])
    moran = eval_out.get("moran_res", {})
    scaling = eval_out.get("scaling_factor", 1.0)
    
    fallback = (
        f"Model: {discovery_out['selected_model']}\n"
        f"Test RMSE: {eval_out['rmse']:.4f}\n"
        f"Target coverage: {eval_out['target_coverage']*100:.0f}% | "
        f"Empirical coverage: {eval_out['empirical_coverage']*100:.1f}%\n"
        f"Conformal Scaling Factor: {scaling:.2f}\n"
        f"Mean interval width: {eval_out['mean_interval_width']:.4f}\n"
        f"Spatial Residual Autocorrelation (Moran's I): {moran.get('morans_i', 0.0)} "
        f"(z-score: {moran.get('z_score', 0.0)}, Interpretation: {moran.get('interpretation', 'N/A')})\n"
        f"{n_flagged} of {eval_out['n_test']} test points flagged as "
        f"high-uncertainty (top quartile of interval width) -- these are "
        f"the predictions a human reviewer should scrutinize first."
    )
    summary = call_llm(
        system_prompt=(
            "You write executive trust reports on spatial prediction models for "
            "domain experts. Explain what the empirical coverage, conformal scaling, "
            "interval width, and spatial error autocorrelation (Moran's I) mean in plain "
            "language, and state clearly whether the model's uncertainty estimates look "
            "well-calibrated and spatially independent."
        ),
        user_prompt=fallback,
        fallback=fallback,
    )
    return summary

