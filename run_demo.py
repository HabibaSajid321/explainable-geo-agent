"""
Run the full TrustGeoAgent pipeline end to end on a synthetic spatial
field and print a results summary.

    python run_demo.py            # batch mode (auto-approve/flag)
    python run_demo.py --interactive   # pause for human approval
"""
import sys
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from data.synthetic import make_spatial_field, train_calib_test_split
from data.real_loader import load_real_field
from data.viz import export_interactive_map
from graph import build_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", default=None,
                         help="Path to a real-data CSV from fetch_openaq.py "
                              "(lat,lon,value columns). Omit to use the "
                              "synthetic benchmark.")
    args = parser.parse_args()

    if args.data:
        coords, values = load_real_field(args.data)
        print(f"Loaded {len(values)} real observations from {args.data}")
    else:
        coords, values, _true = make_spatial_field(seed=args.seed)
    split = train_calib_test_split(coords, values, seed=args.seed)

    app = build_graph()
    result = app.invoke({"split": split, "interactive_review": args.interactive})

    moran = result["eval_out"].get("moran_res", {})
    recal_attempts = result.get("recalibrate_attempts", 0)

    print("\n================ TrustGeoAgent Run Summary ================")
    print(f"Selected model : {result['discovery_out']['selected_model']}")
    print("CV scores:")
    for name, score in result["discovery_out"]["scores"].items():
        print(f"  - {name}: {score:.4f}")
    print(f"\nTest RMSE      : {result['eval_out']['rmse']:.4f}")
    print(f"Conformal Scaling Factor: {result['eval_out'].get('scaling_factor', 1.0):.3f}")
    print(f"Coverage       : target {result['eval_out']['target_coverage']*100:.0f}% "
          f"/ empirical {result['eval_out']['empirical_coverage']*100:.1f}%")
    print(f"Mean interval width: {result['eval_out']['mean_interval_width']:.4f}")
    print(f"Moran's I (Spatial Residual Autocorrelation): {moran.get('morans_i', 0.0)} "
          f"(z-score: {moran.get('z_score', 0.0)})")
    if recal_attempts > 0:
        print(f"LangGraph Autonomous Recalibrations: {recal_attempts} step(s) executed")

    map_path = export_interactive_map(
        result["eval_out"],
        selected_model=result["discovery_out"]["selected_model"],
        output_path="spatial_trust_map.html"
    )
    print(f"\nInteractive Spatial Map Exported: {map_path}")

    print(f"\n--- Explanation ---\n{result['explanation']}")
    print(f"\n--- Review decision ---\n{result['review_out']['decision']}  "
          f"{result['review_out']['note']}")
    print("\n(Full audit trail appended to audit_log.jsonl)")


if __name__ == "__main__":
    main()

