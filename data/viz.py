"""
Interactive Geo-Visualization Exporter
-------------------------------------
Exports spatial predictions, conformal uncertainty intervals, and
high-uncertainty hotspots into a standalone Leaflet.js HTML map.
"""
import os
import json
import numpy as np


def export_interactive_map(eval_out: dict, selected_model: str, output_path: str = "spatial_trust_map.html") -> str:
    """
    Generates a standalone Leaflet HTML visual analytics dashboard.
    """
    coords = eval_out["X_test"]
    preds = eval_out["preds"]
    lo = eval_out["lo"]
    hi = eval_out["hi"]
    y_test = eval_out["y_test"]
    high_idx = set(eval_out["high_uncertainty_idx"])
    
    # Calculate map center
    mean_lat = float(np.mean(coords[:, 0]))
    mean_lon = float(np.mean(coords[:, 1]))

    points_data = []
    for i in range(len(coords)):
        lat, lon = float(coords[i, 0]), float(coords[i, 1])
        is_high = i in high_idx
        obs = float(y_test[i])
        pred = float(preds[i])
        l_val = float(lo[i])
        h_val = float(hi[i])
        width = h_val - l_val
        
        points_data.append({
            "lat": lat,
            "lon": lon,
            "obs": round(obs, 2),
            "pred": round(pred, 2),
            "lo": round(l_val, 2),
            "hi": round(h_val, 2),
            "width": round(width, 2),
            "is_high": is_high
        })

    points_json = json.dumps(points_data)
    rmse = round(float(eval_out["rmse"]), 2)
    emp_cov = round(float(eval_out["empirical_coverage"]) * 100, 1)
    tgt_cov = round(float(eval_out["target_coverage"]) * 100, 0)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>TrustGeoAgent - Spatial Uncertainty Map</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; }}
        #map {{ width: 100vw; height: 100vh; }}
        .header-panel {{
            position: absolute; top: 16px; left: 60px; z-index: 1000;
            background: rgba(15, 23, 42, 0.9); backdrop-filter: blur(8px);
            padding: 16px 24px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        }}
        .header-title {{ font-size: 20px; font-weight: 700; color: #38bdf8; margin: 0 0 6px 0; display: flex; align-items: center; gap: 8px; }}
        .header-subtitle {{ font-size: 13px; color: #94a3b8; margin: 0; }}
        .badge-grid {{ display: flex; gap: 12px; margin-top: 12px; }}
        .badge {{ background: #1e293b; padding: 6px 12px; border-radius: 6px; font-size: 12px; font-weight: 600; border: 1px solid #334155; }}
        .badge-val {{ color: #a7f3d0; }}
        .legend-panel {{
            position: absolute; bottom: 24px; right: 24px; z-index: 1000;
            background: rgba(15, 23, 42, 0.9); backdrop-filter: blur(8px);
            padding: 14px 18px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1); font-size: 12px;
        }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
        .dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
        .dot-normal {{ background: #10b981; box-shadow: 0 0 8px #10b981; }}
        .dot-high {{ background: #ef4444; box-shadow: 0 0 8px #ef4444; }}
    </style>
</head>
<body>
    <div class="header-panel">
        <div class="header-title">🌍 TrustGeoAgent Spatial Analytics</div>
        <div class="header-subtitle">Model: <strong>{selected_model}</strong> | Spatial Uncertainty Dashboard</div>
        <div class="badge-grid">
            <div class="badge">RMSE: <span class="badge-val">{rmse}</span></div>
            <div class="badge">Target Coverage: <span class="badge-val">{tgt_cov}%</span></div>
            <div class="badge">Empirical Coverage: <span class="badge-val">{emp_cov}%</span></div>
        </div>
    </div>

    <div class="legend-panel">
        <div style="font-weight:700; margin-bottom:8px; color:#e2e8f0;">Uncertainty Legend</div>
        <div class="legend-item"><span class="dot dot-normal"></span> Normal Uncertainty</div>
        <div class="legend-item"><span class="dot dot-high"></span> High Uncertainty (Top 25% Width)</div>
    </div>

    <div id="map"></div>

    <script>
        const map = L.map('map').setView([{mean_lat}, {mean_lon}], 6);
        L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
            attribution: '&copy; OpenStreetMap contributors'
        }}).addTo(map);

        const data = {points_json};
        const bounds = [];

        data.forEach(pt => {{
            bounds.push([pt.lat, pt.lon]);
            const color = pt.is_high ? '#ef4444' : '#10b981';
            const radius = pt.is_high ? 9 : 6;
            
            const marker = L.circleMarker([pt.lat, pt.lon], {{
                radius: radius,
                fillColor: color,
                color: '#ffffff',
                weight: 1.5,
                opacity: 0.9,
                fillOpacity: 0.85
            }}).addTo(map);

            const popupContent = `
                <div style="font-family: sans-serif; font-size: 13px; color: #1e293b; padding: 4px;">
                    <div style="font-weight: 700; font-size: 14px; margin-bottom: 6px; color: ${{pt.is_high ? '#dc2626' : '#059669'}};">
                        ${{pt.is_high ? '⚠️ High Uncertainty Flagged' : '✅ Calibrated Prediction'}}
                    </div>
                    <div><strong>Coordinates:</strong> (${{pt.lat.toFixed(4)}}, ${{pt.lon.toFixed(4)}})</div>
                    <div><strong>Observed Value:</strong> ${{pt.obs}}</div>
                    <div><strong>Predicted Value:</strong> ${{pt.pred}}</div>
                    <div><strong>90% Conformal Interval:</strong> [${{pt.lo}}, ${{pt.hi}}]</div>
                    <div><strong>Interval Width:</strong> ${{pt.width}}</div>
                </div>
            `;
            marker.bindPopup(popupContent);
        }});

        if (bounds.length > 0) {{
            map.fitBounds(bounds, {{ padding: [50, 50] }});
        }}
    </script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return os.path.abspath(output_path)
