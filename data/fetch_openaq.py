"""
Fetch real spatial data from OpenAQ (air quality monitoring stations)
and save it as a CSV of (lat, lon, value) rows.

Run this ONCE on your own machine (needs internet + a free OpenAQ key):

    export OPENAQ_API_KEY="your-key-here"
    python data/fetch_openaq.py --country PK --parameter pm25 --limit 200

Get a free key at: https://explore.openaq.org  (sign up -> account -> API keys)

parameters_id reference (common ones): pm25=2, pm10=1, o3=3, no2=5, co=8
"""
import argparse
import os
import time
import csv
import sys
import urllib.request
import json

BASE_URL = "https://api.openaq.org/v3"


def _get(path, api_key, params=None):
    """Small GET helper: builds the URL, attaches the required API-key
    header, and parses the JSON response."""
    if params:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{BASE_URL}{path}?{query}"
    else:
        url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, headers={"X-API-Key": api_key})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def fetch_locations(api_key, country_code, parameter_id, limit=200):
    """Step 1: ask OpenAQ for monitoring STATIONS (their lat/lon) in a
    country that measure the given pollutant."""
    data = _get("/locations", api_key, {
        "iso": country_code,
        "parameters_id": parameter_id,
        "limit": limit,
    })
    return data.get("results", [])


def fetch_latest_value(api_key, location_id):
    """Step 2: for one station, ask for its most recent sensor reading."""
    try:
        data = _get(f"/locations/{location_id}/latest", api_key)
        results = data.get("results", [])
        if not results:
            return None
        return results[0].get("value")
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="PK", help="ISO country code, e.g. PK, US, DE")
    parser.add_argument("--parameter", default="pm25", choices=["pm25", "pm10", "o3", "no2", "co"])
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--out", default="data/real_data.csv")
    args = parser.parse_args()

    param_ids = {"pm10": 1, "pm25": 2, "o3": 3, "no2": 5, "co": 8}

    api_key = os.environ.get("OPENAQ_API_KEY")
    if not api_key:
        sys.exit("Set OPENAQ_API_KEY first: export OPENAQ_API_KEY=your-key-here")

    print(f"Fetching {args.parameter} stations in {args.country} ...")
    locations = fetch_locations(api_key, args.country, param_ids[args.parameter], args.limit)
    print(f"Found {len(locations)} stations. Pulling latest readings ...")

    rows = []
    for i, loc in enumerate(locations):
        coords = loc.get("coordinates") or {}
        lat, lon = coords.get("latitude"), coords.get("longitude")
        if lat is None or lon is None:
            continue
        value = fetch_latest_value(api_key, loc["id"])
        if value is None:
            continue
        rows.append((lat, lon, value))
        time.sleep(0.1)  # be polite to the free-tier rate limit
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(locations)} processed")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon", "value"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} station readings to {args.out}")


if __name__ == "__main__":
    main()
