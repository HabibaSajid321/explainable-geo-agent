"""
Loads the CSV produced by fetch_openaq.py into the same (coords, values)
shape that data/synthetic.py produces -- so nothing downstream (models,
agents, graph) needs to change.
"""
import csv
import numpy as np


def load_real_field(csv_path):
    lats, lons, values = [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lats.append(float(row["lat"]))
            lons.append(float(row["lon"]))
            values.append(float(row["value"]))

    coords = np.column_stack([lats, lons])
    values = np.array(values)

    return coords, values
