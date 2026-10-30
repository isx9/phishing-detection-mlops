import json
import os

import pandas as pd

df = pd.read_csv("dataset_phishing.csv")
os.makedirs("samples", exist_ok=True)

for label in ["phishing", "legitimate"]:
    # Pick one random URL of this class
    row = df[df["status"] == label].sample(1, random_state=7).iloc[0]
    # Keep only the 87 features (drop the URL text and the label) as numbers
    features = row.drop(["url", "status"]).astype(float).to_dict()

    with open(f"samples/{label}.json", "w") as f:
        json.dump(features, f, indent=2)
    print(f"{label}: {row['url']}  ->  samples/{label}.json")