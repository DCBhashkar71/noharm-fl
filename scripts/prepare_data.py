"""Rebuild the cleaned adult SPARCS 2022 cohort from the raw CSV.
Usage: python scripts/prepare_data.py [output_path]"""
import sys, glob
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/interim/sparcs_2022_adults.parquet"

f = glob.glob(str(ROOT / "data/raw/sparcs/2022/*.csv"))[0]
df = pd.read_csv(f, dtype=str, low_memory=False)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
n0 = len(df)

df = df.dropna(subset=["permanent_facility_id"]).copy()
n1 = len(df)
df["y"] = (df["patient_disposition"] == "Expired").astype(int)
df = df[~df["facility_name"].str.contains("Calvary", case=False, na=False)].copy()   # palliative facilities, see prereg.yaml
n2 = len(df)
df = df[df["age_group"] != "0 to 17"].copy()
df = df[df["gender"].isin(["F", "M"])].copy()
df["age_band"] = df["age_group"].replace({"18 to 29": "18-49", "30 to 49": "18-49"})
df["race_grp"] = df["race"].replace({"Multi-racial": "Other/Multi", "Other Race": "Other/Multi"})

out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out)
print(f"raw rows {n0} -> with facility id {n1} -> without Calvary {n2} -> adults, F/M {len(df)}")
print(f"deaths {int(df['y'].sum())}, hospitals {df['permanent_facility_id'].nunique()}")
print("saved", out)