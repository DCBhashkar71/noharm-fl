"""Rebuild the cleaned adult SPARCS cohort for one year from the raw CSV.
Usage: python scripts/prepare_data.py [year] [output_path]   (default year 2022)"""
import sys, glob
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
year = sys.argv[1] if len(sys.argv) > 1 else "2022"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / f"data/interim/sparcs_{year}_adults.parquet"

f = glob.glob(str(ROOT / f"data/raw/sparcs/{year}/*.csv"))[0]
df = pd.read_csv(f, dtype=str, low_memory=False)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
need = ["permanent_facility_id", "facility_name", "patient_disposition", "age_group", "gender", "race", "ethnicity",
        "type_of_admission", "emergency_department_indicator", "payment_typology_1", "ccsr_diagnosis_code"]
missing = [c for c in need if c not in df.columns]
assert not missing, f"columns missing in {year} file: {missing}\nall columns: {list(df.columns)}"
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
print(f"{year}: raw rows {n0} -> with facility id {n1} -> without Calvary {n2} -> adults, F/M {len(df)}")
print(f"deaths {int(df['y'].sum())}, hospitals {df['permanent_facility_id'].nunique()}")
print("saved", out)