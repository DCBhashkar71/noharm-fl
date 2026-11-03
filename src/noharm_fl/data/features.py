from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[3]
FEATURES = ["age_group", "gender", "race", "ethnicity", "type_of_admission",
            "emergency_department_indicator", "payment_typology_1", "dx_sys"]
APR = ["apr_severity_of_illness_code", "apr_risk_of_mortality"]


def load(seed, with_apr=False, min_deaths=100):
    df = pd.read_parquet(ROOT / "data/interim/sparcs_2022_adults.parquet").reset_index(drop=True)
    split = pd.read_parquet(ROOT / f"data/splits/split_seed{seed}.parquet")["split"].values
    df["dx_sys"] = df["ccsr_diagnosis_code"].str[:3]
    cols = FEATURES + (APR if with_apr else [])
    enc = OneHotEncoder(handle_unknown="ignore", min_frequency=50)
    enc.fit(df.loc[split == "train", cols].fillna("NA"))      # encoder sees training rows only
    X = enc.transform(df[cols].fillna("NA")).tocsr()
    y = df["y"].values
    h = df["permanent_facility_id"].values
    deaths = df.groupby("permanent_facility_id")["y"].sum()
    keep = deaths[deaths >= min_deaths].index.values
    return X, y, h, split, keep