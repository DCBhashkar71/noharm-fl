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


def load_temporal(seed, year="2023", with_apr=False, min_deaths=100):
    """2022 train/cert/test exactly as load(); also encodes another year's cohort with the same encoder (fit on 2022 train rows)."""
    df = pd.read_parquet(ROOT / "data/interim/sparcs_2022_adults.parquet").reset_index(drop=True)
    split = pd.read_parquet(ROOT / f"data/splits/split_seed{seed}.parquet")["split"].values
    df["dx_sys"] = df["ccsr_diagnosis_code"].str[:3]
    cols = FEATURES + (APR if with_apr else [])
    enc = OneHotEncoder(handle_unknown="ignore", min_frequency=50)
    enc.fit(df.loc[split == "train", cols].fillna("NA"))
    X = enc.transform(df[cols].fillna("NA")).tocsr()
    df2 = pd.read_parquet(ROOT / f"data/interim/sparcs_{year}_adults.parquet").reset_index(drop=True)
    df2["dx_sys"] = df2["ccsr_diagnosis_code"].str[:3]
    X2 = enc.transform(df2[cols].fillna("NA")).tocsr()
    deaths = df.groupby("permanent_facility_id")["y"].sum()
    keep = deaths[deaths >= min_deaths].index.values
    return X, df["y"].values, df["permanent_facility_id"].values, split, keep, X2, df2["y"].values, df2["permanent_facility_id"].values, df2