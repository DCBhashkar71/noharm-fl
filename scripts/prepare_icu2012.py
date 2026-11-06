"""Build one row per ICU stay from the PhysioNet/CinC 2012 sets A, B, C.
Usage: python scripts/prepare_icu2012.py"""
import glob, os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/physionet2012"
frames = []
for s in ["a", "b", "c"]:
    files = sorted(glob.glob(str(RAW / f"set-{s}" / "*.txt")))
    print(f"set-{s}: {len(files)} files")
    assert files, f"no files found in {RAW / ('set-' + s)}"
    for f in files:
        d = pd.read_csv(f)
        d["rid"] = int(os.path.basename(f).split(".")[0])
        d["set"] = s
        frames.append(d)
raw = pd.concat(frames, ignore_index=True)
raw = raw[raw["Value"] >= 0]                                   # -1 marks missing; valid values are non-negative

GEN = ["Age", "Gender", "Height", "ICUType"]
gen = raw[raw["Parameter"].isin(GEN)].groupby(["rid", "Parameter"])["Value"].first().unstack()
ts = raw[~raw["Parameter"].isin(GEN + ["RecordID"])]
agg = ts.groupby(["rid", "Parameter"])["Value"].agg(["min", "max", "mean", "first", "last", "count"]).unstack()
agg.columns = [f"{p}_{s}" for s, p in agg.columns]
sets = raw.groupby("rid")["set"].first()
out = gen.join(agg).join(sets)

oc = pd.concat([pd.read_csv(RAW / f"Outcomes-{s}.txt") for s in "abc"], ignore_index=True).set_index("RecordID")
out["y"] = oc["In-hospital_death"].reindex(out.index).values
assert out["y"].notna().all(), "some stays have no outcome"
out = out.rename(columns={"ICUType": "icu_type", "Age": "age", "Gender": "gender", "Height": "height"})
assert out["icu_type"].notna().all()
out["icu_type"] = out["icu_type"].astype(int); out["y"] = out["y"].astype(int)
out.index.name = "rid"
out.to_parquet(ROOT / "data/interim/physionet2012.parquet")
print(len(out), "stays;", int(out["y"].sum()), "deaths;", out.shape[1], "columns")
print(out.groupby("icu_type")["y"].agg(stays="size", deaths="sum", rate="mean").round(3))
print("missing age:", int(out["age"].isna().sum()), "gender:", int(out["gender"].isna().sum()))