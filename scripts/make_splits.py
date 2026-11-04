"""Create stratified 50/25/25 train/cert/test splits within hospital x outcome.
Usage: python scripts/make_splits.py <seed> [<seed> ...] [--out DIR]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
args = sys.argv[1:]
out_dir = ROOT / "data/splits"
if "--out" in args:
    i = args.index("--out"); out_dir = Path(args[i + 1]); args = args[:i] + args[i + 2:]
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(ROOT / "data/interim/sparcs_2022_adults.parquet").reset_index(drop=True)
for s in map(int, args):
    rng = np.random.default_rng(s)
    u = pd.Series(rng.random(len(df)))
    r = u.groupby([df["permanent_facility_id"], df["y"]]).rank(pct=True)
    split = np.where(r <= 0.50, "train", np.where(r <= 0.75, "cert", "test"))
    pd.DataFrame({"row": df.index, "hospital": df["permanent_facility_id"], "split": split}).to_parquet(out_dir / f"split_seed{s}.parquet")
    print("seed", s, pd.Series(split).value_counts(normalize=True).round(4).to_dict())