from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
a = pd.read_parquet(ROOT / "data/interim/sparcs_2022_adults.parquet").reset_index(drop=True)
b = pd.read_parquet(ROOT / "data/interim/check/sparcs_2022_adults.parquet").reset_index(drop=True)
pd.testing.assert_frame_equal(a, b)
print("cohort file: identical")
for s in [0, 100, 101, 102, 103, 104]:
    x = pd.read_parquet(ROOT / f"data/splits/split_seed{s}.parquet")
    y = pd.read_parquet(ROOT / f"data/splits_check/split_seed{s}.parquet")
    assert x["split"].equals(y["split"]) and x["hospital"].equals(y["hospital"]), f"seed {s} differs"
    print(f"split seed {s}: identical")
print("ALL CHECKS PASSED")