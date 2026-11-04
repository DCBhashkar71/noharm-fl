# NoHarm-FL
Certified join-or-stay-local federated learning for hospital mortality prediction.

Data (not included): NY SPARCS 2022 de-identified inpatient discharges from https://health.data.ny.gov and the PhysioNet/CinC Challenge 2012 from https://physionet.org/content/challenge-2012/1.0.0/ . Place files in data/raw/ and follow notebooks/01_eda_and_power_table.ipynb.

Run: python scripts/run_harm_seed_v2.py <seed> (confirmatory seeds 101-104), then notebooks 04 onward for tables and figures.
Pre-registration: configs/prereg.yaml (git tags prereg-v1, prereg-frozen, prereg-amendment-1 to 3).