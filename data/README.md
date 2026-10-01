# Data

Small public benchmark datasets are included so the project runs out of the box (about 6 MB).

| File(s) | Source | Notes |
|---|---|---|
| `raw/train_FD001.txt`, `raw/test_FD001.txt`, `raw/RUL_FD001.txt` | NASA Prognostics Center of Excellence, **C-MAPSS** turbofan run-to-failure simulations (FD001: 100 train engines, 100 test engines, 1 operating condition, 1 fault mode). Saxena & Goebel (2008), *Damage propagation modeling for aircraft engine run-to-failure simulation*. | Copied from a public GitHub mirror; row counts match the official release (20,631 train rows, 13,096 test rows, 100 RUL values). |
| `raw/ai4i2020.csv` | **AI4I 2020 Predictive Maintenance Dataset**, UCI Machine Learning Repository. S. Matzka (2020). CC BY 4.0. | 10,000 synthetic machine records, 3.39% failures. |

If you prefer the originals, replace the files with the official downloads; the file names and format are unchanged.
