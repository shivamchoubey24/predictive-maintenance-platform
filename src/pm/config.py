"""Central configuration: paths, constants, cost assumptions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

SEED = 42
SUBSET = "FD001"
RUL_CAP = 125          # piece-wise linear RUL target cap (standard C-MAPSS practice)
WINDOW = 30            # GRU sequence length (cycles)
ROLL = 10              # rolling window for mean/std features
DELTA = 5              # lag for the "delta" feature
HORIZON = 30           # "fails within the next N cycles" classification horizon
HEALTHY_CYCLES = 50    # early-life cycles treated as healthy for anomaly detection

# FD001 has a single operating condition; sensors 1,5,6,10,16,18,19 are constant
# (and the 3 operating settings carry no information), so they are dropped.
SENSORS = [f"s{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]

# (name, upper RUL bound in cycles) - checked in order
ALERT_LEVELS = [("Critical", 15), ("Warning", 30), ("Watch", 60)]

# Illustrative cost assumptions (USD). All are editable in the dashboard.
COSTS = dict(
    unplanned=50_000,        # engine fails in service
    planned=8_000,           # planned maintenance event
    wasted_per_cycle=40,     # value of remaining life thrown away per cycle
    fn=50_000,               # classifier: missed failure
    fp=5_000,                # classifier: unnecessary early maintenance
)


def ensure_dirs() -> None:
    for p in (ARTIFACTS, REPORTS, FIGURES):
        p.mkdir(parents=True, exist_ok=True)
