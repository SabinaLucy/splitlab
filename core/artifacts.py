from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "docs" / "figures"
TABLE_DIR = ROOT / "docs" / "tables"


def save_fig(name, fig=None, dpi=200):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = fig or plt.gcf()
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_table(name, df: pd.DataFrame):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path