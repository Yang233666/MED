#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import random
from pathlib import Path
from datetime import datetime

SEED = 0
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
from sklearn import datasets

from MED_method import MIS_Method, DCS_Method, MED_Method_confidence


random.seed(SEED)
np.random.seed(SEED)

def now_ts():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

DATA_DIR = "../../data/real_dataset"

def load_dataset(dataset_name: str):
    print(f"Loading dataset: {dataset_name}...")

    if dataset_name == "echocardiogram":
        fname = os.path.join(DATA_DIR, "echocardiogram.data")
        df = pd.read_csv(fname, header=None, na_values="?", on_bad_lines="skip").dropna()
        data = df.to_numpy()
        input_data = data[:, :9].astype(float)
        labels_list = data[:, 11].astype(int)

    elif dataset_name == "iris":
        input_data, labels_list = datasets.load_iris(return_X_y=True)

    elif dataset_name == "seeds":
        fname = os.path.join(DATA_DIR, "seeds_dataset.txt")
        data_x_and_y = np.loadtxt(fname, delimiter="\t")
        input_data, labels = np.split(data_x_and_y, [7], axis=1)
        labels_list = labels.flatten().astype(int)

    elif dataset_name == "wine":
        input_data, labels_list = datasets.load_wine(return_X_y=True)

    elif dataset_name == "miceprotein":
        mice = datasets.fetch_openml(name="miceprotein", version=4, as_frame=False, parser="liac-arff")
        valid_rows = ~np.isnan(mice.data).any(axis=1)
        input_data, labels_list = mice.data[valid_rows], mice.target[valid_rows]

    elif dataset_name == "abalone":
        abalone = datasets.fetch_openml(name="abalone", version=1, as_frame=False, parser="liac-arff")
        input_data, labels_list = abalone.data, abalone.target

    elif dataset_name == "cnae-9":
        cnae9 = datasets.fetch_openml(name="cnae-9", version=1, as_frame=False, parser="liac-arff")
        input_data, labels_list = cnae9.data, cnae9.target

    elif dataset_name == "vowel":
        vowel = datasets.fetch_openml(name="vowel", version=1, as_frame=False, parser="liac-arff")
        input_data, labels_list = vowel.data, vowel.target

    elif dataset_name == "synthetic_control":
        synth = datasets.fetch_openml(name="synthetic_control", version=1, as_frame=False, parser="liac-arff")
        input_data, labels_list = synth.data, synth.target

    elif dataset_name == "colon":
        fname_x = os.path.join(DATA_DIR, "Colon Cancer/colon.x.txt")
        fname_y = os.path.join(DATA_DIR, "Colon Cancer/colon.y.txt")
        data_x = np.loadtxt(fname_x, delimiter=" ")
        data_y = np.loadtxt(fname_y, delimiter="\t")
        input_data, labels_list = data_x.transpose(), data_y

    elif dataset_name == "prostate":
        fname_x = os.path.join(DATA_DIR, "Prostate Cancer/prostate.x.txt")
        fname_y = os.path.join(DATA_DIR, "Prostate Cancer/prostate.y.txt")
        data_x = np.loadtxt(fname_x, delimiter=" ")
        data_y = np.loadtxt(fname_y, delimiter="\t")
        input_data, labels_list = data_x.transpose(), data_y

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    input_data = np.asarray(input_data, dtype=np.float64)
    true_k = len(np.unique(labels_list))
    print(f"Dataset loaded: {input_data.shape[0]} samples, {input_data.shape[1]} features, {true_k} clusters.")
    return input_data, labels_list, true_k

def relative_error(k_pred, true_k):
    return abs(int(k_pred) - int(true_k)) / float(true_k)


def exact_match(k_pred, true_k):
    return int(int(k_pred) == int(true_k))

def normalized_estimation_quality(k_pred, true_k):
    """
    Q := 1 - RE, clipped to [0, 1].
    """
    q = 1.0 - relative_error(k_pred, true_k)
    return float(np.clip(q, 0.0, 1.0))

def regression_alignment_metrics(conf, target):
    conf = np.asarray(conf, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    mae = np.mean(np.abs(conf - target))
    rmse = np.sqrt(np.mean((conf - target) ** 2))

    if len(conf) >= 2 and np.std(conf) > 0 and np.std(target) > 0:
        pear = pearsonr(conf, target)[0]
        spear = spearmanr(conf, target)[0]
    else:
        pear = np.nan
        spear = np.nan

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "pearson": None if np.isnan(pear) else float(pear),
        "spearman": None if np.isnan(spear) else float(spear),
    }

def plot_confidence_vs_quality(df, out_path):
    plt.figure(figsize=(6.2, 4.8))

    mask_high = df["quality_group"] == "High quality"
    mask_low = df["quality_group"] == "Low quality"

    plt.scatter(
        df.loc[mask_high, "confidence_med"],
        df.loc[mask_high, "Q_med"],
        marker="o",
        label="High quality",
        alpha=0.9,
    )
    plt.scatter(
        df.loc[mask_low, "confidence_med"],
        df.loc[mask_low, "Q_med"],
        marker="x",
        label="Low quality",
        alpha=0.9,
    )

    x = df["confidence_med"].values.astype(float)
    y = df["Q_med"].values.astype(float)
    if len(x) >= 2 and np.std(x) > 0:
        coef = np.polyfit(x, y, deg=1)
        xx = np.linspace(0, 1, 200)
        yy = coef[0] * xx + coef[1]
        plt.plot(xx, yy, linestyle="--", label="Linear fit")

    plt.xlabel("MED confidence")
    plt.ylabel(r"Normalized estimation quality $Q = 1 - \mathrm{RE}$")
    plt.title("Confidence vs. normalized estimation quality")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_reliability_diagram_continuous(df, out_path, n_bins=4):
    conf = df["confidence_med"].values.astype(float)
    q = df["Q_med"].values.astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys = [], []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (conf >= lo) & (conf < hi)
        else:
            mask = (conf >= lo) & (conf <= hi)

        if np.any(mask):
            xs.append(np.mean(conf[mask]))
            ys.append(np.mean(q[mask]))

    plt.figure(figsize=(5.5, 5.0))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Ideal alignment")
    if len(xs) > 0:
        plt.plot(xs, ys, marker="o", label=f"MED ({n_bins} bins)")

    plt.xlabel("Confidence")
    plt.ylabel(r"Empirical mean quality $Q = 1 - \mathrm{RE}$")
    plt.title(f"Continuous reliability diagram ({n_bins} bins)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_confidence_boxplot_quality_groups(df, out_path):
    high_vals = df.loc[df["quality_group"] == "High quality", "confidence_med"].values.astype(float)
    low_vals = df.loc[df["quality_group"] == "Low quality", "confidence_med"].values.astype(float)

    plt.figure(figsize=(5.5, 4.8))
    data = [high_vals, low_vals]
    labels = ["High quality", "Low quality"]
    plt.boxplot(data, tick_labels=labels)
    plt.ylabel("MED confidence")
    plt.title("Confidence by normalized estimation quality group")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_confidence_histogram_by_quality(df, out_path):
    high_vals = df.loc[df["quality_group"] == "High quality", "confidence_med"].values.astype(float)
    low_vals = df.loc[df["quality_group"] == "Low quality", "confidence_med"].values.astype(float)

    plt.figure(figsize=(6.0, 4.8))
    plt.hist(high_vals, bins=5, alpha=0.7, label="High quality")
    plt.hist(low_vals, bins=5, alpha=0.7, label="Low quality")
    plt.xlabel("MED confidence")
    plt.ylabel("Count")
    plt.title("Confidence distribution by quality group")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def build_continuous_reliability_table(df, n_bins=4):
    conf = df["confidence_med"].values.astype(float)
    y = df["Q_med"].values.astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (conf >= lo) & (conf < hi)
        else:
            mask = (conf >= lo) & (conf <= hi)

        if np.any(mask):
            conf_mean = float(np.mean(conf[mask]))
            quality_mean = float(np.mean(y[mask]))
            gap = float(abs(conf_mean - quality_mean))
            rows.append({
                "bin_left": float(lo),
                "bin_right": float(hi),
                "count": int(np.sum(mask)),
                "mean_confidence": conf_mean,
                "mean_quality_Q": quality_mean,
                "abs_gap": gap,
            })

    return rows

def continuous_reliability_metrics(df, n_bins=4):
    rows = build_continuous_reliability_table(df, n_bins=n_bins)
    if len(rows) == 0:
        return {
            "CRPS_style_L1": None,
            "CRPS_style_L2": None,
            "num_nonempty_bins": 0,
        }

    total = sum(r["count"] for r in rows)
    l1 = sum((r["count"] / total) * abs(r["mean_confidence"] - r["mean_quality_Q"]) for r in rows)
    l2 = sum((r["count"] / total) * (r["mean_confidence"] - r["mean_quality_Q"]) ** 2 for r in rows)

    return {
        "CRPS_style_L1": float(l1),
        "CRPS_style_L2": float(np.sqrt(l2)),
        "num_nonempty_bins": int(len(rows)),
    }

def run_one_dataset(dataset_name):
    print("\n" + "=" * 80)
    print(f"Running dataset: {dataset_name}")
    print("=" * 80)

    X, _, true_k = load_dataset(dataset_name)

    k_min = 1
    k_max = int(np.floor(np.sqrt(X.shape[0])) / 2)
    k_max = max(k_min + 1, k_max)

    t0 = time.time()
    mis = MIS_Method(data=X, n_jobs=1, random_state=0)
    k_mis = int(mis.find_optimal_k(max_k=k_max))
    t_mis = time.time() - t0
    re_mis = relative_error(k_mis, true_k)

    t0 = time.time()
    dcs = DCS_Method(data=X, k_list=list(range(2, k_max + 1)), random_state=0)
    dcs.Distortions()
    k_dcs, _ = dcs.Jumps()
    k_dcs = int(k_dcs)
    t_dcs = time.time() - t0
    re_dcs = relative_error(k_dcs, true_k)

    t0 = time.time()
    med = MED_Method_confidence(data=X, n_jobs=1, random_state=0, eps=1e-12)
    k_med = int(med.find_optimal_k(max_k=k_max))
    t_med = time.time() - t0
    re_med = relative_error(k_med, true_k)
    ex_med = exact_match(k_med, true_k)
    q_med = normalized_estimation_quality(k_med, true_k)
    quality_group = "High quality" if q_med >= 0.7 else "Low quality"

    row = {
        "dataset": dataset_name,
        "n": int(X.shape[0]),
        "p": int(X.shape[1]),
        "true_k": int(true_k),
        "k_max": int(k_max),

        "k_mis": int(k_mis),
        "re_mis": float(re_mis),
        "time_mis_s": float(t_mis),

        "k_dcs": int(k_dcs),
        "re_dcs": float(re_dcs),
        "time_dcs_s": float(t_dcs),

        "k_med": int(k_med),
        "re_med": float(re_med),
        "exact_med": int(ex_med),
        "time_med_s": float(t_med),

        "selected_expert": med.selected_expert_,
        "early_accepted": bool(med.early_accepted_),

        "confidence_margin_raw": None if med.confidence_margin_raw_ is None else float(med.confidence_margin_raw_),
        "confidence_med": None if med.confidence_ is None else float(med.confidence_),

        "H_mis": None if med.H_mis_ is None else float(med.H_mis_),
        "H_dcs": None if med.H_dcs_ is None else float(med.H_dcs_),
        "D_mis": None if med.D_mis_ is None else float(med.D_mis_),
        "D_dcs": None if med.D_dcs_ is None else float(med.D_dcs_),

        "q_mis": None if med.q_mis_ is None else float(med._selected_mass(med.q_mis_, med.k_list_, med.k_mis_)),
        "q_dcs": None if med.q_dcs_ is None else float(med._selected_mass(med.q_dcs_, med.k_list_, med.k_dcs_)),

        "margin_mis": None if med.margin_mis_ is None else float(med.margin_mis_),
        "spread_mis": None if med.spread_mis_ is None else float(med.spread_mis_),

        "early_margin_signed": None if med.early_margin_signed_ is None else float(med.early_margin_signed_),
        "dispatch_margin_signed": None if med.dispatch_margin_signed_ is None else float(med.dispatch_margin_signed_),

        "Q_med": float(q_med),
        "quality_group": quality_group,
        "total_dataset_time_s": float(t_mis + t_dcs + t_med),
    }

    return row


def print_console_table(rows):
    print("-" * 123)
    print(f"{'Dataset':<20} | {'True k':<7} | {'MIS':<8} | {'DCS':<8} | {'MED':<8} | {'RE(MED)':<10} | {'Q=1-RE':<8} | {'Conf':<8} | {'Expert':<6}")
    print("-" * 123)
    for r in rows:
        print(
            f"{r['dataset']:<20} | "
            f"{r['true_k']:<7} | "
            f"{r['k_mis']:<8} | "
            f"{r['k_dcs']:<8} | "
            f"{r['k_med']:<8} | "
            f"{r['re_med']:<10.3f} | "
            f"{r['Q_med']:<8.3f} | "
            f"{r['confidence_med']:<8.3f} | "
            f"{r['selected_expert']:<6}"
        )
    print("-" * 123)


def main():
    dataset_list = [
        "echocardiogram",
        "iris",
        "seeds",
        "wine",
        "colon",
        "prostate",
        "miceprotein",
        "abalone",
        "cnae-9",
        "vowel",
        "synthetic_control",
    ]

    out_dir = Path("results_med_trust") / f"med_eval_{now_ts()}"
    safe_mkdir(str(out_dir))

    rows = []
    for ds in dataset_list:
        rows.append(run_one_dataset(ds))

    print_console_table(rows)

    df = pd.DataFrame(rows)

    confidences = [row["confidence_med"] for row in rows if row["confidence_med"] is not None]
    q_values = [row["Q_med"] for row in rows if row["confidence_med"] is not None]

    alignment_metrics = regression_alignment_metrics(
        conf=np.asarray(confidences, dtype=np.float64),
        target=np.asarray(q_values, dtype=np.float64),
    )

    high_quality_conf = [row["confidence_med"] for row in rows if row["quality_group"] == "High quality"]
    low_quality_conf = [row["confidence_med"] for row in rows if row["quality_group"] == "Low quality"]
    reliability_rows = build_continuous_reliability_table(df, n_bins=4)

    summary = {
        "num_datasets": len(rows),
        "MIS": {
            "ARE": float(np.mean([row["re_mis"] for row in rows if row["re_mis"] is not None])),
            "EM": int(np.sum([1 if row["re_mis"] == 0.0 else 0 for row in rows if row["re_mis"] is not None])),
            "time_total_s": float(np.sum([row["time_mis_s"] for row in rows])),
        },
        "DCS": {
            "ARE": float(np.mean([row["re_dcs"] for row in rows if row["re_dcs"] is not None])),
            "EM": int(np.sum([1 if row["re_dcs"] == 0.0 else 0 for row in rows if row["re_dcs"] is not None])),
            "time_total_s": float(np.sum([row["time_dcs_s"] for row in rows])),
        },
        "MED": {
            "ARE": float(np.mean([row["re_med"] for row in rows])),
            "EM": int(np.sum([row["exact_med"] for row in rows])),
            "time_total_s": float(np.sum([row["time_med_s"] for row in rows])),

            "confidence_mean": float(np.mean(confidences)),
            "confidence_high_quality_mean": float(np.mean(high_quality_conf)) if len(high_quality_conf) > 0 else None,
            "confidence_low_quality_mean": float(np.mean(low_quality_conf)) if len(low_quality_conf) > 0 else None,

            "num_selected_MIS": int(np.sum([1 for row in rows if row["selected_expert"] == "MIS"])),
            "num_selected_DCS": int(np.sum([1 for row in rows if row["selected_expert"] == "DCS"])),
            "num_early_accept": int(np.sum([1 for row in rows if row["early_accepted"]])),

            "alignment_to_Q": alignment_metrics,

            "continuous_reliability": {
                "CRPS_style_L1": float(np.mean(np.abs(np.asarray(confidences) - np.asarray(q_values)))),
                "CRPS_style_L2": float(np.mean((np.asarray(confidences) - np.asarray(q_values)) ** 2)),
            },
        },
    }
    write_json(out_dir / "summary_overall.json", summary)
    write_jsonl(out_dir / "per_dataset_results.jsonl", rows)
    df.to_csv(out_dir / "summary_table.csv", index=False)
    pd.DataFrame(reliability_rows).to_csv(out_dir / "continuous_reliability_table.csv", index=False)
    plot_confidence_vs_quality(df, out_dir / "confidence_vs_quality_Q.png")
    plot_reliability_diagram_continuous(df,out_dir / f"reliability_continuous_quality_{4}bins.png",n_bins=4,)
    plot_confidence_boxplot_quality_groups(df, out_dir / "confidence_boxplot_high_vs_low_quality.png")
    plot_confidence_histogram_by_quality(df, out_dir / "confidence_histogram_high_vs_low_quality.png")

    print("\nOverall summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()