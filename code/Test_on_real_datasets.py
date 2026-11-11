"""
This script evaluates clustering validity indices to find the optimal number of clusters (k).

It can be run on a single dataset or on all available datasets in a batch process.
The script now includes timing information and a final summary table for batch runs.
"""

import os

SEED = 0
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import numpy as np
random.seed(SEED)
np.random.seed(SEED)

import argparse
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from tqdm import tqdm
import time, pickle
from datetime import datetime

from MED_method import MED_Method

def load_dataset(dataset_name: str) -> (np.ndarray, list, int):
    """ Loads the specified dataset. """
    print(f"Loading dataset: {dataset_name}...")

    base_data_path = '../../data/real_dataset'

    if dataset_name == 'echocardiogram':
        fname = os.path.join(base_data_path, 'echocardiogram.data')
        df = pd.read_csv(fname, header=None, na_values='?', on_bad_lines='skip').dropna()
        data = df.to_numpy()
        input_data, labels_list = data[:, :9], data[:, 11].astype(int)
    elif dataset_name == 'iris':
        input_data, labels_list = datasets.load_iris(return_X_y=True)
    elif dataset_name == 'seeds':
        fname = os.path.join(base_data_path, 'seeds_dataset.txt')
        data_x_and_y = np.loadtxt(fname, delimiter='\t')
        input_data, labels = np.split(data_x_and_y, [7], axis=1)
        labels_list = labels.flatten().astype(int)
    elif dataset_name == 'wine':
        input_data, labels_list = datasets.load_wine(return_X_y=True)
    elif dataset_name == 'miceprotein':
        mice = datasets.fetch_openml(name='miceprotein', version=4, as_frame=False, parser='liac-arff')
        valid_rows = ~np.isnan(mice.data).any(axis=1)
        input_data, labels_list = mice.data[valid_rows], mice.target[valid_rows]
    elif dataset_name == 'abalone':
        abalone = datasets.fetch_openml(name='abalone', version=1, as_frame=False, parser='liac-arff')
        input_data, labels_list = abalone.data, abalone.target
    elif dataset_name == 'cnae-9':
        cnae9 = datasets.fetch_openml(name='cnae-9', version=1, as_frame=False, parser='liac-arff')
        input_data, labels_list = cnae9.data, cnae9.target
    elif dataset_name == 'vowel':
        vowel = datasets.fetch_openml(name='vowel', version=1, as_frame=False, parser='liac-arff')
        input_data, labels_list = vowel.data, vowel.target
    elif dataset_name == 'synthetic_control':
        synth = datasets.fetch_openml(name='synthetic_control', version=1, as_frame=False, parser='liac-arff')
        input_data, labels_list = synth.data, synth.target

    elif dataset_name == 'colon':
        fname_x = os.path.join(base_data_path, 'Colon Cancer/colon.x.txt')
        fname_y = os.path.join(base_data_path, 'Colon Cancer/colon.y.txt')
        data_x, data_y = np.loadtxt(fname_x, delimiter=' '), np.loadtxt(fname_y, delimiter='\t')
        input_data, labels_list = data_x.transpose(), data_y
    elif dataset_name == 'prostate':
        fname_x = os.path.join(base_data_path, 'Prostate Cancer/prostate.x.txt')
        fname_y = os.path.join(base_data_path, 'Prostate Cancer/prostate.y.txt')
        data_x, data_y = np.loadtxt(fname_x, delimiter=' '), np.loadtxt(fname_y, delimiter='\t')
        input_data, labels_list = data_x.transpose(), data_y
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    true_k = len(np.unique(labels_list))
    print(f"Dataset loaded: {input_data.shape[0]} samples, {input_data.shape[1]} features, {true_k} clusters.")
    return input_data, labels_list, true_k

def run_analysis_for_dataset(dataset_name: str) -> dict:

    dataset_start_mono = time.monotonic()
    input_data, _, true_k = load_dataset(dataset_name)

    k_min = 1
    k_max = int(np.floor(input_data.shape[0] ** 0.5) / 2)
    k_max = max(k_min + 1, k_max)
    k_list = list(range(k_min, k_max + 1))
    if not k_list:
        print(f"Warning: Calculated k range is empty for '{dataset_name}'. Setting k_max=10.")
        k_max, k_list = 10, list(range(k_min, 11))
    optimal_k_results = {}
    print(f"\n--- Starting Evaluation for '{dataset_name}' (k from {k_min} to {k_max}) ---")

    jump_method = MED_Method(data=input_data)
    optimal_k = jump_method.find_optimal_k(max_k=k_max)

    print("\n" + "="*40)
    print(f"Optimal number of clusters found: {optimal_k}")
    print("="*40)

    method_name = str(jump_method.name)
    dataset_duration = time.monotonic() - dataset_start_mono

    print("\n--- Evaluation Results ---")
    print(f"Dataset: {dataset_name}")
    print(f"Ground Truth Number of Clusters (True k): {true_k}")
    print("-" * 60)
    print(f"{'Method':<30} | {'Predicted k':<12} | {'Relative Error':<15}")
    print("-" * 60)
    error_values = []
    for method, k_pred in sorted(optimal_k_results.items()):
        if k_pred is not None and true_k > 0:
            error = abs(k_pred - true_k) / true_k
            error_values.append(error)
            print(f"{method:<30} | {k_pred:<12} | {error:<15.3f}")
        else:
            print(f"{method:<30} | {'N/A':<12} | {'N/A':<15}")
    if error_values:
        average_re = np.mean(error_values)
        print("-" * 60)
        print(f"{'Average Relative Error (ARE)':<30} | {'':<12} | {average_re:<15.3f}")
    print(f"{'Time taken for this dataset':<30} | {dataset_duration:<12.2f} | {'seconds':<15}")
    print("-" * 60)

    # predicted_k_ijm = optimal_k_results.get('Inverse_JumpsMethod')
    predicted_k_ijm = optimal_k_results.get(method_name)
    rel_error_ijm = None
    if predicted_k_ijm is not None and true_k > 0:
        rel_error_ijm = abs(predicted_k_ijm - true_k) / true_k

    return {
        'name': dataset_name, 'true_k': true_k, 'predicted_k': predicted_k_ijm,
        'relative_error': rel_error_ijm, 'duration': dataset_duration
    }


def print_summary_table(results: list):
    """Prints the final summary table for a batch run."""
    print("\n" + "="*80)
    print(f"{'':<32}FINAL SUMMARY OVERVIEW")
    print("="*80)
    print(f"{'Dataset':<20} | {'True k':<10} | {'Predicted k':<12} | {'Relative Error':<15} | {'Time Taken (s)':<15}")
    print("-" * 80)
    valid_errors, total_time = [], 0
    for res in results:
        pred_k_str = str(res['predicted_k']) if res['predicted_k'] is not None else "N/A"
        error_str = f"{res['relative_error']:.3f}" if res['relative_error'] is not None else "N/A"
        print(f"{res['name']:<20} | {res['true_k']:<10} | {pred_k_str:<12} | {error_str:<15} | {res['duration']:<15.2f}")
        if res['relative_error'] is not None: valid_errors.append(res['relative_error'])
        total_time += res['duration']
    print("-" * 80)
    if valid_errors:
        overall_are = np.mean(valid_errors)
        print(f"{'Overall Average Relative Error (ARE)':<48} | {overall_are:<15.3f} |")
    print(f"{'Total Time for All Datasets':<48} | {total_time:<15.2f} |")
    print("=" * 80)


def main():
    """ Main function to parse arguments and dispatch the analysis. """
    dataset_choices = ['echocardiogram','iris','seeds','wine','colon','prostate','miceprotein','abalone','cnae-9','vowel','synthetic_control']
    parser = argparse.ArgumentParser(description="Run clustering evaluation to find the optimal number of clusters (k).", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('dataset',type=str,nargs='?',default='seeds',choices=dataset_choices,help="Name of the dataset to use. Ignored if --run-all-datasets is set.")
    parser.add_argument('--run-all-datasets',action='store_true', default=True, help="Run the analysis on all available datasets in a batch.")
    args = parser.parse_args()

    script_start_dt = datetime.now()
    print("="*70); print(f"SCRIPT EXECUTION STARTED"); print(f"Start Time: {script_start_dt.strftime('%Y-%m-%d %H:%M:%S')}"); print("="*70)

    if args.run_all_datasets:
        summary_results = []
        for dataset in dataset_choices:
            print("\n" + "="*70); print(f"{'':<12}PROCESSING BATCH DATASET: {dataset.upper()}"); print("="*70)
            result = run_analysis_for_dataset(dataset_name=dataset)
            summary_results.append(result)
        print_summary_table(summary_results)
    else:
        run_analysis_for_dataset(dataset_name=args.dataset)

    script_end_dt = datetime.now()
    print(f"Script Finished at: {script_end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

if __name__ == "__main__":
    main()

