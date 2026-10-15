"""Analyze dynamic ALINA ROI experiments without modifying or rerunning ALINA.
Run from repository root: python analysis/analyze_dynamic.py
Requires: pandas, numpy, matplotlib (pip install pandas matplotlib).
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
METHODS = ['hsv', 'hough', 'ransac', 'kmeans', 'gemini']
VIDEOS = ['vidd_1', 'vidd_2', 'vidd_3']

# Interpret the boolean values stored in the experiment CSV.
def truthy(v):
    return str(v).strip().lower() in {'true', '1', 'yes'}

# Read each saved pixel coordinate as an (x, y) pair.
def load_coords(path):
    if not path.exists():
        return None
    coordinates = set()
    for line_number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        columns = line.split()
        if len(columns) != 2:
            raise ValueError(f'{path}:{line_number} expected 2 columns, got {columns}')
        coordinates.add((int(columns[0]), int(columns[1])))
    return coordinates

# Count exact matches between predicted and reference coordinates.
def metric_counts(gt, pred):
    true_positives = len(gt & pred)
    return (true_positives, len(pred) - true_positives, len(gt) - true_positives)

# Save the calculated values in a CSV for later review.
def save_table(frame, path):
    frame.to_csv(path, index=False, float_format='%.4f')
    print(f'Saved {path}')

# Write the value above each bar so the graph is easier to read.
def bar_label(ax, rects, fmt='{:.1f}'):
    for bar in rects:
        bar_height = bar.get_height()
        if np.isfinite(bar_height):
            ax.annotate(fmt.format(bar_height), (bar.get_x() + bar.get_width() / 2, bar_height), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)

# Build the three summary graphs from the recorded experiment results.
def plot_figures(df, out):

    # Count successful detections for each ROI method.
    method_summaries = []
    for method in METHODS:
        method_rows = df[df.method == method]
        if method_rows.empty:
            continue
        completed_rows = method_rows[method_rows.status == 'ok']
        measured_rows = completed_rows[~completed_rows.cache_hit]
        method_summaries.append(dict(method=method, evaluated=len(method_rows), execution_errors=int((method_rows.status != 'ok').sum()), frames_with_lines=int(method_rows.detected.sum()), success_rate_pct=100 * method_rows.detected.mean(), mean_roi_ms=completed_rows.roi_ms.mean(), mean_alina_ms=completed_rows.alina_ms.mean(), mean_total_ms=completed_rows.total_ms.mean(), live_roi_ms=measured_rows.roi_ms.mean(), live_total_ms=measured_rows.total_ms.mean(), live_samples=len(measured_rows), cache_hits=int(completed_rows.cache_hit.sum())))
    summary_table = pd.DataFrame(method_summaries)
    save_table(summary_table, out / 'overall_summary.csv')
    if summary_table.empty:
        return
    figure, axes = plt.subplots(figsize=(9, 5))
    bars = axes.bar(summary_table.method, summary_table.success_rate_pct)
    bar_label(axes, bars, '{:.1f}%')
    axes.set_ylim(0, 110)
    axes.set_ylabel('Frames with ≥1 detected pixel (%)')
    axes.set_title('Figure 1 — Dynamic ROI detection success rate')
    for method_index, method_result in summary_table.reset_index().iterrows():
        axes.text(method_index, -9, f'n={int(method_result.evaluated)}', ha='center', fontsize=8)
    figure.subplots_adjust(bottom=0.18)
    figure.savefig(out / 'figure_1_detection_success.png', dpi=180, bbox_inches='tight')
    plt.close(figure)

    # Compare the measured time spent selecting the ROI and running ALINA.
    timing_rows = []
    for method in METHODS:
        method_rows = df[(df.method == method) & (df.status == 'ok') & ~df.cache_hit]
        if method_rows.empty:
            continue
        timing_rows.append(dict(method=method, n=len(method_rows), roi_seconds=method_rows.roi_ms.mean() / 1000, alina_seconds=method_rows.alina_ms.mean() / 1000, total_seconds=method_rows.total_ms.mean() / 1000))
    timing_table = pd.DataFrame(timing_rows)
    save_table(timing_table, out / 'processing_time_summary.csv')
    figure, axes = plt.subplots(figsize=(9, 5))
    bar_positions = np.arange(len(timing_table))
    bar_width = 0.38
    roi_bars = axes.bar(bar_positions - bar_width / 2, timing_table.roi_seconds, bar_width, label='ROI selection')
    alina_bars = axes.bar(bar_positions + bar_width / 2, timing_table.alina_seconds, bar_width, label='ALINA processing')
    axes.set_xticks(bar_positions, timing_table.method)
    axes.set_ylabel('Mean time per frame (seconds)')
    axes.set_title('Figure 2 — Processing times (uncached executions only)')
    axes.legend()
    figure.tight_layout()
    figure.savefig(out / 'figure_2_processing_time.png', dpi=180)
    plt.close(figure)

    # Calculate the detection rate separately for each video.
    video_summaries = []
    for method in METHODS:

        # Look for matching reference files in each video folder.
        for video_name in VIDEOS:
            method_rows = df[(df.method == method) & (df.video == video_name)]
            if len(method_rows):
                video_summaries.append(dict(method=method, video=video_name, evaluated=len(method_rows), with_lines=int(method_rows.detected.sum()), execution_errors=int((method_rows.status != 'ok').sum()), success_rate_pct=100 * method_rows.detected.mean()))
    video_table = pd.DataFrame(video_summaries)
    save_table(video_table, out / 'per_video_summary.csv')
    figure, axes = plt.subplots(figsize=(10, 5))
    bar_positions = np.arange(len(METHODS))
    bar_width = 0.25
    for video_index, video_name in enumerate(VIDEOS):
        video_rates = [video_table.loc[(video_table.method == m) & (video_table.video == video_name), 'success_rate_pct'].iloc[0] if ((video_table.method == m) & (video_table.video == video_name)).any() else np.nan for m in METHODS]
        axes.bar(bar_positions + (video_index - 1) * bar_width, video_rates, bar_width, label=video_name)
    axes.set_xticks(bar_positions, METHODS)
    axes.set_ylim(0, 105)
    axes.set_ylabel('Frames with ≥1 detected pixel (%)')
    axes.set_title('Figure 3 — Detection success by video')
    axes.legend()
    figure.tight_layout()
    figure.savefig(out / 'figure_3_per_video.png', dpi=180)
    plt.close(figure)
    print('NOTE: Figure 2 excludes Gemini cache hits; see processing_time_summary.csv for sample sizes.')

# Check the filename patterns used when saving ALINA coordinates.
def find_prediction(root, video, method, stem):
    method_folder = root / video / method
    possible_files = [method_folder / f'{stem}_coords.txt', method_folder / f'{stem}.txt', method_folder / 'coords' / f'{stem}.txt']
    return next((candidate_path for candidate_path in possible_files if candidate_path.is_file()), None)

# Keep the optional ground-truth comparison separate from the main graphs.
def ground_truth_eval(df, root, gtroot, out):
    evaluation_rows = []
    matched_frames = []
    missing_predictions = []

    # Look for matching reference files in each video folder.
    for video_name in VIDEOS:
        video_number = video_name.split('_')[-1]
        ground_truth_folder = gtroot / f'canny_textfiles_{video_number}'
        if not ground_truth_folder.is_dir():
            print(f'WARNING: ground-truth directory missing: {ground_truth_folder}')
            continue
        for frame_name in sorted(df.loc[df.video == video_name, 'frame'].unique()):
            frame_stem = Path(frame_name).stem
            ground_truth_file = ground_truth_folder / f'{frame_stem}.txt'
            if not ground_truth_file.exists():
                continue
            ground_truth_points = load_coords(ground_truth_file)
            matched_frames.append((video_name, frame_name))
            for method_name in METHODS:
                matching_record = df[(df.video == video_name) & (df.frame == frame_name) & (df.method == method_name)]
                if matching_record.empty:
                    continue
                micro_recall = matching_record.iloc[0]
                if micro_recall.status != 'ok':
                    missing_predictions.append(dict(video=video_name, frame=frame_name, method=method_name, reason='execution error'))
                    continue
                prediction_file = find_prediction(root, video_name, method_name, frame_stem)
                if prediction_file is None:
                    missing_predictions.append(dict(video=video_name, frame=frame_name, method=method_name, reason='prediction file missing'))
                    continue
                predicted_points = load_coords(prediction_file)
                true_positives, false_positives, false_negatives = metric_counts(ground_truth_points, predicted_points)
                precision_score = true_positives / (true_positives + false_positives) if true_positives + false_positives else 1.0 if not ground_truth_points else 0.0
                recall_score = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 1.0
                f1_score = 2 * precision_score * recall_score / (precision_score + recall_score) if precision_score + recall_score else 0.0
                evaluation_rows.append(dict(video=video_name, frame=frame_name, method=method_name, gt_pixels=len(ground_truth_points), pred_pixels=len(predicted_points), tp=true_positives, fp=false_positives, fn=false_negatives, precision=precision_score, recall=recall_score, f1=f1_score))
    save_table(pd.DataFrame(evaluation_rows, columns=['video', 'frame', 'method', 'gt_pixels', 'pred_pixels', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1']), out / 'cbem_per_frame.csv')
    save_table(pd.DataFrame(missing_predictions, columns=['video', 'frame', 'method', 'reason']), out / 'cbem_missing_predictions.csv')
    print(f'CBEM matches in evaluated frame IDs: {len(set(matched_frames))}; evaluated method/frame pairs: {len(evaluation_rows)}')
    if not evaluation_rows:
        print('No CBEM pairs evaluated. Check prediction naming and paths.')
        return
    evaluation_table = pd.DataFrame(evaluation_rows)
    method_totals = []
    for method_name, method_rows in evaluation_table.groupby('method'):
        true_positives, false_positives, false_negatives = method_rows[['tp', 'fp', 'fn']].sum()
        micro_precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
        micro_recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
        method_totals.append(dict(method=method_name, frames=len(method_rows), micro_precision=micro_precision, micro_recall=micro_recall, micro_f1=2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall else 0.0, mean_frame_precision=method_rows.precision.mean(), mean_frame_recall=method_rows.recall.mean(), mean_frame_f1=method_rows.f1.mean()))
    save_table(pd.DataFrame(method_totals), out / 'cbem_summary.csv')
    print('CBEM uses EXACT (x,y) coordinate matching; no pixel tolerance. This differs from ALINA repo evaluation, which averages separate X and Y coordinate metrics.')
    print('Verify CBEM and predictions use the same source frame and image resolution before interpreting scores.')

# Read the command-line options and set up the experiment folders.
def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--results', type=Path, default=Path('outputs/dynamic_experiments/results.csv'))
    argument_parser.add_argument('--experiment-root', type=Path, default=Path('outputs/dynamic_experiments'))
    argument_parser.add_argument('--gt-root', type=Path, default=Path('data/gt_alina_labels/canny_textfiles'))
    argument_parser.add_argument('--output', type=Path, default=Path('outputs/dynamic_analysis'))
    options = argument_parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    if not options.results.is_file():
        argument_parser.error(f'Missing results CSV: {options.results}')

    # Read the saved experiment table without rerunning ALINA.
    results_table = pd.read_csv(options.results, dtype={'frame': str}).fillna({'error': ''})
    required_columns = {'video', 'frame', 'method', 'status', 'roi_ms', 'alina_ms', 'total_ms', 'detected_pixels', 'had_lines', 'cache_hit'}
    if not required_columns.issubset(results_table.columns):
        argument_parser.error(f'Missing CSV columns: {sorted(required_columns - set(results_table.columns))}')
    results_table['frame'] = results_table.frame.map(lambda x: Path(x).name)
    for column_name in ['roi_ms', 'alina_ms', 'total_ms', 'detected_pixels']:
        results_table[column_name] = pd.to_numeric(results_table[column_name], errors='coerce')
    results_table['cache_hit'] = results_table.cache_hit.map(truthy)
    results_table['detected'] = (results_table.status == 'ok') & results_table.had_lines.map(truthy) & (results_table.detected_pixels > 0)
    duplicate_rows = results_table.duplicated(['video', 'frame', 'method'], keep=False)
    if duplicate_rows.any():
        argument_parser.error(f'Duplicate experiment records found: {int(duplicate_rows.sum())}; resolve before analysis')

    # Generate the three figures used in the report.
    plot_figures(results_table, options.output)
    ground_truth_eval(results_table, options.experiment_root, options.gt_root, options.output)
    print(f'\nAnalysis complete: {options.output.resolve()}')
if __name__ == '__main__':
    main()
