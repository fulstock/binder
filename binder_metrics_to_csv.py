#!/usr/bin/env python3
"""
Convert Binder NER metrics JSON to CSV format.

Binder metrics have three evaluation modes: span, start, end
Each mode contains per-class precision, recall, f1 scores.

Usage:
    python binder_metrics_to_csv.py predict_metrics.json                 # stdout
    python binder_metrics_to_csv.py predict_metrics.json -o metrics.csv  # file
    python binder_metrics_to_csv.py predict_metrics.json --mode span     # span only
    python binder_metrics_to_csv.py predict_metrics.json --all-modes     # all modes
"""

import json
import csv
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_metrics(file_path: str) -> Dict[str, Any]:
    """Load metrics from JSON file. Handles JSONL (multiple runs) by taking the last entry."""
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            pass
    # Fall back to JSONL: take the last complete JSON object
    with open(file_path, 'r', encoding='utf-8') as f:
        last = None
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
        if last is None:
            raise ValueError(f"No valid JSON found in {file_path}")
        return last


def metrics_to_csv_single_mode(
    metrics: Dict[str, Any],
    mode: str = "span"
) -> List[List[str]]:
    """Convert single mode metrics to CSV rows."""
    rows = []

    mode_data = metrics.get(mode, {})
    if not mode_data:
        print(f"Warning: Mode '{mode}' not found in metrics", file=sys.stderr)
        return rows

    # Header
    header = ["entity_type", "precision", "recall", "f1"]
    rows.append(header)

    # Per-class metrics (excluding 'all')
    entity_types = [k for k in mode_data.keys() if k != "all"]
    for entity_type in sorted(entity_types):
        data = mode_data[entity_type]
        row = [
            entity_type,
            f"{data.get('precision', 0):.4f}",
            f"{data.get('recall', 0):.4f}",
            f"{data.get('f1', 0):.4f}"
        ]
        rows.append(row)

    # Macro averages
    if entity_types:
        macro_p = sum(mode_data[et].get('precision', 0) for et in entity_types) / len(entity_types)
        macro_r = sum(mode_data[et].get('recall', 0) for et in entity_types) / len(entity_types)
        macro_f1 = sum(mode_data[et].get('f1', 0) for et in entity_types) / len(entity_types)
        rows.append([
            "ALL (macro)",
            f"{macro_p:.4f}",
            f"{macro_r:.4f}",
            f"{macro_f1:.4f}"
        ])

    # Overall metrics
    if "all" in mode_data:
        overall = mode_data["all"]
        rows.append([
            "ALL (micro)",
            f"{overall.get('precision', 0):.4f}",
            f"{overall.get('recall', 0):.4f}",
            f"{overall.get('f1', 0):.4f}"
        ])

    return rows


def metrics_to_csv_all_modes(metrics: Dict[str, Any]) -> List[List[str]]:
    """Convert all modes (span, start, end) to CSV with columns for each."""
    rows = []

    modes = ["span", "start", "end"]
    available_modes = [m for m in modes if m in metrics]

    if not available_modes:
        print("Warning: No valid modes found in metrics", file=sys.stderr)
        return rows

    # Header
    header = ["entity_type"]
    for mode in available_modes:
        header.extend([f"{mode}_P", f"{mode}_R", f"{mode}_F1"])
    rows.append(header)

    # Collect all entity types
    all_entity_types = set()
    for mode in available_modes:
        all_entity_types.update(k for k in metrics[mode].keys() if k != "all")

    # Per-class metrics
    for entity_type in sorted(all_entity_types):
        row = [entity_type]
        for mode in available_modes:
            data = metrics[mode].get(entity_type, {})
            row.extend([
                f"{data.get('precision', 0):.4f}",
                f"{data.get('recall', 0):.4f}",
                f"{data.get('f1', 0):.4f}"
            ])
        rows.append(row)

    # Macro averages
    if all_entity_types:
        macro_row = ["ALL (macro)"]
        for mode in available_modes:
            ets = [k for k in metrics[mode].keys() if k != "all"]
            macro_p = sum(metrics[mode][et].get('precision', 0) for et in ets) / len(ets) if ets else 0
            macro_r = sum(metrics[mode][et].get('recall', 0) for et in ets) / len(ets) if ets else 0
            macro_f1 = sum(metrics[mode][et].get('f1', 0) for et in ets) / len(ets) if ets else 0
            macro_row.extend([f"{macro_p:.4f}", f"{macro_r:.4f}", f"{macro_f1:.4f}"])
        rows.append(macro_row)

    # Overall metrics
    overall_row = ["ALL (micro)"]
    for mode in available_modes:
        overall = metrics[mode].get("all", {})
        overall_row.extend([
            f"{overall.get('precision', 0):.4f}",
            f"{overall.get('recall', 0):.4f}",
            f"{overall.get('f1', 0):.4f}"
        ])
    rows.append(overall_row)

    return rows


def metrics_to_csv_comparison(
    metrics_files: List[str],
    mode: str = "span"
) -> List[List[str]]:
    """Compare multiple Binder metrics files."""
    all_metrics = {}
    all_entity_types = set()

    for file_path in metrics_files:
        model_name = Path(file_path).stem
        metrics = load_metrics(file_path)
        mode_data = metrics.get(mode, metrics)  # fallback if no mode structure
        all_metrics[model_name] = mode_data
        all_entity_types.update(k for k in mode_data.keys() if k != "all")

    model_names = list(all_metrics.keys())
    rows = []

    # Header
    header = ["entity_type"]
    for model in model_names:
        header.extend([f"{model}_P", f"{model}_R", f"{model}_F1"])
    rows.append(header)

    # Per-class
    for entity_type in sorted(all_entity_types):
        row = [entity_type]
        for model in model_names:
            data = all_metrics[model].get(entity_type, {})
            row.extend([
                f"{data.get('precision', 0):.4f}",
                f"{data.get('recall', 0):.4f}",
                f"{data.get('f1', 0):.4f}"
            ])
        rows.append(row)

    # Macro averages
    if all_entity_types:
        macro_row = ["ALL (macro)"]
        for model in model_names:
            ets = [k for k in all_metrics[model].keys() if k != "all"]
            macro_p = sum(all_metrics[model][et].get('precision', 0) for et in ets) / len(ets) if ets else 0
            macro_r = sum(all_metrics[model][et].get('recall', 0) for et in ets) / len(ets) if ets else 0
            macro_f1 = sum(all_metrics[model][et].get('f1', 0) for et in ets) / len(ets) if ets else 0
            macro_row.extend([f"{macro_p:.4f}", f"{macro_r:.4f}", f"{macro_f1:.4f}"])
        rows.append(macro_row)

    # Overall
    overall_row = ["ALL (micro)"]
    for model in model_names:
        overall = all_metrics[model].get("all", {})
        overall_row.extend([
            f"{overall.get('precision', 0):.4f}",
            f"{overall.get('recall', 0):.4f}",
            f"{overall.get('f1', 0):.4f}"
        ])
    rows.append(overall_row)

    return rows


def write_csv(rows: List[List[str]], output_file: Optional[str] = None):
    """Write rows to CSV file or stdout."""
    if output_file:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"CSV written to: {output_file}")
    else:
        writer = csv.writer(sys.stdout)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Binder NER metrics JSON to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file, span mode (default)
  python binder_metrics_to_csv.py predict_metrics.json

  # Single file, all modes in columns
  python binder_metrics_to_csv.py predict_metrics.json --all-modes

  # Specific mode
  python binder_metrics_to_csv.py predict_metrics.json --mode start

  # Output to file
  python binder_metrics_to_csv.py predict_metrics.json -o metrics.csv

  # Compare multiple files
  python binder_metrics_to_csv.py metrics1.json metrics2.json -o compare.csv
        """
    )
    parser.add_argument(
        'input_files',
        nargs='+',
        help='Input Binder metrics JSON file(s)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output CSV file (default: stdout)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='span',
        choices=['span', 'start', 'end'],
        help='Evaluation mode to extract (default: span)'
    )
    parser.add_argument(
        '--all-modes',
        action='store_true',
        help='Include all modes (span, start, end) as columns'
    )

    args = parser.parse_args()

    if len(args.input_files) == 1:
        metrics = load_metrics(args.input_files[0])

        if args.all_modes:
            rows = metrics_to_csv_all_modes(metrics)
        else:
            rows = metrics_to_csv_single_mode(metrics, args.mode)
    else:
        # Comparison mode
        rows = metrics_to_csv_comparison(args.input_files, args.mode)

    write_csv(rows, args.output)


if __name__ == "__main__":
    main()
