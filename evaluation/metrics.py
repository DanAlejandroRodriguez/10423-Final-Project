"""
metrics.py
==========
Evaluation metrics for the DriveLM benchmark.

Computes four metrics:
  1. Meta Action IOU — set intersection-over-union between predicted and
     ground-truth meta-action labels.
  2. Trajectory ADE @ 3s — Average Displacement Error over the first 3 seconds
     (6 waypoints at 0.5 s nuScenes keyframe interval).
  3. Trajectory ADE @ 6.4s — Average Displacement Error over the full 6.4-second
     horizon (~13 waypoints).
  4. CoT Time (s) — chain-of-thought inference latency.

Usage
-----
  python -m evaluation.metrics                        # quick sanity-check
  python -m evaluation.metrics --results results.json # evaluate saved results
"""

import re
import json
import argparse
import numpy as np
from typing import List, Optional, Dict, Any


# Constants

# nuScenes annotated keyframe interval
_KEYFRAME_DT = 0.5  # seconds

# Waypoint counts for the two ADE horizons
_STEPS_3S = int(3.0 / _KEYFRAME_DT)      # 6
_STEPS_6_4S = int(6.4 / _KEYFRAME_DT)    # 12  (6.0 s), ceil → 13 for 6.5 s
_STEPS_6_4S = 13  # Use 13 to approximate 6.4 s (13 × 0.5 = 6.5 s ≈ 6.4 s)

# Canonical action vocabulary (matches PromptFormatter.SYSTEM_PROMPT)
VALID_ACTIONS = {
    "STOP", "YIELD", "ACCELERATE", "DECELERATE",
    "TURN_LEFT", "TURN_RIGHT", "LANE_CHANGE",
}


# Meta-Action IOU

def _normalise_actions(raw: str) -> set:
    """Extract a *set* of canonical action labels from free-form text.

    Handles comma / space / semicolon separated lists and is
    case-insensitive.  Unknown tokens are silently dropped.
    """
    if not raw:
        return set()
    # Uppercase + split on common delimiters
    tokens = re.split(r"[,;\s/|]+", raw.strip().upper())
    # Reassemble multi-word actions (e.g. "TURN LEFT" → "TURN_LEFT")
    reassembled = set()
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        # Check for two-word actions
        if i + 1 < len(tokens):
            combined = f"{tok}_{tokens[i + 1]}"
            if combined in VALID_ACTIONS:
                reassembled.add(combined)
                i += 2
                continue
        if tok in VALID_ACTIONS:
            reassembled.add(tok)
        i += 1
    return reassembled


def meta_action_iou(pred_actions: str, gt_actions: str) -> float:
    """Set-based Intersection-over-Union for meta-action labels.

    Parameters
    ----------
    pred_actions : str
        Raw predicted action string (e.g. "ACCELERATE, TURN_LEFT").
    gt_actions : str
        Ground-truth action string.

    Returns
    -------
    float
        IoU in [0, 1].  Returns 0.0 when both sets are empty.
    """
    pred_set = _normalise_actions(pred_actions)
    gt_set = _normalise_actions(gt_actions)

    if not pred_set and not gt_set:
        return 0.0

    intersection = pred_set & gt_set
    union = pred_set | gt_set
    return len(intersection) / len(union)


# 2 / 3.  Trajectory ADE
def trajectory_ade(
    pred: List[List[float]],
    gt: List[List[float]],
    max_steps: Optional[int] = None,
) -> float:
    """Average Displacement Error between two trajectories.

    Both ``pred`` and ``gt`` are lists of [x, y] waypoints.  If
    ``max_steps`` is given only the first *max_steps* common waypoints
    are used; otherwise the full overlap length is used.

    Parameters
    ----------
    pred : list[list[float]]
        Predicted waypoints [[x1,y1], [x2,y2], …].
    gt : list[list[float]]
        Ground-truth waypoints.
    max_steps : int, optional
        Truncate both trajectories to at most this many waypoints.

    Returns
    -------
    float
        ADE in the same coordinate units as the trajectory.
        Returns ``float('inf')`` if there are no overlapping waypoints.
    """
    pred_arr = np.asarray(pred, dtype=np.float64)
    gt_arr = np.asarray(gt, dtype=np.float64)

    if pred_arr.ndim != 2 or gt_arr.ndim != 2:
        return float("inf")

    n = min(len(pred_arr), len(gt_arr))
    if max_steps is not None:
        n = min(n, max_steps)
    if n == 0:
        return float("inf")

    pred_arr = pred_arr[:n]
    gt_arr = gt_arr[:n]

    displacements = np.linalg.norm(pred_arr - gt_arr, axis=1)
    return float(np.mean(displacements))


def ade_3s(pred: List[List[float]], gt: List[List[float]]) -> float:
    """ADE over a 3-second horizon (6 waypoints at 0.5 s intervals)."""
    return trajectory_ade(pred, gt, max_steps=_STEPS_3S)


def ade_6_4s(pred: List[List[float]], gt: List[List[float]]) -> float:
    """ADE over a 6.4-second horizon (≈13 waypoints at 0.5 s intervals)."""
    return trajectory_ade(pred, gt, max_steps=_STEPS_6_4S)


# 4.  Just a note that we need CoT latency  
# (no computation: just a pass-through)
def cot_time(latency_seconds: float) -> float:
    """Identity wrapper so every metric has a function signature.

    Parameters
    ----------
    latency_seconds : float
        Wall-clock time for chain-of-thought generation, as reported
        by the model's ``generate_trajectory`` method.

    Returns
    -------
    float
        Same value, in seconds.
    """
    return float(latency_seconds)


# Aggregate evaluator
def _extract_gt_action(graph: dict) -> str:
    """Pull the ground-truth meta-action from a DriveLM frame graph.

    The DriveLM v1.1 JSON stores QAs under ``QA → behavior`` or
    ``QA → planning``.  We look for answers that contain one of the
    canonical action tokens.
    """
    qa_dict = graph.get("QA", {})
    if not isinstance(qa_dict, dict):
        return ""

    candidates = []
    for task_key in ("behavior", "planning"):
        for qa in qa_dict.get(task_key, []):
            answer = qa.get("A", "")
            normalised = _normalise_actions(answer)
            if normalised:
                candidates.extend(normalised)

    return ", ".join(sorted(candidates))


class DriveLMEvaluator:
    """Collects per-sample results and computes aggregate metrics.

    Typical usage::

        evaluator = DriveLMEvaluator()
        for sample in dataset:
            result = model.generate_trajectory(sample["images"], sample["question"])
            evaluator.add(result, sample["gt_trajectory"], sample["answer"])
        print(evaluator.summarise())
    """

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def add(
        self,
        model_output: Dict[str, Any],
        gt_trajectory: List[List[float]],
        gt_graph: dict,
    ):
        """Register a single sample's results.

        Parameters
        ----------
        model_output : dict
            Return value of ``QwenBaselineVLA.generate_trajectory`` with keys
            ``meta_action``, ``trajectory``, ``latency_seconds``.
        gt_trajectory : list[list[float]]
            Ground-truth ego trajectory from the dataset.
        gt_graph : dict
            Raw DriveLM frame-info dict (``scene_data["graph"]``), used to
            extract the ground-truth meta-action.
        """
        pred_traj = model_output.get("trajectory", [])
        pred_action = model_output.get("meta_action", "")
        latency = model_output.get("latency_seconds", 0.0)

        gt_action = _extract_gt_action(gt_graph)

        self.records.append({
            "meta_action_iou": meta_action_iou(pred_action, gt_action),
            "ade_3s": ade_3s(pred_traj, gt_trajectory),
            "ade_6_4s": ade_6_4s(pred_traj, gt_trajectory),
            "cot_time_s": cot_time(latency),
            # Keep raw data for optional per-sample analysis
            "raw_text": model_output.get("raw_text", ""),
            "pred_action": pred_action,
            "gt_action": gt_action,
            "pred_traj_len": len(pred_traj),
            "gt_traj_len": len(gt_trajectory),
        })

    def summarise(self) -> Dict[str, float]:
        """Return aggregate metrics over all added samples.

        Returns
        -------
        dict
            Keys: ``meta_action_iou``, ``ade_3s``, ``ade_6_4s``,
            ``cot_time_s``, ``n_samples``, ``n_valid_traj_3s``,
            ``n_valid_traj_6_4s``.
        """
        if not self.records:
            return {
                "meta_action_iou": 0.0,
                "ade_3s": float("inf"),
                "ade_6_4s": float("inf"),
                "cot_time_s": 0.0,
                "n_samples": 0,
                "n_valid_traj_3s": 0,
                "n_valid_traj_6_4s": 0,
            }

        ious = [r["meta_action_iou"] for r in self.records]
        cots = [r["cot_time_s"] for r in self.records]

        # Filter out inf ADE values (i.e. samples where pred or gt was empty)
        ades_3s = [r["ade_3s"] for r in self.records
                   if np.isfinite(r["ade_3s"])]
        ades_6_4s = [r["ade_6_4s"] for r in self.records
                     if np.isfinite(r["ade_6_4s"])]

        return {
            "meta_action_iou": float(np.mean(ious)),
            "ade_3s": float(np.mean(ades_3s)) if ades_3s else float("inf"),
            "ade_6_4s": float(np.mean(ades_6_4s)) if ades_6_4s else float("inf"),
            "cot_time_s": float(np.mean(cots)),
            "n_samples": len(self.records),
            "n_valid_traj_3s": len(ades_3s),
            "n_valid_traj_6_4s": len(ades_6_4s),
        }

    def to_json(self, path: str):
        """Dump per-sample records to a JSON file for later analysis."""
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2, default=str)



def _print_summary(summary: Dict[str, float]):
    """Pretty-print the evaluation summary."""
    print("\n" + "=" * 55)
    print("  DriveLM Evaluation Results")
    print("=" * 55)
    print(f"  Samples evaluated      : {summary['n_samples']}")
    print(f"  -------------------------------------------------")
    print(f"  Meta Action IOU        : {summary['meta_action_iou']:.4f}")
    print(f"  Trajectory ADE @ 3s    : {summary['ade_3s']:.4f} m")
    print(f"    (valid trajectories) : {summary['n_valid_traj_3s']}")
    print(f"  Trajectory ADE @ 6.4s  : {summary['ade_6_4s']:.4f} m")
    print(f"    (valid trajectories) : {summary['n_valid_traj_6_4s']}")
    print(f"  CoT Time (mean)        : {summary['cot_time_s']:.3f} s")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DriveLM model predictions."
    )
    parser.add_argument(
        "--results", type=str, default=None,
        help="Path to a JSON file with per-sample records "
             "(as produced by DriveLMEvaluator.to_json)."
    )
    args = parser.parse_args()

    if args.results:
        with open(args.results) as f:
            records = json.load(f)
        evaluator = DriveLMEvaluator()
        evaluator.records = records
        summary = evaluator.summarise()
        _print_summary(summary)
    else:
        # sanity check with synthetic data
        print("Running sanity check with synthetic data…")

        pred_traj = [[1.0, 0.5], [2.1, 1.0], [3.0, 1.5],
                     [4.2, 2.0], [5.0, 2.5], [6.1, 3.0]]
        gt_traj =   [[1.0, 0.5], [2.0, 1.0], [3.0, 1.5],
                     [4.0, 2.0], [5.0, 2.5], [6.0, 3.0]]

        evaluator = DriveLMEvaluator()

        # Simulate model output
        model_output = {
            "meta_action": "ACCELERATE",
            "trajectory": pred_traj,
            "latency_seconds": 1.23,
        }
        gt_graph = {
            "QA": {
                "behavior": [
                    {"Q": "What should the ego car do?",
                     "A": "ACCELERATE, TURN_LEFT"}
                ]
            }
        }

        evaluator.add(model_output, gt_traj, gt_graph)

        # Second sample — exact match
        model_output_2 = {
            "meta_action": "STOP",
            "trajectory": gt_traj,
            "latency_seconds": 0.87,
        }
        gt_graph_2 = {
            "QA": {
                "behavior": [
                    {"Q": "What should the ego car do?",
                     "A": "STOP"}
                ]
            }
        }
        evaluator.add(model_output_2, gt_traj, gt_graph_2)

        summary = evaluator.summarise()
        _print_summary(summary)

        # Verify individual metric functions
        print("Individual metric checks:")
        print(f"  meta_action_iou('ACCELERATE', 'ACCELERATE, TURN_LEFT') "
              f"= {meta_action_iou('ACCELERATE', 'ACCELERATE, TURN_LEFT'):.4f}")
        print(f"  meta_action_iou('STOP', 'STOP') "
              f"= {meta_action_iou('STOP', 'STOP'):.4f}")
        print(f"  ade_3s(pred, gt) = {ade_3s(pred_traj, gt_traj):.4f} m")
        print(f"  ade_6_4s(pred, gt) = {ade_6_4s(pred_traj, gt_traj):.4f} m")


if __name__ == "__main__":
    main()
