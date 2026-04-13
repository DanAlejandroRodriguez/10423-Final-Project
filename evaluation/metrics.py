import numpy as np
import time
from typing import List, Dict, Optional

class DriveLMEvaluator:
    """
    Evaluator for tracking Vision-Language-Action (VLA) performance on DriveLM.
    Computes Meta Action IOU, Trajectory ADE (@3s and @6.4s), and Chain-of-Thought (CoT) latency.
    """
    def __init__(self):
        self.action_ious = []
        self.ade_3s_list = []
        self.ade_6_4s_list = []
        self.cot_times = []
        
    def evaluate_meta_actions(self, pred_actions: List[str], gt_actions: List[str]) -> float:
        """
        Calculates Intersection over Union (IOU) for Meta Actions.
        
        Args:
            pred_actions: List of predicted actions (e.g., ["decelerate", "yield to pedestrian"])
            gt_actions: List of ground-truth actions
            
        Returns:
            IOU score between 0.0 and 1.0 (Higher is better)
        """
        pred_set = set(pred_actions)
        gt_set = set(gt_actions)
        
        intersection = pred_set.intersection(gt_set)
        union = pred_set.union(gt_set)
        
        iou = len(intersection) / len(union) if union else 0.0
        self.action_ious.append(iou)
        
        return iou

    def evaluate_trajectory(self, pred_traj: np.ndarray, gt_traj: np.ndarray) -> Dict[str, float]:
        """
        Calculates Average Displacement Error (ADE) at 3s and 6.4s.
        
        Args:
            pred_traj: Predicted trajectory of shape (T, 2), coordinates in meters
            gt_traj: Ground truth trajectory of shape (T, 2), coordinates in meters
            
        Returns:
            Dictionary containing ADE @ 3s and ADE @ 6.4s (Lower is better)
        """
        # Ensure we're working with numpy arrays
        pred_traj = np.array(pred_traj)
        gt_traj = np.array(gt_traj)
        
        # Calculate Euclidean distance at each future time step
        distances = np.linalg.norm(pred_traj - gt_traj, axis=-1)
        
        # ADE @ 3s (6 waypoints at 2Hz)
        ade_3s = float(np.mean(distances[:6])) if len(distances) >= 6 else float(np.mean(distances))
        
        # ADE @ 6.4s (roughly 13 waypoints at 2Hz)
        num_wp_6_4s = min(13, len(distances))
        ade_6_4s = float(np.mean(distances[:num_wp_6_4s]))
        
        self.ade_3s_list.append(ade_3s)
        self.ade_6_4s_list.append(ade_6_4s)
        
        return {
            "ADE_3s": ade_3s,
            "ADE_6.4s": ade_6_4s
        }

    def measure_cot_time(self, start_time: float, end_time: float = None) -> float:
        """
        Records the Chain-of-Thought (CoT) inference latency in seconds.
        If end_time is not provided, it uses the current time.
        
        Args:
            start_time: Wall-clock time before model generation started
            end_time: Wall-clock time after model generation completed (optional)
            
        Returns:
            CoT time in seconds (Lower is better)
        """
        if end_time is None:
            end_time = time.time()
            
        cot_time = end_time - start_time
        self.cot_times.append(cot_time)
        return cot_time

    def summarize(self) -> Dict[str, float]:
        """
        Returns the averaged evaluation metrics across all evaluated samples.
        """
        summary = {
            "Meta Action (IOU)": float(np.mean(self.action_ious)) if self.action_ious else 0.0,
            "Trajectory (ADE @ 3s)": float(np.mean(self.ade_3s_list)) if self.ade_3s_list else 0.0,
            "Trajectory (ADE @ 6.4s)": float(np.mean(self.ade_6_4s_list)) if self.ade_6_4s_list else 0.0,
            "CoT Time (s)": float(np.mean(self.cot_times)) if self.cot_times else 0.0
        }
        
        # Print summary for convenience
        print("\n=== DriveLM Evaluation Summary ===")
        print(f"Meta Action (IOU) ↑ : {summary['Meta Action (IOU)']:.4f}")
        print(f"Trajectory (ADE @ 3s) ↓ : {summary['Trajectory (ADE @ 3s)']:.4f} m")
        print(f"Trajectory (ADE @ 6.4s) ↓ : {summary['Trajectory (ADE @ 6.4s)']:.4f} m")
        print(f"CoT Time (s) ↓ : {summary['CoT Time (s)']:.4f} s")
        print("==================================\n")
        
        return summary

# Optional functional API for single-shot evaluations
def calculate_meta_action_iou(pred_actions: List[str], gt_actions: List[str]) -> float:
    return DriveLMEvaluator().evaluate_meta_actions(pred_actions, gt_actions)

def calculate_ade_metrics(pred_traj: np.ndarray, gt_traj: np.ndarray) -> Dict[str, float]:
    return DriveLMEvaluator().evaluate_trajectory(pred_traj, gt_traj)
