import argparse
import re
import numpy as np
from tqdm import tqdm

from models.baseline import GemmaBaselineVLA
from data.drivelm_dataset import DriveLMDataset
from data.preprocess import PromptFormatter
from evaluation.metrics import DriveLMEvaluator

def parse_gt_action(text):
    """
    Simple parser to extract ground-truth action from text, 
    assuming it might be wrapped in <action> tags or just plain text.
    """
    if not isinstance(text, str):
        return []
        
    match = re.search(r'<action>(.*?)</action>', text, re.DOTALL)
    if match:
        # split by comma if multiple actions exist
        actions = [a.strip() for a in match.group(1).split(',')]
        return actions
    # fallback to the whole text if tags aren't present
    return [text.strip()]

def main():
    parser = argparse.ArgumentParser(description="Evaluate a VLA model on DriveLM Dataset")
    parser.add_argument("-model", "--model", type=str, required=True, 
                        help="HuggingFace model ID or 'baseline' to evaluate.")
    parser.add_argument("--data_root", type=str, default="./dataset/images", 
                        help="Path to DriveLM images root")
    parser.add_argument("--ann_file", type=str, default="./dataset/val.json", 
                        help="Path to DriveLM val annotations")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (for debugging)")
    
    args = parser.parse_args()

    # 1. Load Dataset
    print(f"Loading dataset from: {args.ann_file} ...")
    dataset = DriveLMDataset(data_root=args.data_root, ann_file=args.ann_file)
    print(f"Loaded {len(dataset)} samples.")

    if len(dataset) == 0:
        print("Dataset is empty. Exiting benchmark.")
        return

    # 2. Load Model
    # Determine the model ID to load. If 'baseline' is passed, default to gemma-4-E2B-it.
    # Otherwise, try to load whatever identifier is passed.
    model_id = "google/gemma-4-E2B-it" if args.model.lower() == "baseline" else args.model
    model = GemmaBaselineVLA(model_id=model_id)
    
    # 3. Setup Evaluator
    evaluator = DriveLMEvaluator()
    
    # Allow truncating for quick dry-runs
    num_samples = len(dataset)
    if args.max_samples is not None:
        num_samples = min(num_samples, args.max_samples)
    
    print(f"\nStarting Evaluation Loop over {num_samples} samples...")
    for idx in tqdm(range(num_samples)):
        images, question, gt_answer, gt_traj = dataset[idx]
        
        # Format the prompt using the established data contract
        text_prompt = PromptFormatter.format(question)
        
        # Process visual and textual input to generate trajectory and actions
        # Using a fallback to list of PIL images if tensors cause issues with transformers processor
        out = model.generate_trajectory(images, text_prompt)
        
        # === 1. Evaluate Meta Action IOU ===
        pred_action_str = out.get("meta_action", "")
        pred_actions = [a.strip() for a in pred_action_str.split(',')] if pred_action_str else []
        gt_actions = parse_gt_action(gt_answer)
        evaluator.evaluate_meta_actions(pred_actions, gt_actions)
        
        # === 2. Evaluate Trajectory ADE ===
        pred_traj = out.get("trajectory", [])
        if not pred_traj or len(pred_traj) == 0:
            # Fallback zero-array if the model hallucinates or outputs malformed numbers
            pred_traj = [[0.0, 0.0] for _ in range(13)] 
            
        # Make sure predicted length roughly matches expected padding, pass slices to evaluator
        evaluator.evaluate_trajectory(np.array(pred_traj), gt_traj.numpy())
        
        # === 3. Evaluate CoT Time (Latency) ===
        # GemmaBaseline tracks and measures total latency per invocation natively
        evaluator.measure_cot_time(0.0, out.get("latency_seconds", 0.0))
        
    print("\nBenchmark Evaluation Complete.")
    
    # Print out standard summary layout using the DriveLMEvaluator instance
    evaluator.summarize()

if __name__ == "__main__":
    main()
