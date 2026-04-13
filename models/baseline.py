import time
import re
import ast
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

class GemmaBaselineVLA:
    def __init__(self, model_id="google/gemma-4-E2B-it"):
        print(f"Loading {model_id} as Baseline...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()

    def generate_trajectory(self, images, text_prompt, max_new_tokens=512):
        """
        The standard interface for all the models.
        Returns a dictionary containing the parsed components and latency.
        """
        formatted_prompt = self.processor.apply_chat_template(
            text_prompt, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = self.processor(text=formatted_prompt, images=images, return_tensors="pt").to(self.model.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        latency = time.time() - start_time
        
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        
        raw_text = self.processor.decode(generated_tokens, skip_special_tokens=True)
        
        return self._parse_output(raw_text, latency)

    def _parse_output(self, raw_text, latency):
        """Extracts the specific tags for evaluation."""
        cot_match = re.search(r'<cot>(.*?)</cot>', raw_text, re.DOTALL)
        action_match = re.search(r'<action>(.*?)</action>', raw_text, re.DOTALL)
        traj_match = re.search(r'<trajectory>(.*?)</trajectory>', raw_text, re.DOTALL)
        
        return {
            "model_type": "Autoregressive Baseline",
            "latency_seconds": latency,
            "raw_text": raw_text,
            "chain_of_thought": cot_match.group(1).strip() if cot_match else None,
            "meta_action": action_match.group(1).strip() if action_match else None,
            "trajectory": self._parse_coordinates(traj_match.group(1)) if traj_match else []
        }
        
    def _parse_coordinates(self, traj_string):
        """Safely converts string '[[x,y], ...]' into a Python list of floats."""
        try:
            return ast.literal_eval(traj_string.strip())
        except (ValueError, SyntaxError):
            return []
