import time
import re
import ast
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from data.preprocess import PromptFormatter

class QwenBaselineVLA:
    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct", attn_implementation="sdpa",
                 max_pixels=360*420):
        print(f"Loading {model_id} as Baseline...")
        self.processor = AutoProcessor.from_pretrained(model_id, max_pixels=max_pixels)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_implementation
        )
        self.model.eval()

    def generate_trajectory(self, images, question, max_new_tokens=512):
        """
        The standard interface for all the models.
        Returns a dictionary containing the parsed components and latency.
        """
        messages = PromptFormatter.format(question=question, images=images)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        
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
            "model_type": "Qwen2.5VL_Autoregressive_Baseline",
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
