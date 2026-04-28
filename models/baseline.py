import time
import re
import ast
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch.nn.functional as F

from data.preprocess import PromptFormatter
from search.mcts import MCTSNode

class QwenBaselineVLA:
    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct", attn_implementation="sdpa",
                 max_pixels=360*420):
        print(f"Loading {model_id} as Baseline...")
        self.processor = AutoProcessor.from_pretrained(model_id, max_pixels=max_pixels)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            # torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation=attn_implementation
        )
        self.model.eval()

    def generate_trajectory(self, images, question, max_new_tokens=200):
        """
        The standard interface for all the models.
        Accepts pre-formatted messages from PromptFormatter.format().
        Returns a dictionary containing the parsed components and latency.
        """
        driving_question = (
            "What should the ego vehicle do next? "
            "Be concise. Describe what you see briefly, then give the action.\n\n"
            "Respond with:\n"
            "<cot> brief reasoning </cot>\n"
            "<action> STOP | YIELD | ACCELERATE | DECELERATE | TURN_LEFT | TURN_RIGHT | LANE_CHANGE </action>"
        )
        messages = PromptFormatter.format(question=driving_question, images=images)

        text = self.processor.apply_chat_template(
            text_prompt, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(text_prompt)
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

    def mcts_generate(self, images, text_prompt, max_new_tokens=512, iterations=5):
        """
        Runs MCTS on the input image and prompt.
        Accepts the same (images, text_prompt) interface as generate_trajectory.
        Returns a parsed result dictionary.
        """
        # Input processing — same as generate_trajectory
        text = self.processor.apply_chat_template(
            text_prompt, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(text_prompt)
        inputs = self.processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # MCTS search
        root = MCTSNode(state=inputs)
        prefix_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            prefix_out = self.model(**inputs, use_cache=True)
        prefix_kv = prefix_out.past_key_values

        start_time = time.time()

        for _ in range(iterations):
            node = root

            while node.children:
                node = max(node.children.values(), key=lambda n: n.ucb_score(root.visits))

            node_new_ids = node.state["input_ids"][:, prefix_length:].to(self.model.device)
            if node_new_ids.shape[1] > 0:
                with torch.no_grad():
                    node_kv = self.model(
                        input_ids=node_new_ids,
                        past_key_values=prefix_kv,
                        use_cache=True,
                    ).past_key_values
            else:
                node_kv = prefix_kv

            # Manual sampling loop (replaces model.generate which doesn't support
            # externally managed past_key_values in newer transformers)
            num_candidates = 5
            actions = []
            for _ in range(num_candidates):
                generated_ids = []
                kv = node_kv
                # Start with the last token from the prefix as the seed
                cur_input = inputs["input_ids"][:, -1:].to(self.model.device)
                for _step in range(max_new_tokens):
                    with torch.no_grad():
                        out = self.model(input_ids=cur_input, past_key_values=kv, use_cache=True)
                    kv = out.past_key_values
                    logits = out.logits[:, -1, :] / 0.7  # temperature
                    # Top-p (nucleus) sampling
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    probs = torch.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(probs, dim=-1)
                    mask = cumulative_probs - probs > 0.9
                    sorted_logits[mask] = float('-inf')
                    probs = torch.softmax(sorted_logits, dim=-1)
                    next_idx = torch.multinomial(probs, num_samples=1)
                    next_token = sorted_indices.gather(-1, next_idx)
                    generated_ids.append(next_token.item())
                    cur_input = next_token
                    # Stop on EOS
                    if next_token.item() == self.processor.tokenizer.eos_token_id:
                        break
                actions.append(torch.tensor(generated_ids, device=self.model.device))

            best_action = max(actions, key=lambda a: self.self_evaluate_state(inputs, node, a))
            action_key = tuple(best_action.tolist())
            new_input_ids = torch.cat([node.state["input_ids"], best_action.unsqueeze(0)], dim=1)
            new_state = dict(node.state)
            new_state["input_ids"] = new_input_ids
            node.children[action_key] = MCTSNode(state=new_state, parent=node)

            verifier_score = self.self_evaluate_state(inputs, node, best_action)
            iter_latency = time.time() - start_time
            reward = node.calculate_reward(verifier_score, iter_latency)

            node.children[action_key].value = reward
            node.children[action_key].visits = 1

            backprop_node = node
            while backprop_node:
                backprop_node.visits += 1
                backprop_node.value += reward
                backprop_node = backprop_node.parent

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        latency = time.time() - start_time

        # Decode the best path
        best_tokens = max(root.children.items(), key=lambda item: item[1].visits)[0]
        raw_text = self.processor.decode(torch.tensor(best_tokens), skip_special_tokens=True)

        return self._parse_output(raw_text, latency)

    def self_evaluate_state(self, inputs, node, action):
        eval_prompt = f""" Task: Given an action and the current state, score the state and action pair by
        outputting a single letter. Your choices are A, B, C, D, E. A represents a terrible choice,
        while E represents an excellent choice.
        State: {node.state}
        Action: {action}"""

        eval_inputs = self.processor(text=eval_prompt, return_tensors="pt")
        eval_inputs = eval_inputs.to(self.model.device) if hasattr(eval_inputs, "to") else eval_inputs
        with torch.no_grad():
            outputs = self.model(**eval_inputs)
            next_token_logits = outputs.logits[:, -1, :]

        choices = ['A', 'B', 'C', 'D', 'E']
        choice_ids = [self.processor.tokenizer.encode(c, add_special_tokens=False)[0] for c in choices]
        choice_logits = next_token_logits[0, choice_ids]
        probs = F.softmax(choice_logits, dim=-1)

        weights = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]).to(self.model.device)
        score = torch.sum(probs * weights).item()

        return score

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
            traj = ast.literal_eval(traj_string.strip())
            return traj[:13]
        except (ValueError, SyntaxError):
            return []
