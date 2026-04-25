import time
import re
import ast
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from search.mcts import MCTSNode
import torch.nn.functional as F

class QwenBaselineVLA:
    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct", attn_implementation="sdpa"):
        print(f"Loading {model_id} as Baseline...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_implementation
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

        inputs = self.processor(text=formatted_prompt, images=images if images else None, return_tensors="pt").to(self.model.device)
        
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
        
    def mcts_generate(self, inputs, max_new_tokens=512, iterations=50):
        """
        Runs MCTS on the input image and prompt
        Returns the action with the highest number of visits
        """
        root = MCTSNode(state=inputs)

        for _ in range(iterations):
            node = root

            # select a child node
            while node.children:
                node = max(node.children.values(), key=lambda n: n.ucb_score(root.visits))

            start_time = time.time()

            # get candidate actions through diverse sampling
            actions = self.model.generate(
                **inputs,
                do_sample=True,
                temperateure=0.7,
                top_p = 0.9,
                num_return_sequences=5,
                max_new_tokens=max_new_tokens
            )

            latency = time.time() - start_time

            for action in actions:
                new_state = node.state + action
                node.children[action] = MCTSNode(state=new_state, parent=node)

                # evaluate the state of the child node
                verifier_score = self.self_evaluate_state(self, inputs, node, action)
                reward = node.calculate_reward(verifier_score, latency)

                node.children[action].value = reward
                node.children[action].visits = 1

                while node:
                    node.visits += 1
                    node.value += reward
                    node = node.parent
        
        return max(root.children.items(), key=lambda item: item[1].visits)[0]

    def self_evaluate_state(self, inputs, node, action):
        # define evaluation prompt
        eval_prompt = f""" Task: Given an action and the current state, score the state and action pair by
        outputting a single letter. Your choices are A, B, C, D, E. A represents a terrible choice,
        while E represents an excellent choice.
        State: {node.state}
        Action: {action}"""

        # calculate logits for the next token
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
            next_token_logits = outputs.logits[:, -1, :]
        
        # map letters to token ids
        choices = ['A', 'B', 'C', 'D', 'E']
        choice_ids = [self.processor.tokenizer.convert_tokens_to_ids(c) for c in choices]

        # extract probabilities of each choice
        choice_logits = next_token_logits[0, choice_ids]
        probs = F.softmax(choice_logits, dim=-1)

        # calculate weighted sum of choices to determine score
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
            return ast.literal_eval(traj_string.strip())
        except (ValueError, SyntaxError):
            return []
