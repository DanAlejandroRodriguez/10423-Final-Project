import torch
import time
import re
from .baseline import QwenBaselineVLA
from .dag_scheduler import DagScheduler


class FastDriveVLA(QwenBaselineVLA):

    DRIVELM_COT_VERTICES = [
        "lighting", "road_condition", "weather", "junction_type", "road_type",
        "traffic_light", "traffic_sign",
        "lanes_enumeration", "lane_detail_0", "lane_detail_1", "lane_detail_2",
        "critical_objects_enumeration",
        "critical_object_0", "critical_object_1", "critical_object_2", "critical_object_3",
        "traffic_regulation_summary", "non_interactive_summary",
        "interactive_summary", "ego_behavior_summary",
    ]

    DRIVELM_COT_EDGES = [
        ("lanes_enumeration", "lane_detail_0"),
        ("lanes_enumeration", "lane_detail_1"),
        ("lanes_enumeration", "lane_detail_2"),
        ("critical_objects_enumeration", "critical_object_0"),
        ("critical_objects_enumeration", "critical_object_1"),
        ("critical_objects_enumeration", "critical_object_2"),
        ("critical_objects_enumeration", "critical_object_3"),
        ("traffic_light", "traffic_regulation_summary"),
        ("traffic_sign", "traffic_regulation_summary"),
        ("traffic_regulation_summary", "non_interactive_summary"),
        ("traffic_regulation_summary", "interactive_summary"),
        ("lane_detail_0", "non_interactive_summary"),
        ("lane_detail_1", "non_interactive_summary"),
        ("lane_detail_2", "non_interactive_summary"),
        ("critical_object_0", "interactive_summary"),
        ("critical_object_1", "interactive_summary"),
        ("critical_object_2", "interactive_summary"),
        ("critical_object_3", "interactive_summary"),
        ("non_interactive_summary", "ego_behavior_summary"),
        ("interactive_summary", "ego_behavior_summary"),
    ]

    DRIVELM_COT_MAX_LENGTHS = {
        "lighting": 8, "road_condition": 10, "weather": 8,
        "junction_type": 10, "road_type": 10,
        "traffic_light": 15, "traffic_sign": 20,
        "lanes_enumeration": 20,
        "lane_detail_0": 40, "lane_detail_1": 40, "lane_detail_2": 40,
        "critical_objects_enumeration": 30,
        "critical_object_0": 50, "critical_object_1": 50,
        "critical_object_2": 50, "critical_object_3": 50,
        "traffic_regulation_summary": 40,
        "non_interactive_summary": 50, "interactive_summary": 50,
        "ego_behavior_summary": 60,
    }

    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
        super().__init__(model_id=model_id, attn_implementation="sdpa")

    def build_dag_attention_mask(self, prefix_length, branch_lengths, ancestor_mask=None, padding_lengths=None):
        """
        Builds a 2D boolean causal attention mask for parallel CoT decoding.

        Per FastDriveCoT (Eq. 1): token in field B attends to field A only if A
        is an ancestor of B in the dependency graph. Fixed (prefix) tokens are
        visible to all following tokens regardless of dependencies. Padding
        tokens are invisible to every other token in the sequence.
        """
        total_len = prefix_length + sum(branch_lengths)
        mask = torch.zeros(total_len, total_len, dtype=torch.bool)

        mask[:prefix_length, :prefix_length] = torch.tril(
            torch.ones(prefix_length, prefix_length, dtype=torch.bool)
        )

        offsets = []
        offset = prefix_length
        for branch_len in branch_lengths:
            offsets.append(offset)
            offset += branch_len

        padding_lengths = padding_lengths or [0] * len(branch_lengths)

        for i, (branch_len, pad_len, start) in enumerate(zip(branch_lengths, padding_lengths, offsets)):
            real_len = branch_len - pad_len
            real_end = start + real_len

            mask[start:real_end, :prefix_length] = True

            if ancestor_mask is not None:
                for j, is_ancestor in enumerate(ancestor_mask[i]):
                    if is_ancestor:
                        anc_start = offsets[j]
                        anc_real_end = anc_start + branch_lengths[j] - padding_lengths[j]
                        mask[start:real_end, anc_start:anc_real_end] = True

            mask[start:real_end, start:real_end] = torch.tril(
                torch.ones(real_len, real_len, dtype=torch.bool)
            )

            pad_start = real_end
            pad_end = start + branch_len
            if pad_start < pad_end:
                mask[pad_start:pad_end, pad_start:pad_end] = torch.eye(
                    pad_end - pad_start, dtype=torch.bool
                )

        return mask

    def build_dag_position_ids(self, prefix_length, branch_lengths):
        """
        Builds position IDs for the DAG layout.

        All branches share the same position range starting at prefix_length,
        preventing RoPE from treating branches as a single long sequence.
        Returns position_ids of shape (1, total_seq_len).
        """
        prefix_pos = torch.arange(prefix_length)
        branch_pos = torch.cat([
            torch.arange(prefix_length, prefix_length + blen)
            for blen in branch_lengths
        ])
        return torch.cat([prefix_pos, branch_pos]).unsqueeze(0)

    def parallel_forward_pass(self, input_ids, branch_lengths, ancestor_mask=None, padding_lengths=None, position_ids=None):
        """
        Executes a single forward pass with the DAG attention mask.

        The DAG scheduler constructs input_ids by concatenating
        [prefix_tokens, branch_1_tokens, ..., branch_n_tokens].
        Returns logits of shape (1, total_seq_len, vocab_size).
        """
        prefix_length = input_ids.shape[1] - sum(branch_lengths)

        bool_mask = self.build_dag_attention_mask(
            prefix_length, branch_lengths, ancestor_mask, padding_lengths
        )
        bool_mask = bool_mask.to(device=self.model.device)

        dtype = self.model.dtype
        additive_mask = torch.zeros_like(bool_mask, dtype=dtype)
        additive_mask = additive_mask.masked_fill(~bool_mask, torch.finfo(dtype).min)
        additive_mask = additive_mask.unsqueeze(0).unsqueeze(0)

        if position_ids is None:
            position_ids = self.build_dag_position_ids(prefix_length, branch_lengths)
        position_ids = position_ids.to(device=self.model.device)
        input_ids = input_ids.to(device=self.model.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=additive_mask,
                position_ids=position_ids,
                use_cache=False,
            )

        return outputs.logits
    
    def build_ancestor_mask(self):
        """
        Builds both the ancestor mask and the related edges
        Returns the vertices, edges, and ancestor mask
        """
        node_to_idx = {name: i for i, name in enumerate(self.DRIVELM_COT_VERTICES)}

        num_source_nodes = len(self.DRIVELM_COT_VERTICES)
        ancestor_mask = torch.zeros(num_source_nodes, num_source_nodes, dtype=torch.bool)

        for (a, b) in self.DRIVELM_COT_EDGES:
            ancestor_mask[node_to_idx[b], node_to_idx[a]] = True
        ancestor_mask[[node_to_idx["non_interactive_summary"], node_to_idx["ego_behavior_summary"]], node_to_idx["lanes_enumeration"]] = True
        ancestor_mask[node_to_idx["ego_behavior_summary"], [node_to_idx["lane_detail_0"], node_to_idx["lane_detail_1"], node_to_idx["lane_detail_2"]]] = True
        ancestor_mask[[node_to_idx["interactive_summary"], node_to_idx["ego_behavior_summary"]], node_to_idx["critical_objects_enumeration"]] = True
        ancestor_mask[node_to_idx["ego_behavior_summary"], [node_to_idx["critical_object_0"], node_to_idx["critical_object_1"], node_to_idx["critical_object_2"], node_to_idx["critical_object_3"]]] = True
        ancestor_mask[[node_to_idx["non_interactive_summary"], node_to_idx["interactive_summary"], node_to_idx["ego_behavior_summary"]], [node_to_idx["traffic_light"], node_to_idx["traffic_sign"]]] = True
        
        return ancestor_mask
    
    def generate_trajectory(self, images, text_prompt, max_new_tokens=512):
        formatted_prompt = self.processor.apply_chat_template(
            text_prompt,
            tokenize=False,
            add_generation_prompt=False
        )

        inputs = self.processor(text=formatted_prompt, images=images if images else None, return_tensors="pt").to(self.model.device)

        ancestor_mask = self.build_ancestor_mask()

        start_time = time.time()

        dag_scheduler = DagScheduler(formatted_prompt, inputs, self, self.DRIVELM_COT_VERTICES, self.DRIVELM_COT_EDGES, self.DRIVELM_COT_MAX_LENGTHS, ancestor_mask)
        generated_tokens = dag_scheduler.run_parallel_decoding()

        latency = time.time() - start_time

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
