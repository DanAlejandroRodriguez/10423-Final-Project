import torch
import time
import re
import ast
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

    # Field name template appended to prefix so the model knows what to generate
    # per Section 3.1: "field name: field content" one entry per line
    COT_TEMPLATE = (
        "Describe the driving scene using these fields:\n"
        "lighting: \n"
        "road_condition: \n"
        "weather: \n"
        "junction_type: \n"
        "road_type: \n"
        "traffic_light: \n"
        "traffic_sign: \n"
        "lanes_enumeration: \n"
        "lane_detail_0: \n"
        "lane_detail_1: \n"
        "lane_detail_2: \n"
        "critical_objects_enumeration: \n"
        "critical_object_0: \n"
        "critical_object_1: \n"
        "critical_object_2: \n"
        "critical_object_3: \n"
        "traffic_regulation_summary: \n"
        "non_interactive_summary: \n"
        "interactive_summary: \n"
        "ego_behavior_summary: \n"
    )

    # Multi-shot example so the model follows the format without fine-tuning
    COT_EXAMPLE = (
        "Example output:\n"
        "lighting: bright daylight\n"
        "road_condition: dry asphalt\n"
        "weather: clear\n"
        "junction_type: T-intersection\n"
        "road_type: urban road\n"
        "traffic_light: green\n"
        "traffic_sign: speed limit 30\n"
        "lanes_enumeration: two lanes ahead\n"
        "lane_detail_0: ego lane clear\n"
        "lane_detail_1: adjacent lane has slow vehicle\n"
        "lane_detail_2: N/A\n"
        "critical_objects_enumeration: one pedestrian, one car\n"
        "critical_object_0: pedestrian on sidewalk, not crossing\n"
        "critical_object_1: car ahead, moving slowly\n"
        "critical_object_2: N/A\n"
        "critical_object_3: N/A\n"
        "traffic_regulation_summary: green light, 30 km/h limit\n"
        "non_interactive_summary: road is clear, no obstacles in path\n"
        "interactive_summary: slow car ahead requires attention\n"
        "ego_behavior_summary: maintain speed, prepare to decelerate\n"
        "Now describe the current scene:\n"
    )

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
        Builds the ancestor mask via transitive closure (DFS).
        ancestor_mask[i][j] = True means field j is an ancestor of field i.
        """
        node_to_idx = {name: i for i, name in enumerate(self.DRIVELM_COT_VERTICES)}
        n = len(self.DRIVELM_COT_VERTICES)
        ancestor_mask = torch.zeros(n, n, dtype=torch.bool)

        def is_ancestor(a, b):
            visited = set()
            stack = [a]
            while stack:
                node = stack.pop()
                if node == b:
                    return True
                if node in visited:
                    continue
                visited.add(node)
                for src, dst in self.DRIVELM_COT_EDGES:
                    if src == node:
                        stack.append(dst)
            return False

        for i, fi in enumerate(self.DRIVELM_COT_VERTICES):
            for j, fj in enumerate(self.DRIVELM_COT_VERTICES):
                if i != j and is_ancestor(fj, fi):
                    ancestor_mask[i, j] = True

        return ancestor_mask
    
    def generate_trajectory_parallel(self, images, text_prompt, max_new_tokens=512):
        from data.preprocess import PromptFormatter

        # Build prompt with field template and multi-shot example appended
        question = text_prompt[1]["content"][-1]["text"].replace("Question: ", "")
        full_question = question + "\n\n" + self.COT_EXAMPLE + self.COT_TEMPLATE

        messages = PromptFormatter.format(full_question, num_images=len(images) if images else 0)
        formatted_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=formatted_prompt,
            images=images if images else None,
            return_tensors="pt"
        ).to(self.model.device)

        ancestor_mask = self.build_ancestor_mask()

        start_time = time.time()

        dag_scheduler = DagScheduler(
            formatted_prompt, inputs, self,
            self.DRIVELM_COT_VERTICES, self.DRIVELM_COT_EDGES,
            self.DRIVELM_COT_MAX_LENGTHS, ancestor_mask
        )
        generated_tokens = dag_scheduler.run_parallel_decoding()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        latency = time.time() - start_time

        raw_text = self.processor.decode(generated_tokens, skip_special_tokens=True)

        result = self._parse_output(raw_text, latency)
        result["model_type"] = "Qwen2.5VL_FastDriveCoT_Parallel"
        return result
