import copy
import torch
import time
import re
from qwen_vl_utils import process_vision_info
from .baseline import QwenBaselineVLA
from .dag_scheduler import DagScheduler
from data.preprocess import PromptFormatter


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
        "lighting": 3, "road_condition": 4, "weather": 3,
        "junction_type": 4, "road_type": 4,
        "traffic_light": 3, "traffic_sign": 4,
        "lanes_enumeration": 4,
        "lane_detail_0": 6, "lane_detail_1": 6, "lane_detail_2": 6,
        "critical_objects_enumeration": 6,
        "critical_object_0": 8, "critical_object_1": 8,
        "critical_object_2": 8, "critical_object_3": 8,
        "traffic_regulation_summary": 6,
        "non_interactive_summary": 8, "interactive_summary": 8,
        "ego_behavior_summary": 10,
    }

    COT_EXAMPLE = (
        "Fill in each field with a short phrase describing what you observe.\n\n"
        "Example 1:\n"
        "lighting: bright daylight\n"
        "road_condition: dry asphalt\n"
        "weather: clear sky\n"
        "junction_type: T-intersection\n"
        "road_type: urban road\n"
        "traffic_light: green\n"
        "traffic_sign: speed limit sign\n"
        "lanes_enumeration: two lanes\n"
        "lane_detail_0: ego lane clear\n"
        "lane_detail_1: adjacent lane occupied\n"
        "lane_detail_2: N/A\n"
        "critical_objects_enumeration: one car\n"
        "critical_object_0: car ahead moving slowly\n"
        "critical_object_1: N/A\n"
        "critical_object_2: N/A\n"
        "critical_object_3: N/A\n"
        "traffic_regulation_summary: green light, proceed\n"
        "non_interactive_summary: road ahead is clear\n"
        "interactive_summary: slow car requires attention\n"
        "ego_behavior_summary: ACCELERATE, maintain speed\n\n"
        "Example 2:\n"
        "lighting: night, street lamps on\n"
        "road_condition: wet surface\n"
        "weather: rainy\n"
        "junction_type: signalized intersection\n"
        "road_type: urban road\n"
        "traffic_light: red\n"
        "traffic_sign: stop sign\n"
        "lanes_enumeration: one lane\n"
        "lane_detail_0: blocked by red light\n"
        "lane_detail_1: N/A\n"
        "lane_detail_2: N/A\n"
        "critical_objects_enumeration: pedestrians\n"
        "critical_object_0: pedestrian crossing\n"
        "critical_object_1: N/A\n"
        "critical_object_2: N/A\n"
        "critical_object_3: N/A\n"
        "traffic_regulation_summary: red light, must stop\n"
        "non_interactive_summary: intersection blocked\n"
        "interactive_summary: pedestrians crossing ahead\n"
        "ego_behavior_summary: STOP, wait for green\n\n"
    )

    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
        super().__init__(model_id=model_id, attn_implementation="eager")

    def build_dag_position_ids(self, prefix_length, branch_lengths):
        prefix_pos = torch.arange(prefix_length)
        branch_pos = torch.cat([
            torch.arange(prefix_length, prefix_length + blen)
            for blen in branch_lengths
        ])
        return torch.cat([prefix_pos, branch_pos]).unsqueeze(0)

    def parallel_forward_pass(self, input_ids, branch_lengths, ancestor_mask=None, padding_lengths=None, position_ids=None, past_key_values=None, new_token_positions=None):
        device = self.model.device
        input_ids = input_ids.to(device)
        prefix_length = input_ids.shape[1] - sum(branch_lengths)

        prefix_ids = input_ids[:, :prefix_length].to(device)
        extra = {}
        if hasattr(self, '_last_inputs'):
            for k in ('pixel_values', 'image_grid_thw'):
                if k in self._last_inputs:
                    extra[k] = self._last_inputs[k].to(device)
        with torch.no_grad():
            prefix_out = self.model(input_ids=prefix_ids, use_cache=True, **extra)
        return prefix_out.logits, prefix_out.past_key_values

    def _get_stop_token_ids(self):
        if hasattr(self, '_stop_ids'):
            return self._stop_ids
        nl_ids = self.processor.tokenizer.encode('\n', add_special_tokens=False)
        eos_id = self.processor.tokenizer.eos_token_id
        im_end_ids = self.processor.tokenizer.encode('<|im_end|>', add_special_tokens=False)
        self._stop_ids = set(nl_ids + im_end_ids + ([eos_id] if eos_id else []))
        return self._stop_ids

    def _copy_kv(self, kv):
        new_kv = copy.copy(kv)
        new_kv.layers = []
        for layer in kv.layers:
            new_layer = copy.copy(layer)
            if layer.is_initialized:
                new_layer.keys = layer.keys.clone()
                new_layer.values = layer.values.clone()
            new_kv.layers.append(new_layer)
        return new_kv

    def get_field_first_token(self, prefix_kv, stub_ids, prefix_length, branch_lengths):
        device = self.model.device
        field_kv = self._copy_kv(prefix_kv)
        stub_tensor = torch.tensor([stub_ids], dtype=torch.long, device=device)
        stub_pos = torch.arange(
            prefix_length, prefix_length + len(stub_ids), dtype=torch.long, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            out = self.model(
                input_ids=stub_tensor,
                position_ids=stub_pos,
                past_key_values=field_kv,
                use_cache=True,
            )
        token = out.logits[0, -1].argmax(-1).item()
        if token in self._get_stop_token_ids():
            return None, out.past_key_values
        return token, out.past_key_values

    def get_fields_first_tokens_batched(self, prefix_kv, fields_and_stubs, prefix_length):
        results = {}
        for v, stub_ids in fields_and_stubs.items():
            results[v] = self.get_field_first_token(prefix_kv, stub_ids, prefix_length, [])
        return results

    def decode_next_token(self, token_id, position, field_kv):
        device = self.model.device
        tok = torch.tensor([[token_id]], dtype=torch.long, device=device)
        pos = torch.tensor([[position]], dtype=torch.long, device=device)
        with torch.no_grad():
            out = self.model(
                input_ids=tok,
                position_ids=pos,
                past_key_values=field_kv,
                use_cache=True,
            )
        return out.logits[0, -1].argmax(-1).item(), out.past_key_values

    def build_ancestor_mask(self):
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
        question = text_prompt[1]["content"][-1]["text"].replace("Question: ", "")
        prefix_question = question + "\n\n" + self.COT_EXAMPLE + "Now describe the current scene:\n"

        messages = PromptFormatter.format(prefix_question, images=images)
        formatted_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[formatted_prompt],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        ancestor_mask = self.build_ancestor_mask()
        self._last_inputs = inputs

        field_stub_ids = {
            v: self.processor.tokenizer.encode(f"{v}:", add_special_tokens=False)
            for v in self.DRIVELM_COT_VERTICES
        }

        start_time = time.time()

        dag_scheduler = DagScheduler(
            formatted_prompt, inputs, self,
            self.DRIVELM_COT_VERTICES, self.DRIVELM_COT_EDGES,
            self.DRIVELM_COT_MAX_LENGTHS, ancestor_mask,
            field_stub_ids=field_stub_ids,
        )
        generated_tokens = dag_scheduler.run_parallel_decoding()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        latency = time.time() - start_time

        raw_text = self.processor.decode(generated_tokens, skip_special_tokens=True)

        offset = 0
        field_texts = {}
        for v in self.DRIVELM_COT_VERTICES:
            length = self.DRIVELM_COT_MAX_LENGTHS[v]
            field_toks = [t for t in generated_tokens[offset:offset + length].tolist() if t != 0]
            field_texts[v] = self.processor.decode(field_toks, skip_special_tokens=True)
            offset += length

        cot_lines = [f"{v}: {field_texts[v]}" for v in self.DRIVELM_COT_VERTICES if v != "ego_behavior_summary"]
        ego_text = field_texts.get("ego_behavior_summary", "")

        action = None
        for candidate in ["STOP", "YIELD", "ACCELERATE", "DECELERATE", "TURN_LEFT", "TURN_RIGHT", "LANE_CHANGE"]:
            if candidate.lower().replace("_", " ") in ego_text.lower() or candidate in ego_text.upper():
                action = candidate
                break

        structured_text = (
            f"<cot> {chr(10).join(cot_lines)} </cot>\n"
            f"<action> {action or 'ACCELERATE'} </action>"
        )

        result = self._parse_output(structured_text, latency)
        result["raw_text"] = raw_text
        result["model_type"] = "Qwen2.5VL_FastDriveCoT_Parallel"
        return result
