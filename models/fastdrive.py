import copy
import torch
import time
import re
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from .baseline import QwenBaselineVLA
from .dag_scheduler import DagScheduler
from data.preprocess import PromptFormatter
from search.mcts import DagMCTSNode


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
        "junction_type": 12, "road_type": 10,
        "traffic_light": 8, "traffic_sign": 10,
        "lanes_enumeration": 12,
        "lane_detail_0": 20, "lane_detail_1": 20, "lane_detail_2": 20,
        "critical_objects_enumeration": 15,
        "critical_object_0": 25, "critical_object_1": 25,
        "critical_object_2": 25, "critical_object_3": 25,
        "traffic_regulation_summary": 20,
        "non_interactive_summary": 25, "interactive_summary": 25,
        "ego_behavior_summary": 30,
    }

    COT_EXAMPLE = (
        "Complete each field with a short phrase describing ONLY what you observe in the images.\n"
        "Rules:\n"
        "- lighting: always describe the ambient light (e.g. bright daylight, overcast, night)\n"
        "- road_condition: always describe the road surface (e.g. dry, wet, clear)\n"
        "- weather: always describe the sky/weather (e.g. clear, cloudy, rainy)\n"
        "- junction_type: describe the intersection type visible ahead (e.g. signalized intersection, "
        "T-junction, roundabout, straight road). Use your visual observation, not a default.\n"
        "- Use N/A only for lane_detail_1, lane_detail_2, critical_object_1/2/3 when those "
        "lanes or objects are not present.\n"
        "- ego_behavior_summary: end with exactly one of: "
        "ACCELERATE, DECELERATE, STOP, YIELD, TURN_LEFT, TURN_RIGHT, LANE_CHANGE\n\n"
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

    def get_field_first_token(self, prefix_kv, stub_ids, prefix_length, branch_lengths, do_sample=False, temperature=1.0):
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
        logits = out.logits[0, -1]
        if do_sample:
            scaled = logits / max(temperature, 1e-6)
            probs = torch.softmax(scaled, dim=-1)
            token = torch.multinomial(probs, num_samples=1).item()
        else:
            token = logits.argmax(-1).item()
        if token in self._get_stop_token_ids():
            return None, out.past_key_values
        return token, out.past_key_values

    def get_fields_first_tokens_batched(self, prefix_kv, fields_and_stubs, prefix_length):
        results = {}
        for v, stub_ids in fields_and_stubs.items():
            results[v] = self.get_field_first_token(prefix_kv, stub_ids, prefix_length, [])
        return results

    def decode_next_token(self, token_id, position, field_kv, do_sample=False, temperature=1.0):
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
        logits = out.logits[0, -1]
        if do_sample:
            scaled = logits / max(temperature, 1e-6)
            probs = torch.softmax(scaled, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
        else:
            next_token = logits.argmax(-1).item()
        return next_token, out.past_key_values

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
            messages,
            tokenize=False,
            add_generation_prompt=True
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

        cot_block = f"<cot> {chr(10).join(cot_lines)} </cot>"
        structured_text = f"{cot_block}\n<action> {action} </action>" if action else cot_block

        result = self._parse_output(structured_text, latency)
        result["raw_text"] = raw_text
        result["model_type"] = "Qwen2.5VL_FastDriveCoT_Parallel"
        return result

    def mcts_fastdrive_generate(
        self,
        images,
        text_prompt,
        iterations=10,
        branches_per_wave=3,
        temperature=0.8,
        explor_const=1.0,
    ):
        """
        MCTS over DAG wave completions (FastDriveCoT-MCTS).

        Instead of searching over raw token sequences, the tree branches at
        the *wave* level of the CoT DAG.  Each node holds a partially-decoded
        CoT (some waves filled, some pending).  Expansion runs one parallel
        forward pass per wave (FastDriveCoT's strength) with temperature
        sampling to produce diverse branches.  Leaf nodes are scored with
        self_evaluate_state and the score is backpropagated.

        Tree depth  = number of DAG waves (~5-6 for the DriveLM DAG).
        Branching   = branches_per_wave sampled completions per expansion.
        Cost        = iterations × (waves × parallel passes + 1 eval call).
        """
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
            return_tensors="pt",
        ).to(self.model.device)

        ancestor_mask = self.build_ancestor_mask()
        self._last_inputs = inputs

        field_stub_ids = {
            v: self.processor.tokenizer.encode(f"{v}:", add_special_tokens=False)
            for v in self.DRIVELM_COT_VERTICES
        }

        prefix_length = inputs["input_ids"].shape[1]
        branch_lengths = [self.DRIVELM_COT_MAX_LENGTHS[v] for v in self.DRIVELM_COT_VERTICES]
        slot_tensors = [
            torch.zeros(self.DRIVELM_COT_MAX_LENGTHS[v], dtype=torch.long, device=self.model.device)
            for v in self.DRIVELM_COT_VERTICES
        ]
        full_input_ids = torch.cat(
            [inputs["input_ids"].squeeze(0)] + slot_tensors
        ).unsqueeze(0)
        _, prefix_kv = self.parallel_forward_pass(
            full_input_ids, branch_lengths,
            ancestor_mask=ancestor_mask,
            padding_lengths=branch_lengths,
        )

        scheduler = DagScheduler(
            formatted_prompt, inputs, self,
            self.DRIVELM_COT_VERTICES, self.DRIVELM_COT_EDGES,
            self.DRIVELM_COT_MAX_LENGTHS, ancestor_mask,
            field_stub_ids=field_stub_ids,
        )
        waves = scheduler.get_waves()

        empty_tokens = {v: [] for v in self.DRIVELM_COT_VERTICES}
        root = DagMCTSNode(field_tokens=empty_tokens, wave_index=0)

        start_time = time.time()

        for _ in range(iterations):
            node = root
            while node.children and not node.is_terminal(waves):
                node = max(node.children, key=lambda c: c.ucb_score(root.visits + 1, explor_const))

            if node.is_terminal(waves):
                reward = self._evaluate_dag_node(node, inputs)
                self._backprop(node, reward)
                continue

            wave_fields = waves[node.wave_index]
            for _ in range(branches_per_wave):
                child_tokens = node.clone_tokens()
                wave_result = scheduler.decode_wave(
                    wave_fields,
                    seed_field_tokens=child_tokens,
                    prefix_kv=prefix_kv,
                    prefix_length=prefix_length,
                    do_sample=True,
                    temperature=temperature,
                )
                child_tokens.update(wave_result)
                child = DagMCTSNode(
                    field_tokens=child_tokens,
                    wave_index=node.wave_index + 1,
                    parent=node,
                )
                node.children.append(child)

            sim_node = node.children[0]
            sim_tokens = sim_node.clone_tokens()
            for future_wave in waves[sim_node.wave_index:]:
                wave_result = scheduler.decode_wave(
                    future_wave,
                    seed_field_tokens=sim_tokens,
                    prefix_kv=prefix_kv,
                    prefix_length=prefix_length,
                    do_sample=False,
                )
                sim_tokens.update(wave_result)

            terminal = DagMCTSNode(field_tokens=sim_tokens, wave_index=len(waves))
            reward = self._evaluate_dag_node(terminal, inputs)
            self._backprop(sim_node, reward)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.time() - start_time

        best_node = self._best_leaf(root, waves)
        best_tokens = best_node.field_tokens

        for wave in waves[best_node.wave_index:]:
            wave_result = scheduler.decode_wave(
                wave,
                seed_field_tokens=best_tokens,
                prefix_kv=prefix_kv,
                prefix_length=prefix_length,
                do_sample=False,
            )
            best_tokens.update(wave_result)

        cot_lines = []
        for v in self.DRIVELM_COT_VERTICES:
            if v == "ego_behavior_summary":
                continue
            toks = [t for t in best_tokens.get(v, []) if t != 0]
            text = self.processor.decode(toks, skip_special_tokens=True) if toks else ""
            cot_lines.append(f"{v}: {text}")

        ego_toks = [t for t in best_tokens.get("ego_behavior_summary", []) if t != 0]
        ego_text = self.processor.decode(ego_toks, skip_special_tokens=True) if ego_toks else ""

        action = None
        for candidate in ["STOP", "YIELD", "ACCELERATE", "DECELERATE", "TURN_LEFT", "TURN_RIGHT", "LANE_CHANGE"]:
            if candidate.lower().replace("_", " ") in ego_text.lower() or candidate in ego_text.upper():
                action = candidate
                break

        cot_block = f"<cot> {chr(10).join(cot_lines)} </cot>"
        structured_text = f"{cot_block}\n<action> {action} </action>" if action else cot_block

        result = self._parse_output(structured_text, latency)
        result["model_type"] = "Qwen2.5VL_FastDriveCoT_MCTS"
        return result

    def _evaluate_dag_node(self, node, inputs):
        from search.mcts import MCTSNode as _MCTSNode

        ego_toks = [t for t in node.field_tokens.get("ego_behavior_summary", []) if t != 0]
        if ego_toks:
            action_tensor = torch.tensor(ego_toks, dtype=torch.long, device=self.model.device)
        else:
            action_tensor = torch.tensor(
                self.processor.tokenizer.encode("unknown", add_special_tokens=False),
                dtype=torch.long, device=self.model.device,
            )

        cot_text = node.as_text(self.DRIVELM_COT_VERTICES, self.processor)
        fake_node = _MCTSNode(state={"input_ids": inputs["input_ids"], "cot": cot_text})
        return self.self_evaluate_state(inputs, fake_node, action_tensor)

    @staticmethod
    def _backprop(node, reward):
        current = node
        while current is not None:
            current.visits += 1
            current.value += reward
            current = current.parent

    @staticmethod
    def _best_leaf(root, waves):
        node = root
        while node.children:
            node = max(node.children, key=lambda c: c.visits)
        return node
