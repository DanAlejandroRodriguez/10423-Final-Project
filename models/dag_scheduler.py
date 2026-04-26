import torch


class DagScheduler():    
    def __init__(self, prompt, inputs, model, vertices, edges, max_lengths, ancestor_mask=None):
        self.prefix = prompt
        self.inputs = inputs
        self.model = model
        self.vertices = vertices
        self.max_lengths = max_lengths
        self.ancestor_mask = ancestor_mask

        self.num_incoming_edges = {name: 0 for name in vertices}
        self.edges = edges
        for (_, b) in edges:
            self.num_incoming_edges[b] += 1

        self.S = []
        for name, num_edges in self.num_incoming_edges.items():
            if num_edges == 0:
                self.S.append(name)

    def run_parallel_decoding(self):
        prefix_length = self.inputs["input_ids"].shape[1]
        branch_lengths = [self.max_lengths[v] for v in self.vertices]
        field_tokens = {v: [] for v in self.vertices}
        field_done = {v: False for v in self.vertices}

        while len(self.S) != 0:
            # Build full template sequence: prefix + all field slots
            device = self.inputs["input_ids"].device
            slot_tensors = []
            for v in self.vertices:
                tokens = field_tokens[v]
                pad_len = self.max_lengths[v] - len(tokens)
                slot_tensors.append(torch.tensor(tokens + [0] * pad_len, dtype=torch.long, device=device))

            input_ids = torch.cat(
                [self.inputs["input_ids"].squeeze(0)] + slot_tensors
            ).unsqueeze(0)

            padding_lengths = [self.max_lengths[v] - len(field_tokens[v]) for v in self.vertices]

            # One forward pass — get logits for all field positions
            logits = self.model.parallel_forward_pass(
                input_ids,
                branch_lengths,
                ancestor_mask=self.ancestor_mask,
                padding_lengths=padding_lengths,
            )

            # Greedy decode one new token for each active field in S
            for v in self.S:
                v_idx = self.vertices.index(v)
                field_offset = prefix_length + sum(branch_lengths[:v_idx])
                token_pos = field_offset + len(field_tokens[v])
                next_token = logits[0, token_pos].argmax(-1).item()
                field_tokens[v].append(next_token)

            # Check which fields finished this step
            finished = []
            for v in list(self.S):
                if len(field_tokens[v]) >= self.max_lengths[v]:
                    field_done[v] = True
                    finished.append(v)

            for v in finished:
                self.S.remove(v)
                for (a, b) in self.edges:
                    if a == v:
                        self.num_incoming_edges[b] -= 1
                        if self.num_incoming_edges[b] == 0:
                            self.S.append(b)

        # Concatenate all field tokens in order
        all_tokens = []
        for v in self.vertices:
            all_tokens.extend(field_tokens[v])
        return torch.tensor(all_tokens, dtype=torch.long)
