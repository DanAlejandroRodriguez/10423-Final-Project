import torch


class DagScheduler():
    def __init__(self, prompt, inputs, model, vertices, edges, max_lengths, ancestor_mask=None, field_stub_ids=None):
        self.prefix = prompt
        self.inputs = inputs
        self.model = model
        self.vertices = vertices
        self.max_lengths = max_lengths
        self.ancestor_mask = ancestor_mask
        self.field_stub_ids = field_stub_ids

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
        field_kv = {}
        step1_done = False
        stop_ids = self.model._get_stop_token_ids() if hasattr(self.model, '_get_stop_token_ids') else set()

        while len(self.S) != 0:
            device = self.inputs["input_ids"].device

            if not step1_done:
                slot_tensors = [
                    torch.tensor([0] * self.max_lengths[v], dtype=torch.long, device=device)
                    for v in self.vertices
                ]
                input_ids = torch.cat(
                    [self.inputs["input_ids"].squeeze(0)] + slot_tensors
                ).unsqueeze(0)
                _, prefix_kv = self.model.parallel_forward_pass(
                    input_ids, branch_lengths,
                    ancestor_mask=self.ancestor_mask,
                    padding_lengths=[self.max_lengths[v] for v in self.vertices],
                    past_key_values=None,
                    new_token_positions=None,
                )
                source_stubs = {
                    v: self.field_stub_ids[v]
                    for v in self.S
                    if self.field_stub_ids and v in self.field_stub_ids
                }
                if source_stubs and hasattr(self.model, 'get_fields_first_tokens_batched'):
                    batch_results = self.model.get_fields_first_tokens_batched(
                        prefix_kv, source_stubs, prefix_length
                    )
                    for v in self.S:
                        first_token, stub_kv = batch_results.get(v, (0, prefix_kv))
                        field_kv[v] = stub_kv
                        if first_token is not None:
                            field_tokens[v].append(first_token)
                else:
                    for v in self.S:
                        stub_ids = self.field_stub_ids.get(v, []) if self.field_stub_ids else []
                        if stub_ids:
                            first_token, stub_kv = self.model.get_field_first_token(
                                prefix_kv, stub_ids, prefix_length, branch_lengths
                            )
                            field_kv[v] = stub_kv
                        else:
                            first_token, field_kv[v] = 0, prefix_kv
                        if first_token is not None:
                            field_tokens[v].append(first_token)
                step1_done = True
            else:
                for v in self.S:
                    if not field_tokens[v]:
                        continue
                    stub_ids = self.field_stub_ids.get(v, []) if self.field_stub_ids else []
                    tok_pos = prefix_length + len(stub_ids) + len(field_tokens[v]) - 1
                    next_token, updated_kv = self.model.decode_next_token(
                        field_tokens[v][-1], tok_pos, field_kv.get(v)
                    )
                    field_kv[v] = updated_kv
                    if next_token in stop_ids:
                        field_tokens[v].extend([0] * (self.max_lengths[v] - len(field_tokens[v])))
                    else:
                        field_tokens[v].append(next_token)

            finished = [
                v for v in list(self.S)
                if len(field_tokens[v]) >= self.max_lengths[v] or (step1_done and not field_tokens[v])
            ]

            for v in finished:
                self.S.remove(v)
                for (a, b) in self.edges:
                    if a == v:
                        self.num_incoming_edges[b] -= 1
                        if self.num_incoming_edges[b] == 0:
                            self.S.append(b)
                            if step1_done:
                                stub_ids = self.field_stub_ids.get(b, []) if self.field_stub_ids else []
                                if stub_ids:
                                    first_token, stub_kv = self.model.get_field_first_token(
                                        prefix_kv, stub_ids, prefix_length, branch_lengths
                                    )
                                    field_kv[b] = stub_kv
                                else:
                                    first_token, field_kv[b] = 0, prefix_kv
                                if first_token is not None:
                                    field_tokens[b].append(first_token)

        all_tokens = []
        for v in self.vertices:
            toks = field_tokens[v]
            all_tokens.extend(toks + [0] * (self.max_lengths[v] - len(toks)))
        return torch.tensor(all_tokens, dtype=torch.long)
