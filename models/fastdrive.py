import torch
from .baseline import QwenBaselineVLA


class FastDriveVLA(QwenBaselineVLA):
    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
        super().__init__(model_id=model_id)

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

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=additive_mask,
                position_ids=position_ids,
                use_cache=False,
            )

        return outputs.logits
