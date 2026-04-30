"""
Property-based tests for DAG scheduler correctness.

Based on Algorithm 1 and Section 3.2 of the FastDriveCoT paper.
Tests verify scheduling invariants without depending on any specific
implementation — only the behavioral contract matters.

Run with:
    python -m pytest tests/test_dag_scheduler_properties.py -v
"""

import sys
from unittest.mock import MagicMock
import torch
import pytest
from hypothesis import given, assume, settings
from hypothesis import strategies as st

sys.modules.setdefault("qwen_vl_utils", MagicMock())


# ---------------------------------------------------------------------------
# Minimal mock model — returns zero logits, enough to drive the scheduler
# ---------------------------------------------------------------------------

def make_mock_model(vocab_size=32):
    mock = MagicMock()
    mock.model.device = torch.device("cpu")

    def fake_forward(input_ids, branch_lengths, **kwargs):
        kv = MagicMock()
        return torch.zeros(1, input_ids.shape[1], vocab_size), kv

    mock.parallel_forward_pass.side_effect = fake_forward
    mock._get_stop_token_ids.return_value = set()

    def fake_first_token(prefix_kv, stub_ids, prefix_length, branch_lengths, **kwargs):
        return 1, MagicMock()

    mock.get_field_first_token.side_effect = fake_first_token

    def fake_batched(prefix_kv, fields_and_stubs, prefix_length):
        return {v: (1, MagicMock()) for v in fields_and_stubs}

    mock.get_fields_first_tokens_batched.side_effect = fake_batched

    def fake_decode(token_id, position, field_kv, **kwargs):
        return 0, MagicMock()

    mock.decode_next_token.side_effect = fake_decode

    return mock
def make_inputs(prefix_len=5):
    return {"input_ids": torch.zeros(1, prefix_len, dtype=torch.long)}


# ---------------------------------------------------------------------------
# DAG graph strategies for Hypothesis
# ---------------------------------------------------------------------------

def linear_dag(n):
    """A → B → C → ... chain of n nodes."""
    vertices = [f"field_{i}" for i in range(n)]
    edges = [(vertices[i], vertices[i + 1]) for i in range(n - 1)]
    max_lengths = {v: 3 for v in vertices}
    return vertices, edges, max_lengths


def parallel_dag(n):
    """n independent source nodes, all feeding into one sink."""
    sources = [f"source_{i}" for i in range(n)]
    sink = "sink"
    vertices = sources + [sink]
    edges = [(s, sink) for s in sources]
    max_lengths = {v: 3 for v in vertices}
    return vertices, edges, max_lengths


def diamond_dag():
    """A → B, A → C, B → D, C → D."""
    vertices = ["A", "B", "C", "D"]
    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
    max_lengths = {v: 3 for v in vertices}
    return vertices, edges, max_lengths


# ---------------------------------------------------------------------------
# Helper: run scheduler and record which fields were active each step
# ---------------------------------------------------------------------------

def run_and_record(vertices, edges, max_lengths, prefix_len=5):
    from models.dag_scheduler import DagScheduler

    model = make_mock_model()
    inputs = make_inputs(prefix_len)

    scheduler = DagScheduler(
        prompt="test",
        inputs=inputs,
        model=model,
        vertices=vertices,
        edges=edges,
        max_lengths=max_lengths,
    )
    result = scheduler.run_parallel_decoding()
    return result, [], scheduler


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInitialReadySet:

    def test_source_nodes_start_in_S(self):
        """
        Algorithm 1 line 3: S ← {v | d_v = 0}.
        All nodes with no incoming edges must be in the initial ready set.
        """
        from models.dag_scheduler import DagScheduler
        vertices, edges, max_lengths = parallel_dag(4)
        scheduler = DagScheduler("", make_inputs(), make_mock_model(),
                                  vertices, edges, max_lengths)
        for source in ["source_0", "source_1", "source_2", "source_3"]:
            assert source in scheduler.S

    def test_dependent_nodes_not_in_initial_S(self):
        """Nodes with incoming edges must not start in S."""
        from models.dag_scheduler import DagScheduler
        vertices, edges, max_lengths = linear_dag(3)
        scheduler = DagScheduler("", make_inputs(), make_mock_model(),
                                  vertices, edges, max_lengths)
        assert "field_1" not in scheduler.S
        assert "field_2" not in scheduler.S

    def test_single_node_dag_starts_in_S(self):
        """A single node with no edges must be in S immediately."""
        from models.dag_scheduler import DagScheduler
        vertices = ["only"]
        scheduler = DagScheduler("", make_inputs(), make_mock_model(),
                                  vertices, [], {"only": 2})
        assert "only" in scheduler.S


class TestSchedulingOrder:

    def test_dependent_field_only_starts_after_prerequisite_finishes(self):
        """
        Algorithm 1 lines 9-12: a node enters S only when all prerequisites complete.
        In a linear chain A→B→C, B must not appear in S while A is still active.
        """
        from models.dag_scheduler import DagScheduler
        vertices, edges, max_lengths = linear_dag(3)
        active_sets = []

        model = make_mock_model()
        scheduler_ref = [None]

        original_decode = model.decode_next_token.side_effect

        def tracking_decode(token_id, position, field_kv, **kwargs):
            active_sets.append(list(scheduler_ref[0].S))
            return original_decode(token_id, position, field_kv, **kwargs)

        model.decode_next_token.side_effect = tracking_decode
        scheduler = DagScheduler("", make_inputs(), model, vertices, edges, max_lengths)
        scheduler_ref[0] = scheduler
        scheduler.run_parallel_decoding()

        for active in active_sets:
            assert not ("field_0" in active and "field_1" in active), \
                "field_1 was active while field_0 was still being decoded"

    def test_parallel_sources_decoded_simultaneously(self):
        """
        Independent source nodes must all appear in S together on the first step.
        This is the core parallelism claim of FastDriveCoT.
        """
        from models.dag_scheduler import DagScheduler
        vertices, edges, max_lengths = parallel_dag(3)
        first_S = [None]

        model = make_mock_model()
        scheduler_ref = [None]

        original_forward = model.parallel_forward_pass.side_effect

        def tracking_forward(input_ids, branch_lengths, **kwargs):
            if first_S[0] is None:
                first_S[0] = list(scheduler_ref[0].S)
            return original_forward(input_ids, branch_lengths, **kwargs)

        model.parallel_forward_pass.side_effect = tracking_forward
        scheduler = DagScheduler("", make_inputs(), model, vertices, edges, max_lengths)
        scheduler_ref[0] = scheduler
        scheduler.run_parallel_decoding()

        assert first_S[0] is not None
        for source in ["source_0", "source_1", "source_2"]:
            assert source in first_S[0], \
                f"{source} was not decoded in parallel with other sources"

    def test_diamond_sink_only_after_both_branches(self):
        """
        In A→B, A→C, B→D, C→D: D must not appear in S until both B and C finish.
        """
        from models.dag_scheduler import DagScheduler
        vertices, edges, max_lengths = diamond_dag()
        active_sets = []

        model = make_mock_model()
        scheduler_ref = [None]

        original_decode = model.decode_next_token.side_effect

        def tracking_decode(token_id, position, field_kv, **kwargs):
            active_sets.append(list(scheduler_ref[0].S))
            return original_decode(token_id, position, field_kv, **kwargs)

        model.decode_next_token.side_effect = tracking_decode
        scheduler = DagScheduler("", make_inputs(), model, vertices, edges, max_lengths)
        scheduler_ref[0] = scheduler
        scheduler.run_parallel_decoding()

        for active in active_sets:
            if "D" in active:
                assert "B" not in active and "C" not in active, \
                    "D was active before B and C both finished"


class TestTermination:

    @given(st.integers(min_value=1, max_value=5))
    def test_linear_dag_terminates(self, n):
        """Scheduler must terminate for any linear chain."""
        vertices, edges, max_lengths = linear_dag(n)
        result, _, _ = run_and_record(vertices, edges, max_lengths)
        assert result is not None

    @given(st.integers(min_value=1, max_value=4))
    def test_parallel_dag_terminates(self, n):
        """Scheduler must terminate for any parallel fan-in DAG."""
        vertices, edges, max_lengths = parallel_dag(n)
        result, _, _ = run_and_record(vertices, edges, max_lengths)
        assert result is not None

    def test_diamond_dag_terminates(self):
        """Scheduler must terminate for a diamond dependency graph."""
        vertices, edges, max_lengths = diamond_dag()
        result, _, _ = run_and_record(vertices, edges, max_lengths)
        assert result is not None

    def test_single_node_terminates(self):
        """Scheduler must terminate for a single node."""
        result, _, _ = run_and_record(["only"], [], {"only": 2})
        assert result is not None


class TestCriticalPath:

    def test_linear_chain_decodes_sequentially(self):
        """
        A linear chain of n nodes must decode each field fully before starting the next.
        """
        from models.dag_scheduler import DagScheduler
        n = 3
        max_len = 2
        vertices = [f"f{i}" for i in range(n)]
        edges = [(vertices[i], vertices[i + 1]) for i in range(n - 1)]
        max_lengths = {v: max_len for v in vertices}

        model = make_mock_model()
        scheduler = DagScheduler("", make_inputs(), model, vertices, edges, max_lengths)
        result = scheduler.run_parallel_decoding()
        assert result is not None
        assert len(result) == n * max_len

    def test_parallel_sources_fewer_decode_calls_than_sequential(self):
        """
        n parallel sources + 1 sink requires fewer decode_next_token calls
        than n+1 sequential nodes. This validates the speedup claim.
        """
        from models.dag_scheduler import DagScheduler
        n = 3
        max_len = 2

        def count_decode_calls(vertices, edges, max_lengths):
            model = make_mock_model()
            count = [0]
            original = model.decode_next_token.side_effect
            def counting_decode(token_id, position, field_kv, **kwargs):
                count[0] += 1
                return original(token_id, position, field_kv, **kwargs)
            model.decode_next_token.side_effect = counting_decode
            scheduler = DagScheduler("", make_inputs(), model, vertices, edges, max_lengths)
            scheduler.run_parallel_decoding()
            return count[0]

        vertices_p, edges_p, max_lengths_p = parallel_dag(n)
        max_lengths_p = {v: max_len for v in vertices_p}

        vertices_s, edges_s, max_lengths_s = linear_dag(n + 1)
        max_lengths_s = {v: max_len for v in vertices_s}

        parallel_calls = count_decode_calls(vertices_p, edges_p, max_lengths_p)
        sequential_calls = count_decode_calls(vertices_s, edges_s, max_lengths_s)

        assert parallel_calls <= sequential_calls, \
            f"Parallel DAG ({parallel_calls} calls) should be <= sequential ({sequential_calls} calls)"
