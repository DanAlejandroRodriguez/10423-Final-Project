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
    def fake_forward(input_ids, branch_lengths, **kwargs):
        total_len = input_ids.shape[1]
        return torch.zeros(1, total_len, vocab_size)
    mock.parallel_forward_pass.side_effect = fake_forward
    mock.model.device = torch.device("cpu")
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

    steps = []
    model = make_mock_model()
    inputs = make_inputs(prefix_len)

    original_forward = model.parallel_forward_pass.side_effect

    def tracking_forward(input_ids, branch_lengths, **kwargs):
        # Record which fields are currently being decoded (non-zero padding = active)
        steps.append(list(scheduler.S))
        return original_forward(input_ids, branch_lengths, **kwargs)

    model.parallel_forward_pass.side_effect = tracking_forward

    scheduler = DagScheduler(
        prompt="test",
        inputs=inputs,
        model=model,
        vertices=vertices,
        edges=edges,
        max_lengths=max_lengths,
    )
    result = scheduler.run_parallel_decoding()
    return result, steps, scheduler


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

        def tracking_forward(input_ids, branch_lengths, **kwargs):
            active_sets.append(list(scheduler_ref[0].S))
            return torch.zeros(1, input_ids.shape[1], 32)

        model.parallel_forward_pass.side_effect = tracking_forward
        scheduler = DagScheduler("", make_inputs(), model, vertices, edges, max_lengths)
        scheduler_ref[0] = scheduler
        scheduler.run_parallel_decoding()

        # field_1 must never appear in S at the same time as field_0
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
        first_step = [None]

        model = make_mock_model()
        scheduler_ref = [None]

        def tracking_forward(input_ids, branch_lengths, **kwargs):
            if first_step[0] is None:
                first_step[0] = list(scheduler_ref[0].S)
            return torch.zeros(1, input_ids.shape[1], 32)

        model.parallel_forward_pass.side_effect = tracking_forward
        scheduler = DagScheduler("", make_inputs(), model, vertices, edges, max_lengths)
        scheduler_ref[0] = scheduler
        scheduler.run_parallel_decoding()

        assert first_step[0] is not None
        for source in ["source_0", "source_1", "source_2"]:
            assert source in first_step[0], \
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

        def tracking_forward(input_ids, branch_lengths, **kwargs):
            active_sets.append(list(scheduler_ref[0].S))
            return torch.zeros(1, input_ids.shape[1], 32)

        model.parallel_forward_pass.side_effect = tracking_forward
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

    def test_linear_chain_requires_n_steps(self):
        """
        A linear chain of n nodes requires exactly n * max_length forward passes
        (one token per step, fields are sequential).
        """
        from models.dag_scheduler import DagScheduler
        n = 3
        max_len = 2
        vertices = [f"f{i}" for i in range(n)]
        edges = [(vertices[i], vertices[i + 1]) for i in range(n - 1)]
        max_lengths = {v: max_len for v in vertices}

        call_count = [0]
        model = make_mock_model()

        def counting_forward(input_ids, branch_lengths, **kwargs):
            call_count[0] += 1
            return torch.zeros(1, input_ids.shape[1], 32)

        model.parallel_forward_pass.side_effect = counting_forward
        scheduler = DagScheduler("", make_inputs(), model, vertices, edges, max_lengths)
        scheduler.run_parallel_decoding()

        assert call_count[0] == n * max_len, \
            f"Linear chain of {n} nodes with max_len={max_len} should need {n * max_len} steps, got {call_count[0]}"

    def test_parallel_sources_fewer_steps_than_sequential(self):
        """
        n parallel sources + 1 sink requires fewer forward passes than n+1 sequential nodes.
        This validates the speedup claim of FastDriveCoT.
        """
        from models.dag_scheduler import DagScheduler
        n = 3
        max_len = 2

        # Parallel: n sources in parallel, then sink
        vertices_p, edges_p, max_lengths_p = parallel_dag(n)
        max_lengths_p = {v: max_len for v in vertices_p}

        # Sequential: n+1 nodes in a chain
        vertices_s, edges_s, max_lengths_s = linear_dag(n + 1)
        max_lengths_s = {v: max_len for v in vertices_s}

        def count_steps(vertices, edges, max_lengths):
            count = [0]
            model = make_mock_model()
            def counting_forward(input_ids, branch_lengths, **kwargs):
                count[0] += 1
                return torch.zeros(1, input_ids.shape[1], 32)
            model.parallel_forward_pass.side_effect = counting_forward
            scheduler = DagScheduler("", make_inputs(), model, vertices, edges, max_lengths)
            scheduler.run_parallel_decoding()
            return count[0]

        parallel_steps = count_steps(vertices_p, edges_p, max_lengths_p)
        sequential_steps = count_steps(vertices_s, edges_s, max_lengths_s)

        assert parallel_steps < sequential_steps, \
            f"Parallel DAG ({parallel_steps} steps) should be faster than sequential ({sequential_steps} steps)"
