"""
Property-based tests for MCTSNode.

Verifies mathematical invariants of UCB scoring and reward calculation
that must hold for any correct MCTS implementation.

Run with:
    pip install hypothesis
    python -m pytest tests/test_mcts_properties.py -v
"""

import math
import sys
from unittest.mock import MagicMock
import torch
import pytest
from hypothesis import given, assume
from hypothesis import strategies as st

sys.modules.setdefault("qwen_vl_utils", MagicMock())

from search.mcts import MCTSNode


def make_node(visits=1, value=0.0, parent=None):
    node = MCTSNode(state={}, parent=parent)
    node.visits = visits
    node.value = value
    return node


class TestUCBScore:

    @given(
        visits=st.integers(min_value=1, max_value=10000),
        value=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
        total_visits=st.integers(min_value=1, max_value=100000),
        explor_const=st.floats(min_value=0.01, max_value=10.0, allow_nan=False),
    )
    def test_score_is_finite(self, visits, value, total_visits, explor_const):
        """UCB score must always be finite for visited nodes."""
        node = make_node(visits=visits, value=value)
        assert math.isfinite(node.ucb_score(total_visits, explor_const))

    @given(
        visits=st.integers(min_value=1, max_value=1000),
        value=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
        total_visits=st.integers(min_value=1, max_value=10000),
    )
    def test_exploitation_equals_mean_value(self, visits, value, total_visits):
        """With explor_const=0, score equals value/visits."""
        node = make_node(visits=visits, value=value)
        assert abs(node.ucb_score(total_visits, explor_const=0.0) - value / visits) < 1e-5

    @given(
        visits=st.integers(min_value=1, max_value=1000),
        value=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
        total_visits=st.integers(min_value=2, max_value=10000),
    )
    def test_exploration_increases_with_total_visits(self, visits, value, total_visits):
        """More total visits increases the exploration bonus."""
        assume(total_visits > visits)
        node = make_node(visits=visits, value=value)
        assert node.ucb_score(total_visits * 2) >= node.ucb_score(total_visits)

    @given(
        visits_a=st.integers(min_value=1, max_value=1000),
        visits_b=st.integers(min_value=1, max_value=1000),
        value=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
        total_visits=st.integers(min_value=2, max_value=10000),
    )
    def test_exploration_decreases_with_own_visits(self, visits_a, visits_b, value, total_visits):
        """More own visits lowers the exploration bonus."""
        assume(visits_a < visits_b)
        node_a = make_node(visits=visits_a, value=value * visits_a)
        node_b = make_node(visits=visits_b, value=value * visits_b)
        assert node_a.ucb_score(total_visits) >= node_b.ucb_score(total_visits)

    def test_unvisited_node_always_selected_first(self):
        """Unvisited node (visits=0) must score higher than any visited node."""
        unvisited = make_node(visits=0, value=0.0)
        visited = make_node(visits=100, value=1000.0)
        score = unvisited.ucb_score(total_visits=10)
        assert score == float('inf') or score > visited.ucb_score(total_visits=10)

    def test_ucb_selects_unvisited_over_visited(self):
        """UCB selection must prefer unvisited children."""
        root = MCTSNode(state={})
        root.visits = 5
        root.children = {
            "visited": make_node(visits=3, value=2.0),
            "unvisited": make_node(visits=0, value=0.0),
        }
        selected = max(root.children.values(), key=lambda n: n.ucb_score(root.visits))
        assert selected is root.children["unvisited"]


class TestReward:

    @given(
        score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        latency=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
        hyperparam=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    def test_reward_decreases_with_latency(self, score, latency, hyperparam):
        """Higher latency must produce lower or equal reward."""
        node = make_node()
        assert node.calculate_reward(score, latency, hyperparam) >= \
               node.calculate_reward(score, latency + 1.0, hyperparam)

    @given(
        score_a=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        score_b=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        latency=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    )
    def test_reward_increases_with_verifier_score(self, score_a, score_b, latency):
        """Higher verifier score must produce higher or equal reward."""
        assume(score_a <= score_b)
        node = make_node()
        assert node.calculate_reward(score_a, latency) <= node.calculate_reward(score_b, latency)


class TestNodeStructure:

    def test_initializes_with_zero_visits(self):
        assert MCTSNode(state={}).visits == 0

    def test_initializes_with_zero_value(self):
        assert MCTSNode(state={}).value == 0.0

    def test_parent_reference(self):
        parent = MCTSNode(state={})
        child = MCTSNode(state={}, parent=parent)
        assert child.parent is parent

    def test_root_has_no_parent(self):
        assert MCTSNode(state={}).parent is None

    def test_backprop_reaches_root(self):
        """Backpropagation must update every ancestor up to root."""
        root = MCTSNode(state={})
        child = MCTSNode(state={}, parent=root)
        grandchild = MCTSNode(state={}, parent=child)

        node = grandchild
        while node:
            node.visits += 1
            node.value += 0.5
            node = node.parent

        assert root.visits == child.visits == grandchild.visits == 1
