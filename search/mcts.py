import torch
import math

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state      # the current visual inputs and VLA context
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.visits_per_child = {}
        self.value = 0.0

    def ucb_score(self, total_visits, explor_const=1.00):
        """
        Calculate the upper confidence bound for trees
        """
        return (self.value / self.visits) + explor_const * math.sqrt(math.log(total_visits) / self.visits)

    def calculate_reward(self, verifier_score, latency, latency_hyperparam=0.1):
        """
        Calculate the search reward using the learned verifier model (which is Qwen-VL itself)
        """
        return verifier_score - latency_hyperparam * latency
