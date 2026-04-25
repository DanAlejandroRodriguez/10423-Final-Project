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
        return (self.value / 