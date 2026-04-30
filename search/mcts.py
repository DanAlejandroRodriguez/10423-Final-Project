import math
import copy


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state      # the current visual inputs and VLA context
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0

    def ucb_score(self, total_visits, explor_const=1.00):
        """
        Calculate the upper confidence bound for trees
        """
        return (self.value / self.visits) + explor_const * math.sqrt(math.log(total_visits) / self.visits) if self.visits > 0 else float('inf')

    def calculate_reward(self, verifier_score, latency, latency_hyperparam=0.1):
        """
        Calculate the search reward using the learned verifier model (which is Qwen-VL itself)
        """
        return verifier_score - latency_hyperparam * latency


class DagMCTSNode:
    """
    MCTS node for search over DAG wave completions (FastDriveCoT-MCTS).

    Each node represents a partially-completed CoT: some DAG waves have been
    decoded, the rest are still pending.  Branching happens at the wave level —
    each child holds a different sampled completion of the *next* wave.
    """

    def __init__(self, field_tokens, wave_index=0, parent=None):
        self.field_tokens = field_tokens
        self.wave_index = wave_index
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

    def ucb_score(self, total_visits, explor_const=1.0):
        if self.visits == 0:
            return float("inf")
        exploitation = self.value / self.visits
        exploration = explor_const * math.sqrt(math.log(total_visits) / self.visits)
        return exploitation + exploration

    def is_terminal(self, waves):
        return self.wave_index >= len(waves)

    def clone_tokens(self):
        return {k: list(v) for k, v in self.field_tokens.items()}

    def as_text(self, vertices, processor):
        lines = []
        for v in vertices:
            toks = [t for t in self.field_tokens.get(v, []) if t != 0]
            if toks:
                text = processor.decode(toks, skip_special_tokens=True)
                lines.append(f"{v}: {text}")
        return "\n".join(lines)
