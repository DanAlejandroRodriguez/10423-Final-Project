import torch
import fastdrive.FastDriveVLA

class DagScheduler():    
    def __init__(self, prompt, inputs, model, vertices, edges, max_lengths, ancestor_mask=None):
        self.prefix = prompt
        self.inputs = inputs
        self.model = model
        self.vertices = vertices
        self.max_lengths = max_lengths
        self.ancestor_mask = ancestor_mask

        # initialize set of incoming edges per vertex
        self.num_incoming_edges = {name: 0 for name in vertices}
        self.edges = edges
        for (_, b) in edges:
            self.num_incoming_edges[b] += 1

        # initialize set of ready vertices
        self.S = []
        for name, num_edges in self.num_incoming_edges:
            if num_edges == 0:
                self.S.append(name)

    def run_parallel_decoding(self):
        final_tokens = []

        # algorithm in fastdrive CoT paper
        while len(self.S) != 0:
            # decode a new token in each vertex v in S in parallel
            # TODO: construct input_ids array
            outputs_logits = self.model.parallel_forward_pass(input_ids, self.max_lengths, self.ancestor_mask)
            # TODO: extract the last word with the highest probability
            final_tokens.append()
            for ind, v in enumerate(self.S):
                # TODO: check field corresponding to vertex v is done
                if v == 'done':
                    self.S.remove(v)

                    for (a, b) in self.edges:
                        if a == v:
                            self.num_incoming_edges[b] -= 1
                            if self.num_incoming_edges[b] == 0:
                                # add node to S
                                self.S.append(b)
        return final_tokens