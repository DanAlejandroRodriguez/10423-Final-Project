"""
Adaptive hybrid model that routes between FastDriveCoT and MCTSr
based on scene complexity (number of critical objects).
"""

from .fastdrive import FastDriveVLA


class HybridVLA(FastDriveVLA):

    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct", object_threshold=3):
        super().__init__(model_id=model_id)
        self.object_threshold = object_threshold

    def generate_trajectory_hybrid(self, images, text_prompt, num_critical_objects=0, mcts_iterations=5):
        if num_critical_objects < self.object_threshold:
            result = self.generate_trajectory_parallel(images=images, text_prompt=text_prompt)
            result["routing_decision"] = "FastDriveCoT"
        else:
            result = self.mcts_fastdrive_generate(
                images=images, text_prompt=text_prompt, iterations=mcts_iterations,
            )
            result["routing_decision"] = "MCTSr"

        result["model_type"] = "Qwen2.5VL_Hybrid"
        result["num_critical_objects"] = num_critical_objects
        return result
