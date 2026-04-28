"""
hybrid.py — Adaptive hybrid model that switches between MCTS and FastDriveCoT
based on scene complexity (number of critical objects).

Strategy:
  - Simple scenes (few critical objects): Use MCTS for deeper, more deliberate
    search over the action space since the scene is tractable.
  - Complex scenes (many critical objects): Use FastDriveCoT's parallel DAG
    decoding to quickly reason over many interacting objects simultaneously.
"""

from .fastdrive import FastDriveVLA


class HybridVLA(FastDriveVLA):
    """Adaptive model that routes to MCTS or FastDriveCoT based on scene complexity.

    Inherits from FastDriveVLA which itself inherits from QwenBaselineVLA,
    giving access to both ``mcts_generate`` and the DAG parallel decoding.

    Parameters
    ----------
    model_id : str
        Hugging Face model identifier.
    object_threshold : int
        If the number of critical objects in a scene is strictly greater than
        this value, use FastDriveCoT. Otherwise, use MCTS.
        Default is 3.
    """

    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct", object_threshold=5):
        super().__init__(model_id=model_id)
        self.object_threshold = object_threshold

    def generate_trajectory_hybrid(self, images, text_prompt, num_critical_objects=0, max_new_tokens=512):
        """Route to MCTS or FastDriveCoT based on scene complexity.

        Parameters
        ----------
        images : list[PIL.Image]
            The 6 camera images for this frame.
        text_prompt : list[dict]
            Pre-formatted messages from ``PromptFormatter.format()``.
        num_critical_objects : int
            Number of critical objects in the current scene (from the dataset's
            ``key_object_infos``). Used to decide which strategy to employ.
        max_new_tokens : int
            Maximum number of tokens to generate.

        Returns
        -------
        dict
            Parsed result with ``raw_text``, ``meta_action``, ``trajectory``,
            ``latency_seconds``, and ``model_type``.
        """
        if num_critical_objects < self.object_threshold:
            # Simple scene — use baseline
            # result = super().generate_trajectory_parallel(images, text_prompt, max_new_tokens=max_new_tokens)
            result = self.generate_trajectory(images, text_prompt, max_new_tokens=max_new_tokens)
            result["routing_decision"] = "baseline"
            result["routing_reason"] = (
                f"{num_critical_objects} critical objects < threshold {self.object_threshold}"
            )
        else:
            # Complex scene — use MCTS for deeper search 
            result = self.mcts_generate(images, text_prompt, max_new_tokens=max_new_tokens)
            result["routing_decision"] = "MCTS"
            result["routing_reason"] = (
                f"{num_critical_objects} critical objects >= threshold {self.object_threshold}"
            )

        result["model_type"] = "Qwen2.5VL_Hybrid_MCTS_FastDriveCoT"
        result["num_critical_objects"] = num_critical_objects
        return result
