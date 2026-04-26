"""
preprocess.py
=============
DriveLM feature extraction, tokenisation, and prompting helpers.

This module provides:
  - Prompt formatting to enforce a strict Data Contract (PromptFormatter).
  - Image pre-processing transforms compatible with Qwen2.5-VL's vision encoder.
  - Text tokenisation helpers that wrap the Qwen2.5-VL tokeniser.
  - Scene-difficulty estimation used by the adaptive-compute router.
  - Trajectory normalisation / de-normalisation utilities.
"""

class PromptFormatter:
    """
    Enforces a single output format across all models so the evaluator
    can always parse <cot>, <action>, and <trajectory> tags regardless
    of which model is under test.
    """
    SYSTEM_PROMPT = (
        "You are an autonomous driving Vision-Language-Action model. "
        "Given multi-camera images and a question, output your response "
        "using exactly these tags:\n"
        "<cot> step-by-step reasoning </cot>\n"
        "<action> one of: STOP, YIELD, ACCELERATE, DECELERATE, TURN_LEFT, TURN_RIGHT, LANE_CHANGE </action>\n\n"
        "Examples:\n\n"
        "Example 1:\n"
        "<cot> The road ahead is clear with a green light. No pedestrians or obstacles are present. </cot>\n"
        "<action> ACCELERATE </action>\n\n"
        "Example 2:\n"
        "<cot> A pedestrian is crossing at the intersection and the traffic light is red. The ego vehicle must stop. </cot>\n"
        "<action> STOP </action>\n\n"
        "Example 3:\n"
        "<cot> The vehicle ahead is braking. There is a stop sign visible. The ego vehicle should slow down. </cot>\n"
        "<action> DECELERATE </action>"
    )

    CAMERA_LABELS = [
        "front camera", "front-left camera", "front-right camera",
        "back camera", "back-left camera", "back-right camera",
    ]

    @staticmethod
    def format(question, images=None):
        user_content = []

        if images:
            for i, img in enumerate(images):
                label = PromptFormatter.CAMERA_LABELS[i] if i < len(PromptFormatter.CAMERA_LABELS) else f"camera {i}"
                user_content.append({"type": "text", "text": f"[{label}]"})
                user_content.append({"type": "image", "image": img})

        user_content.append({"type": "text", "text": f"Question: {question}"})

        return [
            {
                "role": "system",
                "content": PromptFormatter.SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
