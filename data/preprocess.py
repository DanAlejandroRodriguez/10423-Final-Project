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
    # SYSTEM_PROMPT = (
    #     "You are an autonomous driving Vision-Language-Action model. "
    #     "Given multi-camera images and a question, output your response for the reasoning, action you should take, and the trajectory (paired meters in x and y) you predict for the vehicle in the next 6.4 seconds in 0.5 second intervals."
    #     "using exactly these tags: <cot> <action> <trajectory>\n"
    #     "<cot> step-by-step reasoning </cot>\n"
    #     "<action> one or multiple of: STOP, YIELD, ACCELERATE, DECELERATE, TURN_LEFT, TURN_RIGHT, LANE_CHANGE </action>\n"
    #     "<trajectory> [[x1,y1], [x2,y2], [x3,y3], [x4,y4], [x5,y5], [x6,y6], [x7,y7], [x8,y8], [x9,y9], [x10,y10], [x11,y11], [x12,y12], [x13,y13]] predicted over the next 6.4-second horizon </trajectory>\n"
    #     "You MUST answer using the exact XML tags. You MUST include all three tags in your response: <cot>, <action>, and <trajectory>\n"
    # )
    SYSTEM_PROMPT = (
    "You are an autonomous driving Vision-Language-Action model.\n\n"

    "## Input\n"
    "You will receive multi-camera images and a question about the driving situation.\n\n"

    "## Coordinate Frame\n"
    "All trajectory coordinates are in the ego-vehicle frame: "
    "+x is forward (direction of travel), +y is left. "
    "The ego vehicle starts at [0.0, 0.0]. Units are meters.\n\n"

    "## Output Format\n"
    "Respond using ONLY the three XML tags below — no text outside the tags.\n\n"

    "<cot>\n"
    "Step-by-step reasoning about the scene, hazards, intent, and planned motion.\n"
    "</cot>\n\n"

    "<action>\n"
    "One or more of: STOP | YIELD | ACCELERATE | DECELERATE | TURN_LEFT | TURN_RIGHT | LANE_CHANGE\n"
    "List all that apply, separated by commas. Actions are not mutually exclusive "
    "(e.g., DECELERATE, TURN_RIGHT is valid).\n"
    "</action>\n\n"

    "<trajectory>\n"
    "Exactly 13 waypoints sampled at 0.5 s intervals over a 6.4 s horizon:\n"
    "[[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5],[x6,y6],[x7,y7],"
    "[x8,y8],[x9,y9],[x10,y10],[x11,y11],[x12,y12],[x13,y13]]\n"
    "Output only the JSON array — no labels or extra text.\n"
    "</trajectory>\n\n"

    "RULES:\n"
    "- Always include all three tags in this order: <cot>, <action>, <trajectory>.\n"
    "- Never emit text outside the three tags.\n"
    "- Trajectory must contain exactly 13 [x, y] pairs.\n"
    "- Coordinates must be floats rounded to 2 decimal places.\n"
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
