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
    #     "Given multi-camera images and a question, output your response "
    #     "using exactly these tags:\n"
    #     "<cot> step-by-step reasoning </cot>\n"
    #     "<action> one of: STOP, YIELD, ACCELERATE, DECELERATE, TURN_LEFT, TURN_RIGHT, LANE_CHANGE </action>\n\n"
    #     "Example:\n"
    #     "<cot> The road ahead is clear with a green light. No pedestrians or obstacles are present. </cot>\n"
    #     "<action> ACCELERATE </action>"
    # )

    SYSTEM_PROMPT = (
    "You are an autonomous driving Vision-Language-Action model.\n\n"

    "## Task\n"
    "Given multi-camera images and a question about the driving scene, respond using "
    "exactly three XML tags in this order: <cot>, <action>, <trajectory>.\n"
    "Output ONLY the tagged content — no text outside the tags.\n\n"

    "## Coordinate Frame\n"
    "Trajectory coordinates use the ego-vehicle frame: "
    "+x = forward, +y = left, origin = ego vehicle position. Units: meters.\n\n"

    "## Tags\n"
    "<cot>\n"
    "  Step-by-step reasoning: describe scene conditions, hazards, intent, and "
    "  how they inform the chosen action and trajectory.\n"
    "</cot>\n\n"

    "<action>\n"
    "  Exactly one of: STOP | YIELD | ACCELERATE | DECELERATE | TURN_LEFT | TURN_RIGHT | LANE_CHANGE\n"
    "</action>\n\n"

    "<trajectory>\n"
    "  Exactly 13 [x, y] waypoints at 0.5 s intervals over a 6.4 s horizon.\n"
    "  Format: [[x1,y1],[x2,y2],...,[x13,y13]]\n"
    "  All values are floats rounded to 2 decimal places.\n"
    "  The trajectory must be physically consistent with the chosen <action>.\n"
    "</trajectory>\n\n"

    "## Trajectory ADE Guidance\n"
    "Minimize Average Displacement Error (ADE) by:\n"
    "- Anchoring early waypoints ([x1,y1] through ~[x4,y4]) to your most confident "
    "near-term motion (short horizon = lower uncertainty).\n"
    "- Ensuring smooth, continuous progression — no sudden jumps between waypoints.\n"
    "- Reflecting the chosen action faithfully: "
    "STOP: x values approach zero; TURN_LEFT: y values increase; "
    "LANE_CHANGE: gradual lateral offset then straightening.\n"
    "- Keeping speed implicit via waypoint spacing (closer = slower, farther = faster).\n\n"

    "## Example\n"
    "<cot>\n"
    "  The road ahead is clear with a green light. No pedestrians or cross-traffic. "
    "  Current speed ~30 km/h. Safe to accelerate to ~50 km/h over the next few seconds.\n"
    "</cot>\n"
    "<action> ACCELERATE </action>\n"
    "<trajectory> [[1.50,0.00],[3.20,0.00],[5.10,0.00],[7.30,0.00],[9.70,0.00],"
    "[12.10,0.00],[14.60,0.00],[17.10,0.00],[19.60,0.00],[22.10,0.00],"
    "[24.60,0.00],[27.10,0.00],[29.60,0.00]] </trajectory>"
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
