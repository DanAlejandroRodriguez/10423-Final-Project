"""
preprocess.py
=============
DriveLM feature extraction, tokenisation, and prompting helpers.

This module provides:
  - Prompt formatting to enforce a strict Data Contract (PromptFormatter).
  - Image pre-processing transforms compatible with Gemma 4's vision encoder.
  - Text tokenisation helpers that wrap the Gemma 4 tokeniser.
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
        "<action> one of: STOP, YIELD, ACCELERATE, DECELERATE, TURN_LEFT, TURN_RIGHT, LANE_CHANGE </action>\n"
        "<trajectory> [[x1,y1],[x2,y2],...] over a 6.4-second horizon </trajectory>"
    )

    @staticmethod
    def format(question: str) -> list:
        """
        Returns a structured conversation list for Gemma 4's native chat template.
        """
        # Note: The <image> tag is implicitly mapped to standard tokens by
        # Gemma's chat template processing when images are passed inside the list
        return [
            {"role": "system", "content": PromptFormatter.SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Question: {question}"}]}
        ]
