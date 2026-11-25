"""Parse final model metadata without accepting model text as executable tools."""

import json


def parse_final_response(content: object) -> tuple[list[str], str | None]:
    if not isinstance(content, str):
        raise ValueError("Model content must be text")
    # Some local templates leak a thinking prefix into content, occasionally
    # without an opening tag. Only the answer after the explicit end tag counts.
    answer = content.rsplit("</think>", 1)[-1].strip()
    if answer.startswith("```json\n") and answer.endswith("```"):
        answer = answer[len("```json\n"):-3].strip()
    elif answer.startswith("```\n") and answer.endswith("```"):
        answer = answer[4:-3].strip()
    final = json.loads(answer)
    if not isinstance(final, dict) or set(final) not in (
        {"missing_information"}, {"missing_information", "reply"},
    ):
        raise ValueError("Unexpected final output")
    missing = final["missing_information"]
    if not isinstance(missing, list) or not all(isinstance(item, str) for item in missing):
        raise ValueError("Invalid missing information")
    reply = final.get("reply")
    if "reply" in final and (not isinstance(reply, str) or not reply.strip()):
        raise ValueError("Reply must be non-empty text")
    return missing, reply.strip() if reply is not None else None


def parse_missing_information(content: object) -> list[str]:
    """Compatibility helper for metadata-only responses."""
    return parse_final_response(content)[0]
