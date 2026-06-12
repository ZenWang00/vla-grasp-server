from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PIL import Image

RGB_IMAGE_LAYOUT = "The model receives exactly one RGB scene image."

PLAN_B_PROMPT_TEMPLATE = (
    "Role:\n"
    "You are an expert in Embodied AI and robot vision. Your task is to analyze one RGB image and provide precise grasp target localization for a robotic arm.\n\n"
    "Image description:\n"
    "Image details:\n"
    "- You are given a single RGB image with size {w} (width) x {h} (height) pixels.\n"
    "- The image contains only color appearance information; no separate depth map or stitched image is provided.\n"
    "- You must infer the most stable parallel-jaw grasp region using only object appearance, shape contours, occlusion relationships, and common sense.\n\n"
    "Reasoning task:\n"
    "Instructions:\n"
    "1. Target identification: find the object corresponding to \"{task_spec}\" in the image.\n"
    "2. Target localization: first output the full visible region of the target object as an object box, and provide an object point that lies inside the main body of the target object.\n"
    "3. Geometric inference: infer a suitable local grasp area from the visible contours, main structure, and contact stability in the RGB image.\n"
    "4. Grasp region selection: propose {num_candidates} candidate local grasp regions, not the full object box. Rank candidates from most to least recommended. Prefer main-body or middle sections, thick and stable parts, unoccluded areas, and approximately parallel contact bands.\n"
    "5. Must avoid: handles, spouts, edges, tips, weak joints, high-curvature regions, and areas that are likely to slip.\n"
    "6. Coordinate output: output both the full object box and the local grasp region box; do not confuse them.\n\n"
    "Output format requirements:\n"
    "Strict requirements:\n"
    "- Use normalized coordinates, with coordinate values ranging from 0 to 1000.\n"
    "- Your coordinates must map directly to this RGB image coordinate system, where x and y correspond to the full image width and height.\n"
    "- The top-level `object_box` must cover the full visible region of the target object; do not box only the local grasp area.\n"
    "- The top-level `object_point` must lie inside the main body of the target object, preferably far from boundaries and occlusions.\n"
    "- Each candidate `grasp_region_box` must be as small as possible while fully covering the stable area where the gripper would make contact; do not include the whole object.\n"
    "- Each candidate `grasp_point` must lie inside its corresponding `grasp_region_box`, preferably near the center of that local grasp region.\n"
    "- Candidates should cover different but reasonable grasp parts to improve robustness; the first candidate should be your most recommended option.\n"
    "- The output format must be JSON:\n"
    '{{"target": "{task_spec}", "object_box": [ymin, xmin, ymax, xmax], "object_point": [y, x], "candidates": [{{"rank": 1, "grasp_region_box": [ymin, xmin, ymax, xmax], "grasp_point": [y, x], "reasoning": "Briefly explain the grasp logic"}}]}}\n'
)



ALIGN_POINT_MULTI_PROMPT_TEMPLATE = (
    "Role:\n"
    "You are an expert in Embodied AI and robot vision. Your task is to analyze one RGB image and "
    "propose {num_candidates} DIVERSE candidate alignment/cut-in points for a robotic parallel-jaw gripper, "
    "ranked from most to least preferred.\n\n"
    "Image description:\n"
    "- You are given a single RGB image with size {w} (width) x {h} (height) pixels.\n"
    "- The image contains only color appearance information; no depth map is provided.\n"
    "- The robot already has a separate depth sensor; you must only locate the 2D points and the "
    "in-image gripper orientations. The 3D distance is recovered later from depth.\n\n"
    "Reasoning task:\n"
    "1. Target identification: find the object/rail corresponding to \"{task_spec}\" in the image.\n"
    "2. Diversity planning (do this BEFORE choosing coordinates): mentally divide the target object "
    "into distinct spatial zones (e.g. upper / middle / lower section, left side / right side, "
    "narrow end / wide end). Assign each candidate to a DIFFERENT zone so that no two candidates "
    "share the same body region. Also vary the gripper approach: at least one candidate should have "
    "a substantially different `gripper_angle_deg` (≥30° difference) from the others when the "
    "object geometry allows it.\n"
    "3. Candidate alignment points: for each zone chosen in step 2, output one `align_point` [y, x] "
    "at a stable, unoccluded contact band. "
    "Avoid handles, spouts, edges, tips, weak joints and high-curvature regions.\n"
    "4. Table / support-surface clearance: the gripper fingers close around the chosen point and "
    "extend past it, so the point needs free space around it. Do NOT place a point on the lowest "
    "part of the object where it meets the table or another supporting surface — the fingers would "
    "collide with the table. Keep every `align_point` clearly above the contact line between the "
    "object and its support; prefer the middle or upper section when the geometry allows it.\n"
    "5. Gripper angle: for each candidate, output `gripper_angle_deg`, the in-image rotation (degrees) "
    "of the gripper's closing direction. 0 means fingers close along the image +x (horizontal) axis; "
    "positive rotates toward image +y. Choose the angle so the jaws close across the narrow dimension.\n"
    "6. Self-check: before finalising, verify that no two candidates are within 80 normalized units "
    "of each other in `align_point`, and that every candidate keeps the table clearance required in "
    "step 4. If any pair is too close, move one to a more distant zone.\n\n"
    "Output format requirements (server-enforced — violations cause a hard parse error):\n"
    "- `align_point` must be [y, x] — exactly 2 numeric elements, no more, no less.\n"
    "- Normalized coordinate range is 0 to 1000 for both axes (0 = top/left edge, 1000 = bottom/right edge). "
    "Values outside [0, 1000] are out-of-bounds and will be rejected.\n"
    "- `align_point` must land on a solid, visible surface of the target object — not on the background, "
    "air, a transparent region, or a reflective surface that has no real depth. "
    "The server samples depth in a 5×5 pixel window around the point; if no valid depth exists there "
    "the candidate is rejected. Prefer the thickest, most opaque part of the object.\n"
    "- `gripper_angle_deg` must be a plain number (int or float). Do not wrap it in quotes or a list.\n"
    "- `candidates` must be a non-empty JSON array even when only one candidate is requested.\n"
    "- Each candidate's `reasoning` must name its body zone and explain how it differs from the "
    "other candidates (location, angle, or contact geometry).\n"
    "- The output format must be JSON:\n"
    '{{"target": "{task_spec}", "candidates": [{{"rank": 1, "align_point": [y, x], '
    '"gripper_angle_deg": <number>, '
    '"reasoning": "Zone: <zone name>. <Why this point and how it differs from others.>"}}]}}\n'
)


def build_align_prompt_multi(task_spec: str, w: int, h: int, num_candidates: int) -> str:
    """Prompt for the multi-candidate alignment-point flow."""
    return ALIGN_POINT_MULTI_PROMPT_TEMPLATE.format(
        task_spec=task_spec.strip(), w=w, h=h, num_candidates=num_candidates
    )


def build_grasp_task_spec(task_spec: str) -> str:
    """Rewrite a generic object query into a local-grasp-region query."""
    text = task_spec.strip()
    lower = text.lower()
    grasp_keywords = (
        "grasp",
        "graspable",
        "grasp region",
        "grasp point",
        "mid-body",
        "contact band",
        "stable region",
    )
    if any(keyword in lower for keyword in grasp_keywords):
        return text
    return (
        f"the best local grasp region on {text}, specifically the stable mid-body contact band "
        "for a parallel-jaw gripper, excluding handles, spouts, edges, and thin parts"
    )


def build_grounding_prompt(task_spec: str, w: int, h: int, num_candidates: int = 1) -> str:
    return PLAN_B_PROMPT_TEMPLATE.format(
        task_spec=build_grasp_task_spec(task_spec),
        w=w,
        h=h,
        num_candidates=num_candidates,
    )


def build_grounding_messages(
    task_spec: str,
    w: int,
    h: int,
    images: Sequence[Image.Image],
    num_candidates: int = 3,
) -> list[dict[str, Any]]:
    if len(images) != 1:
        raise ValueError("Exactly one RGB image is required")
    text = build_grounding_prompt(task_spec, w, h, num_candidates=num_candidates)
    content: list[dict[str, Any]] = [{"type": "image", "image": images[0]}]
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]
