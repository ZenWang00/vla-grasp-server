from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PIL import Image

RGB_IMAGE_LAYOUT = "The model receives exactly one RGB scene image."

PLAN_B_PROMPT_TEMPLATE = (
    "角色设定：\n"
    "你是一个精通具身智能（Embodied AI）和机器人视觉的专家。你的任务是分析一张 RGB 图像，并为机械臂提供精准的抓取目标定位。\n\n"
    "图像结构描述：\n"
    "图像说明：\n"
    "- 你看到的是一张单独的 RGB 图像，尺寸为 {w} (宽) x {h} (高) 像素。\n"
    "- 图中只包含彩色外观信息，不包含单独提供的深度图或拼接图。\n"
    "- 你必须仅根据物体外观、形状轮廓、遮挡关系和常识来推断最稳定的平行夹爪抓取区域。\n\n"
    "推理任务：\n"
    "执行指令：\n"
    "1. 识别目标：请在图中找到“{task_spec}”对应的物体。\n"
    "2. 目标定位：请先输出该目标物体在图中的整体可见区域框（object box），并给出一个落在目标物体主体内部的目标点（object point）。\n"
    "3. 几何推断：根据 RGB 图中的可见轮廓、主体结构和接触稳定性，推断适合抓取的局部区域。\n"
    "4. 抓取区域选择：请提出 {num_candidates} 个候选局部抓取区域，而不是整个物体框。候选需要按推荐优先级从高到低排序。优先选择主干/中部、厚实、稳定、无遮挡、近似平行的接触带。\n"
    "5. 必须避开：把手、壶嘴、边缘、尖端、薄弱连接处、剧烈曲率区域，以及容易滑脱的区域。\n"
    "6. 坐标回传：请同时输出整物体框（object box）和局部抓取区域框（grasp region box），二者不要混淆。\n\n"
    "输出格式要求：\n"
    "严格要求：\n"
    "- 请使用归一化坐标格式输出，即坐标值在 0 到 1000 之间。\n"
    "- 你的坐标输出必须直接映射到这张 RGB 图像的坐标系内，即 x 和 y 都对应整张 RGB 图像的宽高范围。\n"
    "- 顶层 `object_box` 必须覆盖目标物体在图中的整体可见区域；不要只框抓取局部。\n"
    "- 顶层 `object_point` 必须落在目标物体主体内部，尽量远离边界和遮挡。\n"
    "- 每个候选的 `grasp_region_box` 必须尽可能小，但要完整覆盖夹爪实际接触的稳定区域；不要把整个物体都框进去。\n"
    "- 每个候选的 `grasp_point` 必须落在对应的 `grasp_region_box` 内，并尽量位于该局部抓取区域中心。\n"
    "- 候选之间尽量给出不同但都合理的抓取部位，以提升容错；第一个候选应是你最推荐的方案。\n"
    "- 输出格式必须为 JSON：\n"
    '{{"target": "{task_spec}", "object_box": [ymin, xmin, ymax, xmax], "object_point": [y, x], "candidates": [{{"rank": 1, "grasp_region_box": [ymin, xmin, ymax, xmax], "grasp_point": [y, x], "reasoning": "简述抓取逻辑"}}]}}\n'
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


def build_grounding_prompt(task_spec: str, w: int, h: int, num_candidates: int = 3) -> str:
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
