from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .io import rgb_to_pil


@dataclass(frozen=True)
class Sam2SegmentationResult:
    mask: np.ndarray
    segmap: np.ndarray
    model_name: str
    device: str
    positive_pixels: int


def _import_sam2_dependencies():
    try:
        import torch
        from transformers import Sam2Model, Sam2Processor
    except Exception as exc:
        raise RuntimeError(
            "SAM2 segmentation requires local dependencies in the project environment. "
            "Install or activate an environment with `torch` and `transformers` first."
        ) from exc
    return torch, Sam2Model, Sam2Processor


def _resolve_device(requested_device: str | None) -> str:
    torch, _, _ = _import_sam2_dependencies()
    if requested_device and requested_device.lower() != "auto":
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=4)
def _load_sam2_model(model_name: str, device: str):
    torch, Sam2Model, Sam2Processor = _import_sam2_dependencies()
    processor = Sam2Processor.from_pretrained(model_name)
    model = Sam2Model.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return processor, model, torch


def _extract_binary_mask(mask_data: object) -> np.ndarray:
    if hasattr(mask_data, "detach"):
        mask_array = mask_data.detach().cpu().numpy()
    else:
        mask_array = np.asarray(mask_data)
    while mask_array.ndim > 2:
        mask_array = mask_array[0]
    if mask_array.dtype == np.bool_:
        return mask_array.astype(np.uint8, copy=False)
    return (mask_array > 0).astype(np.uint8, copy=False)


def refine_mask_with_depth(
    mask: np.ndarray,
    depth: np.ndarray,
    *,
    object_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Placeholder hook for future depth-aware cleanup."""
    _ = depth, object_box
    return mask


def run_sam2_segmentation(
    rgb: np.ndarray,
    *,
    object_box: tuple[int, int, int, int],
    object_point: tuple[int, int] | None = None,
    model_name: str = "facebook/sam2.1-hiera-small",
    device: str | None = None,
    depth: np.ndarray | None = None,
) -> Sam2SegmentationResult:
    processor, model, torch = _load_sam2_model(model_name, _resolve_device(device))
    pil_image = rgb_to_pil(rgb)
    ymin, xmin, ymax, xmax = object_box

    input_boxes = [[[float(xmin), float(ymin), float(xmax), float(ymax)]]]
    input_points = None
    input_labels = None
    if object_point is not None:
        py, px = object_point
        input_points = [[[[float(px), float(py)]]]]
        input_labels = [[[1]]]

    inputs = processor(
        images=pil_image,
        input_boxes=input_boxes,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        binarize=True,
    )[0]
    mask = _extract_binary_mask(masks[0] if isinstance(masks, list) else masks)
    if depth is not None:
        mask = refine_mask_with_depth(mask, depth, object_box=object_box)
    segmap = np.zeros(mask.shape, dtype=np.uint8)
    segmap[mask.astype(bool)] = 1
    return Sam2SegmentationResult(
        mask=mask,
        segmap=segmap,
        model_name=model_name,
        device=str(model.device),
        positive_pixels=int(mask.astype(bool).sum()),
    )


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(str(path), format="PNG")


def save_mask_overlay_png(
    rgb: np.ndarray,
    mask: np.ndarray,
    path: Path,
    *,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.4,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = np.asarray(rgb, dtype=np.uint8).copy()
    mask_bool = mask.astype(bool)
    color_arr = np.array(color, dtype=np.float32)
    base_float = base.astype(np.float32)
    base_float[mask_bool] = (1.0 - alpha) * base_float[mask_bool] + alpha * color_arr
    Image.fromarray(np.clip(base_float, 0, 255).astype(np.uint8), mode="RGB").save(
        str(path), format="PNG"
    )
