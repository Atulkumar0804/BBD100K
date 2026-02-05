"""BDD100K object detection parser with clean dataclasses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

DETECTION_CLASSES: List[str] = [
    "person",
    "car",
    "bus",
    "truck",
    "bike",
    "motor",
    "traffic light",
    "traffic sign",
    "train",
    "rider",
]


@dataclass
class BoundingBox:
    """Single 2D bounding box for an object."""

    class_name: str
    x1: int
    y1: int
    x2: int
    y2: int
    area: int
    occluded: bool
    truncated: bool


@dataclass
class ImageAnnotation:
    """Annotations for one image in the dataset."""

    image_name: str
    width: int
    height: int
    objects: List[BoundingBox]
    weather: str
    scene: str
    timeofday: str


def _safe_int(value: float | int | None) -> int:
    """Convert numeric values to int with safe fallback."""

    if value is None:
        return 0
    return int(round(float(value)))


def _default_resolution() -> Tuple[int, int]:
    """Return the default BDD100K resolution."""

    return (1280, 720)


def _extract_resolution(image_entry: dict) -> Tuple[int, int]:
    """Extract image resolution if present, else fallback."""

    attributes = image_entry.get("attributes") or {}
    width = attributes.get("width")
    height = attributes.get("height")
    if width is None or height is None:
        return _default_resolution()
    return (_safe_int(width), _safe_int(height))


def parse_bdd_json(
    json_path: Path | str,
    allowed_classes: Optional[Sequence[str]] = None,
) -> List[ImageAnnotation]:
    """Parse a BDD100K JSON label file into dataclasses.

    Args:
        json_path: Path to BDD100K labels JSON file.
        allowed_classes: Optional list of allowed classes.

    Returns:
        List of ImageAnnotation objects.
    """

    json_path = Path(json_path)
    if allowed_classes is None:
        allowed_classes = DETECTION_CLASSES
    allowed_set = set(allowed_classes)

    with json_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    annotations: List[ImageAnnotation] = []

    for image_entry in raw:
        image_name = image_entry.get("name", "")
        width, height = _extract_resolution(image_entry)

        # Extract image attributes
        attrs = image_entry.get("attributes", {})
        weather = attrs.get("weather", "undefined")
        scene = attrs.get("scene", "undefined")
        timeofday = attrs.get("timeofday", "undefined")

        objects: List[BoundingBox] = []

        for label in image_entry.get("labels", []) or []:
            category = label.get("category")
            if category not in allowed_set:
                continue

            box = label.get("box2d")
            if not box:
                continue

            # Extract object attributes
            obj_attrs = label.get("attributes", {})
            occluded = obj_attrs.get("occluded", False)
            truncated = obj_attrs.get("truncated", False)

            x1 = _safe_int(box.get("x1"))
            y1 = _safe_int(box.get("y1"))
            x2 = _safe_int(box.get("x2"))
            y2 = _safe_int(box.get("y2"))

            width_box = max(0, x2 - x1)
            height_box = max(0, y2 - y1)
            area = width_box * height_box

            bbox = BoundingBox(
                class_name=category,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                area=area,
                occluded=occluded,
                truncated=truncated,
            )
            objects.append(bbox)

        annotations.append(
            ImageAnnotation(
                image_name=image_name,
                width=width,
                height=height,
                objects=objects,
                weather=weather,
                scene=scene,
                timeofday=timeofday,
            )
        )

    return annotations


def iter_all_objects(annotations: Iterable[ImageAnnotation]) -> Iterable[BoundingBox]:
    """Yield all BoundingBox entries from annotations."""

    for image in annotations:
        for obj in image.objects:
            yield obj
