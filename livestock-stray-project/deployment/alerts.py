"""
Alert rules for 牛马占道 (livestock straying into public areas).
Customize thresholds and notification targets per deployment.
"""
from datetime import datetime

ALERT_RULES = {
    "livestock-stray": {
        "classes": ["cattle", "horse"],
        "zone_crossing": True,      # alert when animal enters defined road zone
        "min_confidence": 0.25,
        "cooldown_seconds": 30,     # suppress repeated alerts within 30s
        "notify": {
            "webhook": None,        # set to webhook URL for WeChat/DingTalk alerts
            "log_file": "./alerts.log",
        },
    },
}

# Optional: define road/public-area polygon zones
# Coordinates are normalized [0-1] relative to image width/height
# Replace with actual zone coordinates from your camera setup
ZONE_POLYGONS = {
    "road_zone": [
        [0.0, 0.4],   # top-left
        [1.0, 0.4],   # top-right
        [1.0, 1.0],   # bottom-right
        [0.0, 1.0],   # bottom-left
    ],
}


def point_in_polygon(px: float, py: float, polygon: list) -> bool:
    """Ray-casting algorithm for point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def bbox_in_zone(bbox: list, zone_name: str = "road_zone",
                 img_w: int = 640, img_h: int = 640) -> bool:
    """
    Check if the center of a bounding box falls within a zone polygon.
    bbox: [x1, y1, x2, y2] in pixel coordinates
    """
    polygon = ZONE_POLYGONS.get(zone_name)
    if not polygon:
        return False
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    return point_in_polygon(cx, cy, polygon)


def should_alert(detections: list, img_w: int = 640, img_h: int = 640) -> dict:
    """
    Evaluate detections against alert rules.
    Returns {"alert": bool, "reason": str, "animals": list}
    """
    rules = ALERT_RULES["livestock-stray"]
    target_classes = set(rules["classes"])
    animals_in_zone = [
        d for d in detections
        if d["class"] in target_classes
        and bbox_in_zone(d["bbox"], img_w=img_w, img_h=img_h)
    ]

    if animals_in_zone:
        return {
            "alert": True,
            "reason": f"{len(animals_in_zone)} animal(s) detected in road zone",
            "animals": animals_in_zone,
            "timestamp": datetime.now().isoformat(),
        }
    return {"alert": False, "reason": "no animals in road zone", "animals": []}
