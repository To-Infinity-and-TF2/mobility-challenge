"""
ArtPark Arena Bot Navigation — OpenCV Pipeline
================================================
Camera: Front-facing, 0° inclination (flat view of floor logos)

Strategy:
  1. Detect AprilTags → if visible, use tag direction (highest priority)
  2. Detect the ArtPark logo's GREEN and ORANGE colour blobs
  3. Find centroids of each colour region
  4. Determine relative positions (front/back/side) to decide motion command

Logo orientation logic (camera looking straight down at logo):
  - GREEN centroid BELOW orange centroid  → green is "facing the bot" → FOLLOW GREEN
  - GREEN centroid ABOVE orange centroid  → green is "away from bot"  → FOLLOW ORANGE
  - Centroids roughly at same Y (side-by-side) → GO STRAIGHT

Motion commands: FORWARD | BACKWARD | TURN_LEFT | TURN_RIGHT | STOP | U_TURN
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# CONFIG — tune these HSV ranges on your actual camera feed
# ---------------------------------------------------------------------------

# ArtPark logo GREEN  (lime-green outer ring)
GREEN_HSV_LOW  = np.array([40,  80, 80],  dtype=np.uint8)
GREEN_HSV_HIGH = np.array([85, 255, 255], dtype=np.uint8)

# ArtPark logo ORANGE (orange outer ring)
ORANGE_HSV_LOW  = np.array([5,  120, 120], dtype=np.uint8)
ORANGE_HSV_HIGH = np.array([22, 255, 255], dtype=np.uint8)

# Minimum blob area to be considered a valid colour region (px²)
MIN_BLOB_AREA = 500

# How close centroids need to be (in Y) to call them "side-by-side"
SIDE_THRESHOLD_RATIO = 0.15   # fraction of frame height

# AprilTag family used on the arena
APRILTAG_FAMILY = "tag36h11"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ColourBlob:
    centroid: Tuple[int, int]   # (cx, cy) in pixels
    area: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class NavDecision:
    command: str                        # FORWARD / BACKWARD / TURN_LEFT / TURN_RIGHT / STOP / U_TURN
    reason: str                         # human-readable explanation
    active_colour: Optional[str]        # "GREEN" | "ORANGE" | None
    tag_id: Optional[int]               # AprilTag id if used


# ---------------------------------------------------------------------------
# Colour detection helpers
# ---------------------------------------------------------------------------

def detect_colour_blob(hsv_frame: np.ndarray,
                        low: np.ndarray,
                        high: np.ndarray,
                        label: str,
                        debug_frame: Optional[np.ndarray] = None
                        ) -> Optional[ColourBlob]:
    """Return the largest blob of a given HSV colour range, or None."""
    mask = cv2.inRange(hsv_frame, low, high)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < MIN_BLOB_AREA:
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    x, y, w, h = cv2.boundingRect(largest)

    if debug_frame is not None:
        colour_bgr = (0, 200, 0) if label == "GREEN" else (0, 140, 255)
        cv2.drawContours(debug_frame, [largest], -1, colour_bgr, 2)
        cv2.circle(debug_frame, (cx, cy), 8, colour_bgr, -1)
        cv2.putText(debug_frame, f"{label} ({area:.0f}px²)",
                    (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour_bgr, 2)

    return ColourBlob(centroid=(cx, cy), area=area, bbox=(x, y, w, h))


# ---------------------------------------------------------------------------
# AprilTag detection
# ---------------------------------------------------------------------------

def detect_apriltag(gray_frame: np.ndarray,
                    debug_frame: Optional[np.ndarray] = None
                    ) -> Optional[dict]:
    """
    Detect the nearest (largest) AprilTag.
    Returns dict with keys: id, corners, center, direction_hint
    Requires:  pip install pupil-apriltags
    """
    try:
        from pupil_apriltags import Detector
    except ImportError:
        # fallback: try opencv-contrib ArUco
        return _detect_aruco_fallback(gray_frame, debug_frame)

    detector = Detector(families=APRILTAG_FAMILY,
                        nthreads=2,
                        quad_decimate=1.0)
    detections = detector.detect(gray_frame)
    if not detections:
        return None

    # Pick the largest tag (closest to camera)
    best = max(detections, key=lambda d: _tag_area(d.corners))
    cx = int(best.center[0])
    cy = int(best.center[1])

    # Infer direction from tag ID (customize mapping for your arena)
    direction = _tag_id_to_direction(best.tag_id)

    if debug_frame is not None:
        pts = best.corners.astype(int)
        cv2.polylines(debug_frame, [pts], True, (255, 0, 255), 2)
        cv2.circle(debug_frame, (cx, cy), 6, (255, 0, 255), -1)
        cv2.putText(debug_frame, f"TAG {best.tag_id} → {direction}",
                    (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    return {"id": best.tag_id, "center": (cx, cy),
            "corners": best.corners, "direction_hint": direction}


def _detect_aruco_fallback(gray_frame, debug_frame):
    """OpenCV ArUco fallback if pupil-apriltags not installed."""
    aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    aruco_params = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners_list, ids, _ = detector.detectMarkers(gray_frame)

    if ids is None or len(ids) == 0:
        return None

    # Pick largest
    best_idx  = int(np.argmax([cv2.contourArea(c) for c in corners_list]))
    best_id   = int(ids[best_idx][0])
    best_corners = corners_list[best_idx][0]
    cx = int(best_corners[:, 0].mean())
    cy = int(best_corners[:, 1].mean())
    direction = _tag_id_to_direction(best_id)

    if debug_frame is not None:
        cv2.aruco.drawDetectedMarkers(debug_frame, corners_list, ids)
        cv2.putText(debug_frame, f"TAG {best_id} → {direction}",
                    (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    return {"id": best_id, "center": (cx, cy),
            "corners": best_corners, "direction_hint": direction}


def _tag_area(corners) -> float:
    pts = corners.astype(np.float32)
    return float(cv2.contourArea(pts))


def _tag_id_to_direction(tag_id: int) -> str:
    """
    Map AprilTag IDs to directional hints.
    Edit this table to match YOUR arena's tag placement:
      Waypoint 1 → FORWARD
      Waypoint 2 → TURN_LEFT (U-TURN section)
      Waypoint 3 → TURN_RIGHT
      Waypoint 4 → U_TURN
      Waypoint 5 → FORWARD (after left turn)
    """
    mapping = {
        1: "FORWARD",
        2: "U_TURN",      # top-right corner in reference map
        3: "TURN_RIGHT",
        4: "U_TURN",
        5: "FORWARD",
    }
    return mapping.get(tag_id, "FORWARD")


# ---------------------------------------------------------------------------
# Core navigation logic
# ---------------------------------------------------------------------------

def decide_navigation(green_blob: Optional[ColourBlob],
                       orange_blob: Optional[ColourBlob],
                       frame_height: int,
                       active_colour: str = "GREEN"   # "GREEN" or "ORANGE"
                       ) -> NavDecision:
    """
    Given detected blobs, decide what motion command to issue.

    active_colour: which colour we are currently supposed to follow
                   (GREEN for green-arrow tiles, ORANGE for orange-arrow tiles)
    """
    primary   = green_blob  if active_colour == "GREEN"  else orange_blob
    secondary = orange_blob if active_colour == "GREEN"  else green_blob
    sec_label = "ORANGE"    if active_colour == "GREEN"  else "GREEN"

    threshold = frame_height * SIDE_THRESHOLD_RATIO

    if primary is None and secondary is None:
        return NavDecision("FORWARD", "No blobs detected — continue straight", active_colour, None)

    if primary is None:
        # Lost the primary colour — perhaps rotated; use secondary as reference
        return NavDecision("FORWARD",
                           f"Lost {active_colour}, tracking {sec_label} as backup",
                           sec_label, None)

    if secondary is None:
        # Only primary visible — can't compare, go forward
        return NavDecision("FORWARD",
                           f"Only {active_colour} visible, advancing",
                           active_colour, None)

    primary_cy   = primary.centroid[1]
    secondary_cy = secondary.centroid[1]
    dy           = primary_cy - secondary_cy   # positive = primary is LOWER in frame

    # In a top-down camera: LOWER Y value = further from bot; HIGHER Y = closer to bot
    # Camera looks forward flat → image row 0 = far end, row max = near (bot's feet)
    # So: primary centroid with HIGHER row number = it is on the NEAR side = facing the bot

    if abs(dy) < threshold:
        # Centroids at roughly same height → colours are side by side → GO STRAIGHT
        return NavDecision("FORWARD",
                           f"{active_colour} & {sec_label} side-by-side → straight",
                           active_colour, None)
    elif dy > 0:
        # primary is LOWER (closer to bot) → it's facing us → FOLLOW (go forward)
        return NavDecision("FORWARD",
                           f"{active_colour} is closer side → advance",
                           active_colour, None)
    else:
        # primary is UPPER (far side) → we have overshot or wrong orientation → BACKWARD / recheck
        return NavDecision("BACKWARD",
                           f"{active_colour} is on far side → may have overshot",
                           active_colour, None)


# ---------------------------------------------------------------------------
# Per-frame processing
# ---------------------------------------------------------------------------

def process_frame(frame: np.ndarray,
                  active_colour: str = "GREEN"
                  ) -> Tuple[NavDecision, np.ndarray]:
    """
    Full pipeline for one camera frame.
    Returns (NavDecision, annotated_debug_frame).
    """
    debug = frame.copy()
    h, w  = frame.shape[:2]

    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. AprilTag check (highest priority)
    tag = detect_apriltag(gray, debug)
    if tag is not None:
        cmd = tag["direction_hint"]
        decision = NavDecision(cmd,
                               f"AprilTag {tag['id']} override → {cmd}",
                               active_colour,
                               tag["id"])
        _draw_decision(debug, decision)
        return decision, debug

    # 2. Colour blob detection
    green_blob  = detect_colour_blob(hsv, GREEN_HSV_LOW,  GREEN_HSV_HIGH,  "GREEN",  debug)
    orange_blob = detect_colour_blob(hsv, ORANGE_HSV_LOW, ORANGE_HSV_HIGH, "ORANGE", debug)

    # 3. Navigation decision
    decision = decide_navigation(green_blob, orange_blob, h, active_colour)
    _draw_decision(debug, decision)

    return decision, debug


def _draw_decision(frame: np.ndarray, decision: NavDecision):
    label  = f"CMD: {decision.command}  |  {decision.reason}"
    colour = (0, 255, 0) if decision.command == "FORWARD" else \
             (0, 0, 255) if decision.command in ("BACKWARD", "STOP") else \
             (0, 200, 255)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)


# ---------------------------------------------------------------------------
# HSV Calibration helper  (run once to tune your ranges)
# ---------------------------------------------------------------------------

def launch_hsv_tuner(source=0):
    """
    Interactive HSV tuner window.
    Press 'q' to quit, 's' to print current HSV ranges.
    """
    cap = cv2.VideoCapture(source)
    cv2.namedWindow("HSV Tuner")

    def nothing(x): pass
    for name, val in [("H_lo", 0),  ("H_hi", 179),
                      ("S_lo", 0),  ("S_hi", 255),
                      ("V_lo", 0),  ("V_hi", 255)]:
        cv2.createTrackbar(name, "HSV Tuner", val, 179 if "H" in name else 255, nothing)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo  = np.array([cv2.getTrackbarPos(n, "HSV Tuner") for n in ("H_lo","S_lo","V_lo")])
        hi  = np.array([cv2.getTrackbarPos(n, "HSV Tuner") for n in ("H_hi","S_hi","V_hi")])
        mask   = cv2.inRange(hsv, lo, hi)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow("HSV Tuner", np.hstack([frame, result]))

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('s'):
            print(f"LOW  = {lo.tolist()}")
            print(f"HIGH = {hi.tolist()}")

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(source=0, active_colour="GREEN"):
    """
    source       : camera index (0,1,2…) or video file path
    active_colour: starting colour to follow — update per waypoint
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    print("[INFO] Navigation started. Press 'q' to quit, 'g'/'o' to switch colour mode.")

    prev_cmd  = None
    cmd_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break

        decision, debug_frame = process_frame(frame, active_colour)

        # Debounce — only print when command changes
        if decision.command != prev_cmd:
            print(f"[NAV] {decision.command:12s} | {decision.reason}")
            prev_cmd  = decision.command
            cmd_count = 0
        else:
            cmd_count += 1

        # ── Send to robot here ──────────────────────────────────────────────
        # robot.send_command(decision.command)
        # ───────────────────────────────────────────────────────────────────

        cv2.imshow("ArtPark Navigation", debug_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            active_colour = "GREEN"
            print("[INFO] Switched to GREEN mode")
        elif key == ord('o'):
            active_colour = "ORANGE"
            print("[INFO] Switched to ORANGE mode")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    # Usage:
    #   python artpark_navigation.py                → webcam, green mode
    #   python artpark_navigation.py 1 ORANGE       → cam 1, orange mode
    #   python artpark_navigation.py video.mp4 GREEN
    #   python artpark_navigation.py tune           → HSV calibration tool

    if len(sys.argv) > 1 and sys.argv[1] == "tune":
        launch_hsv_tuner(0)
    else:
        src   = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() \
                else (sys.argv[1] if len(sys.argv) > 1 else 0)
        col   = sys.argv[2].upper() if len(sys.argv) > 2 else "GREEN"
        main(source=src, active_colour=col)