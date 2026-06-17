#!/usr/bin/env python3

import threading
from pathlib import Path
import os
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from dataclasses import dataclass, field
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image
from typing import Optional, Tuple, List
from ament_index_python.packages import get_package_share_directory

# ---------------------------------------------------------------------------
# SIMULATION CAMERA TOPICS
# ---------------------------------------------------------------------------
RGB_TOPIC   = '/r1_mini/camera/image_raw'
DEPTH_TOPIC = '/r1_mini/depth_cam/image_raw'

# ---------------------------------------------------------------------------
# TEMPLATE MATCHING CONFIG
# ---------------------------------------------------------------------------

# Path to the ArtPark logo image — put it next to this script or give full path
# The logo should be a clean top-down crop with transparent/white background

pkg_path=get_package_share_directory('mini_r1_v1_description')
TEMPLATE_PATH =os.path.join()

# Confidence threshold (0–1). Above = bot is ON a tile. Tune this.
# Start at 0.55 and increase if you get false positives in your sim
TEMPLATE_MATCH_THRESHOLD = 0.55

# Template is checked at multiple scales to handle perspective/distance changes
# (min_scale, max_scale, step) — covers the logo looking bigger/smaller as bot approaches
TEMPLATE_SCALES = np.arange(0.3, 1.2, 0.1)

# How many consecutive frames below threshold before we call it "tile exited"
# Prevents a single noisy frame from triggering a false exit
EXIT_DEBOUNCE_FRAMES = 8

# ---------------------------------------------------------------------------
# VELOCITY CONSTANTS
# ---------------------------------------------------------------------------
LINEAR_SPEED_FWD  =  0.2
LINEAR_SPEED_BWD  = -0.15
ANGULAR_TURN      =  0.4
ANGULAR_U_TURN    =  0.8

# ---------------------------------------------------------------------------
# HSV COLOUR RANGES
# ---------------------------------------------------------------------------
GREEN_HSV_LOW   = np.array([40,  80,  80],  dtype=np.uint8)
GREEN_HSV_HIGH  = np.array([85, 255, 255],  dtype=np.uint8)

ORANGE_HSV_LOW  = np.array([5,  120, 120],  dtype=np.uint8)
ORANGE_HSV_HIGH = np.array([22, 255, 255],  dtype=np.uint8)

MIN_BLOB_AREA        = 500
SIDE_THRESHOLD_RATIO = 0.15

# ---------------------------------------------------------------------------
# AprilTag ID → (linear.x, angular.z)
# ---------------------------------------------------------------------------
TAG_TWIST_MAP = {
    1: (LINEAR_SPEED_FWD, 0.0),
    2: (0.0,  ANGULAR_U_TURN),
    3: (0.0, -ANGULAR_TURN),
    4: (0.0,  ANGULAR_U_TURN),
    5: (LINEAR_SPEED_FWD, 0.0),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ColourBlob:
    centroid:   Tuple[int, int]
    area:       float
    distance_m: Optional[float] = None


@dataclass
class TemplateResult:
    on_tile:    bool             # True if currently over an ArtPark tile
    confidence: float            # best match score this frame (0–1)
    match_loc:  Optional[Tuple[int, int]] = None   # top-left of best match in frame
    match_size: Optional[Tuple[int, int]] = None   # (w, h) of matched template scale


@dataclass
class TileTracker:
    """Tracks tile enter/exit events across frames."""
    tiles_passed:       int  = 0       # total tiles fully exited
    on_tile:            bool = False   # currently over a tile?
    tile_done:          bool = False   # rose True for ONE frame when tile exits
    _below_thresh_count: int = field(default=0, repr=False)   # debounce counter

    def update(self, result: TemplateResult) -> None:
        """
        Call once per frame with the latest TemplateResult.
        Sets self.tile_done = True for exactly one frame when the bot
        fully exits a tile (falling edge with debounce).
        """
        self.tile_done = False   # reset every frame

        if result.on_tile:
            # Rising edge — bot just entered or is on a tile
            if not self.on_tile:
                pass   # could log "entered tile" here if useful
            self.on_tile = True
            self._below_thresh_count = 0

        else:
            # Below threshold — count consecutive frames
            if self.on_tile:
                self._below_thresh_count += 1
                if self._below_thresh_count >= EXIT_DEBOUNCE_FRAMES:
                    # Confirmed exit — tile is done
                    self.on_tile             = False
                    self._below_thresh_count = 0
                    self.tiles_passed       += 1
                    self.tile_done           = True   # one-frame pulse


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------
class ArtParkNavNode(Node):

    def __init__(self):
        super().__init__('artpark_nav_node')

        self.declare_parameter('active_colour', 'GREEN')
        self.declare_parameter('show_debug',    True)
        self.declare_parameter('template_path', str(TEMPLATE_PATH))

        self.active_colour = self.get_parameter('active_colour').value.upper()
        self.show_debug    = self.get_parameter('show_debug').value

        self._depth_lock  = threading.Lock()
        self._depth_frame: Optional[np.ndarray] = None

        self._debug_lock  = threading.Lock()
        self._debug_frame: Optional[np.ndarray] = None

        self.bridge = CvBridge()

        # Tile tracking state
        self.tile_tracker = TileTracker()

        # Load template once
        tmpl_path = self.get_parameter('template_path').value
        self._template_gray, self._template_scales_cache = self._load_template(tmpl_path)

        # AprilTag detector — created once
        self._apriltag_detector = None
        self._init_apriltag_detector()

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_subscription(Image, RGB_TOPIC,   self._rgb_cb,   10)
        self.create_subscription(Image, DEPTH_TOPIC, self._depth_cb, 10)

        if self.show_debug:
            self.create_timer(1.0 / 30.0, self._display_timer_cb)

        self.get_logger().info(
            f'\nArtPark Nav Node started'
            f'\n  RGB   : {RGB_TOPIC}'
            f'\n  Depth : {DEPTH_TOPIC}'
            f'\n  Mode  : {self.active_colour}'
            f'\n  Tmpl  : {tmpl_path} '
            f'({"loaded" if self._template_gray is not None else "NOT FOUND — matching disabled"})'
        )

    # ------------------------------------------------------------------
    # Template loading
    # ------------------------------------------------------------------
    def _load_template(
        self, path: str
    ) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
        """
        Load the logo image and pre-build all scaled grayscale versions.
        Returns (base_gray_template, [scaled_gray, ...])
        If file not found, returns (None, []) and matching is skipped.
        """
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            self.get_logger().warn(
                f'Template not found at {path} — tile counting disabled.\n'
                f'Place artpark_logo.png next to this script.'
            )
            return None, []

        # If RGBA (PNG with alpha), composite onto white background
        if img.shape[2] == 4:
            alpha   = img[:, :, 3:4].astype(np.float32) / 255.0
            rgb     = img[:, :, :3].astype(np.float32)
            white   = np.ones_like(rgb) * 255.0
            img_bgr = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)
        else:
            img_bgr = img[:, :, :3]

        base_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Pre-build scaled versions so we don't resize every frame
        scaled_cache: List[np.ndarray] = []
        h, w = base_gray.shape
        for scale in TEMPLATE_SCALES:
            sw, sh = int(w * scale), int(h * scale)
            if sw < 10 or sh < 10:
                continue
            scaled_cache.append(cv2.resize(base_gray, (sw, sh)))

        self.get_logger().info(
            f'Template loaded: {w}×{h}px, {len(scaled_cache)} scales pre-built'
        )
        return base_gray, scaled_cache

    # ------------------------------------------------------------------
    # Template matching — multi-scale
    # ------------------------------------------------------------------
    def _run_template_match(self, frame_gray: np.ndarray) -> TemplateResult:
        """
        Slide the pre-scaled templates over the grayscale frame.
        Returns the best match across all scales.
        """
        if self._template_gray is None or not self._template_scales_cache:
            # Template not loaded — always report "not on tile"
            return TemplateResult(on_tile=False, confidence=0.0)

        fh, fw = frame_gray.shape
        best_conf  = 0.0
        best_loc   = None
        best_size  = None

        for tmpl in self._template_scales_cache:
            th, tw = tmpl.shape
            if th > fh or tw > fw:
                continue   # template bigger than frame at this scale — skip

            result = cv2.matchTemplate(frame_gray, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_conf:
                best_conf = max_val
                best_loc  = max_loc
                best_size = (tw, th)

        on_tile = best_conf >= TEMPLATE_MATCH_THRESHOLD

        return TemplateResult(
            on_tile    = on_tile,
            confidence = best_conf,
            match_loc  = best_loc  if on_tile else None,
            match_size = best_size if on_tile else None,
        )

    # ------------------------------------------------------------------
    # Depth callback
    # ------------------------------------------------------------------
    def _depth_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            with self._depth_lock:
                self._depth_frame = frame
        except Exception as e:
            self.get_logger().warn(f'Depth decode error: {e}')

    # ------------------------------------------------------------------
    # RGB callback
    # ------------------------------------------------------------------
    def _rgb_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB decode error: {e}')
            return

        self.active_colour = self.get_parameter('active_colour').value.upper()

        with self._depth_lock:
            depth_snapshot = (
                self._depth_frame.copy() if self._depth_frame is not None else None
            )

        twist, debug_frame = self._process_frame(frame, depth_snapshot)
        self.cmd_pub.publish(twist)

        if self.show_debug and debug_frame is not None:
            with self._debug_lock:
                self._debug_frame = debug_frame

    # ------------------------------------------------------------------
    # Display timer
    # ------------------------------------------------------------------
    def _display_timer_cb(self):
        with self._debug_lock:
            frame = self._debug_frame
        if frame is not None:
            cv2.imshow('ArtPark Nav [sim]', frame)
            cv2.waitKey(1)

    # ------------------------------------------------------------------
    # Full per-frame pipeline
    # ------------------------------------------------------------------
    def _process_frame(
        self,
        frame: np.ndarray,
        depth: Optional[np.ndarray],
    ) -> Tuple[Twist, Optional[np.ndarray]]:

        debug = frame.copy() if self.show_debug else None
        h, w  = frame.shape[:2]

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Template matching ──────────────────────────────────────────
        tmpl_result = self._run_template_match(gray)
        self.tile_tracker.update(tmpl_result)

        if self.tile_tracker.tile_done:
            self.get_logger().info(
                f'✓ Tile DONE — total tiles passed: {self.tile_tracker.tiles_passed}'
            )

        if debug is not None:
            self._draw_tile_info(debug, tmpl_result)

        # ── AprilTag — highest priority ────────────────────────────────
        tag = self._detect_apriltag(gray, debug)
        if tag is not None:
            twist = self._tag_to_twist(tag['id'])
            self.get_logger().info(
                f"[TAG {tag['id']}] lin={twist.linear.x:.2f} ang={twist.angular.z:.2f}"
            )
            if debug is not None:
                self._draw_overlay(debug, f"APRILTAG {tag['id']}", twist)
            return twist, debug

        # ── Colour blobs ───────────────────────────────────────────────
        green_blob  = self._detect_blob(hsv, GREEN_HSV_LOW,  GREEN_HSV_HIGH,  'GREEN',  debug, depth)
        orange_blob = self._detect_blob(hsv, ORANGE_HSV_LOW, ORANGE_HSV_HIGH, 'ORANGE', debug, depth)

        # ── Twist decision ─────────────────────────────────────────────
        twist = self._blobs_to_twist(green_blob, orange_blob, h)

        # ── Depth obstacle guard ───────────────────────────────────────
        if depth is not None and twist.linear.x > 0:
            twist = self._apply_depth_guard(twist, depth)

        if debug is not None:
            self._draw_overlay(debug, f'COLOUR:{self.active_colour}', twist)

        return twist, debug

    # ------------------------------------------------------------------
    # Draw tile tracker info onto debug frame
    # ------------------------------------------------------------------
    def _draw_tile_info(self, frame: np.ndarray, result: TemplateResult):
        h, w = frame.shape[:2]

        # Draw match rectangle if on tile
        if result.on_tile and result.match_loc and result.match_size:
            x, y   = result.match_loc
            mw, mh = result.match_size
            cv2.rectangle(frame, (x, y), (x + mw, y + mh), (255, 255, 0), 2)
            cv2.putText(frame, f'TILE MATCH {result.confidence:.2f}',
                        (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Tile count — bottom-left corner
        status_colour = (0, 255, 255) if self.tile_tracker.on_tile else (180, 180, 180)
        on_str  = 'ON TILE' if self.tile_tracker.on_tile else 'between tiles'
        done_str = '  ← DONE!' if self.tile_tracker.tile_done else ''
        cv2.putText(
            frame,
            f'Tiles passed: {self.tile_tracker.tiles_passed}  [{on_str}]{done_str}',
            (8, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_colour, 2,
        )

    # ------------------------------------------------------------------
    # Colour blob detection
    # ------------------------------------------------------------------
    def _detect_blob(
        self,
        hsv:   np.ndarray,
        low:   np.ndarray,
        high:  np.ndarray,
        label: str,
        debug: Optional[np.ndarray],
        depth: Optional[np.ndarray] = None,
    ) -> Optional[ColourBlob]:

        mask = cv2.inRange(hsv, low, high)
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area    = cv2.contourArea(largest)
        if area < MIN_BLOB_AREA:
            return None

        M = cv2.moments(largest)
        if M['m00'] == 0:
            return None

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # Sample depth at centroid
        distance_m: Optional[float] = None
        if depth is not None:
            dh, dw = depth.shape[:2]
            sx = int(cx * dw / (debug.shape[1] if debug is not None else dw))
            sy = int(cy * dh / (debug.shape[0] if debug is not None else dh))
            sx = max(0, min(sx, dw - 1))
            sy = max(0, min(sy, dh - 1))
            patch = depth[max(0, sy-2):sy+3, max(0, sx-2):sx+3]
            valid = patch[np.isfinite(patch)]
            if valid.size > 0:
                distance_m = float(np.median(valid))

        if debug is not None:
            colour_bgr = (0, 220, 0) if label == 'GREEN' else (0, 140, 255)
            cv2.drawContours(debug, [largest], -1, colour_bgr, 2)
            cv2.circle(debug, (cx, cy), 9, colour_bgr, -1)
            dist_text = f'{distance_m:.2f}m' if (distance_m and np.isfinite(distance_m)) else f'{area:.0f}px'
            cv2.putText(debug, f'{label} {dist_text}',
                        (cx + 12, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour_bgr, 2)

        return ColourBlob(centroid=(cx, cy), area=area, distance_m=distance_m)

    # ------------------------------------------------------------------
    # Blob geometry → Twist
    # ------------------------------------------------------------------
    def _blobs_to_twist(
        self,
        green:   Optional[ColourBlob],
        orange:  Optional[ColourBlob],
        frame_h: int,
    ) -> Twist:

        twist     = Twist()
        primary   = green  if self.active_colour == 'GREEN'  else orange
        secondary = orange if self.active_colour == 'GREEN'  else green
        threshold = frame_h * SIDE_THRESHOLD_RATIO

        if primary is None and secondary is None:
            twist.linear.x = LINEAR_SPEED_FWD * 0.5
            self.get_logger().warn('No blobs — creeping forward')
            return twist

        if primary is None:
            twist.linear.x = LINEAR_SPEED_FWD * 0.4
            self.get_logger().warn(f'Lost {self.active_colour} — creeping')
            return twist

        if secondary is None:
            twist.linear.x = LINEAR_SPEED_FWD
            return twist

        dy = primary.centroid[1] - secondary.centroid[1]

        if abs(dy) < threshold:
            twist.linear.x = LINEAR_SPEED_FWD
            self.get_logger().info('Side-by-side → STRAIGHT')
        elif dy > 0:
            twist.linear.x = LINEAR_SPEED_FWD
            self.get_logger().info(f'{self.active_colour} near-side → FORWARD')
        else:
            twist.linear.x = LINEAR_SPEED_BWD
            self.get_logger().warn(f'{self.active_colour} far-side → BACKWARD')

        return twist

    # ------------------------------------------------------------------
    # Depth obstacle guard
    # ------------------------------------------------------------------
    def _apply_depth_guard(self, twist: Twist, depth: np.ndarray) -> Twist:
        h, w  = depth.shape[:2]
        strip = depth[int(h * 0.4):int(h * 0.7), int(w * 0.4):int(w * 0.6)]
        strip = np.where(np.isfinite(strip), strip, 10.0)
        min_d = float(np.nanmin(strip))
        if min_d < 0.3:
            twist.linear.x  = 0.0
            twist.angular.z = 0.0
            self.get_logger().warn(f'Obstacle at {min_d:.2f} m — STOP')
        return twist

    # ------------------------------------------------------------------
    # AprilTag detection
    # ------------------------------------------------------------------
    def _init_apriltag_detector(self):
        try:
            from pupil_apriltags import Detector
            self._apriltag_detector = Detector(
                families='tag36h11', nthreads=2, quad_decimate=1.0
            )
            self.get_logger().info('AprilTag detector: pupil_apriltags')
        except ImportError:
            self._apriltag_detector = None
            self.get_logger().warn('pupil_apriltags not found — using ArUco fallback')

    def _detect_apriltag(
        self, gray: np.ndarray, debug: Optional[np.ndarray]
    ) -> Optional[dict]:

        if self._apriltag_detector is not None:
            try:
                detections = self._apriltag_detector.detect(gray)
                if detections:
                    best = max(
                        detections,
                        key=lambda d: cv2.contourArea(d.corners.astype(np.float32)),
                    )
                    cx, cy = int(best.center[0]), int(best.center[1])
                    if debug is not None:
                        pts = best.corners.astype(int)
                        cv2.polylines(debug, [pts], True, (255, 0, 255), 2)
                        cv2.circle(debug, (cx, cy), 6, (255, 0, 255), -1)
                        cv2.putText(debug, f'TAG {best.tag_id}',
                                    (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (255, 0, 255), 2)
                    return {'id': best.tag_id, 'center': (cx, cy)}
            except Exception as e:
                self.get_logger().warn(f'AprilTag error: {e}')
            return None

        try:
            aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            aruco_params = cv2.aruco.DetectorParameters()
            det          = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            corners_list, ids, _ = det.detectMarkers(gray)
            if ids is None:
                return None
            best_idx     = int(np.argmax([cv2.contourArea(c) for c in corners_list]))
            best_id      = int(ids[best_idx][0])
            best_corners = corners_list[best_idx][0]
            cx = int(best_corners[:, 0].mean())
            cy = int(best_corners[:, 1].mean())
            if debug is not None:
                cv2.aruco.drawDetectedMarkers(debug, corners_list, ids)
                cv2.putText(debug, f'TAG {best_id}',
                            (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            return {'id': best_id, 'center': (cx, cy)}
        except Exception as e:
            self.get_logger().warn(f'ArUco error: {e}')
            return None

    def _tag_to_twist(self, tag_id: int) -> Twist:
        twist = Twist()
        lin, ang = TAG_TWIST_MAP.get(tag_id, (LINEAR_SPEED_FWD, 0.0))
        twist.linear.x  = lin
        twist.angular.z = ang
        return twist

    # ------------------------------------------------------------------
    # Debug overlay
    # ------------------------------------------------------------------
    def _draw_overlay(self, frame: np.ndarray, label: str, twist: Twist):
        text = (f'{label} | lin={twist.linear.x:+.2f} m/s  '
                f'ang={twist.angular.z:+.2f} rad/s')
        colour = (
            (0, 255,   0) if twist.linear.x > 0 else
            (0,   0, 255) if twist.linear.x < 0 else
            (0, 200, 255)
        )
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(frame, text, (8, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = ArtParkNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = Twist()
        node.cmd_pub.publish(stop)
        node.get_logger().info('Safety stop published. Shutting down.')
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()