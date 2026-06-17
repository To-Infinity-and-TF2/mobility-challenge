#!/usr/bin/env python3
"""
ArtPark Arena Bot Navigation — ROS2 Node (Simulation)
=====================================================
Transform chain built here:
    world (static, broadcast once)
      └── odom          ← /odom
            └── base_link       ← bot pose from odom
                  └── camera_link     ← static, forward-facing at bot centre
                        └── tag_<id>  ← estimated each detection via PnP

AprilTag → Action mapping (36h11, tag size 0.175 m):
    Tag 0 → TURN RIGHT  (goal: 1 tile to the RIGHT  of tag)
    Tag 1 → TURN LEFT   (goal: 1 tile to the LEFT   of tag)  [stub]
    Tag 2 → FOLLOW GREEN                                       [stub]
    Tag 3 → U-TURN      (goal: 1 tile BEHIND        of tag)  [stub]
    Tag 4 → FOLLOW ORANGE                                      [stub]

Goal execution (two-waypoint system):
    1. Detect tag → estimate pose in camera frame via solvePnP
    2. Compute WP1 = bot_pos + APPROACH_DIST (1.5 m) in bot's current heading
    3. Compute WP2 = WP1 + TILE_LENGTH in the tag-action direction
       (derived from TF2 world goal, or fallback nominal angle)
    4. Drive straight to WP1 (P-controller on heading, LiDAR guard)
    5. At WP1: compute delta_yaw to face WP2, check clearance, rotate

Clearance guard (pre-rotation):
    Before rotating, LiDAR dual-check (narrow cone + broad arc) confirms
    >= TILE_LENGTH free space in the goal direction.
    If not clear → creep forward, re-check at 20 Hz.
"""

import math
import threading
import time
from enum import Enum, auto
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import cv2
import numpy as np

import rclpy
import rclpy.parameter
from rclpy.node import Node
from cv_bridge import CvBridge

from geometry_msgs.msg import Twist, TransformStamped, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan, CameraInfo

import tf2_ros
import tf2_geometry_msgs                          # registers PoseStamped transforms
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster, Buffer, TransformListener
from tf_transformations import (
    quaternion_from_euler, euler_from_quaternion, quaternion_multiply
)


ARUCO_LOG_PATH = Path(__file__).parent / 'aruco_detections.log'


# ---------------------------------------------------------------------------
# TOPICS
# ---------------------------------------------------------------------------
RGB_TOPIC         = '/r1_mini/camera/image_raw'
DEPTH_TOPIC       = '/r1_mini/depth_cam/image_raw'
LIDAR_TOPIC       = '/r1_mini/scan'
ODOM_TOPIC        = '/odom'
CAMERA_INFO_TOPIC = '/r1_mini/camera/camera_info'
CMD_TOPIC         = '/cmd_vel'

# ---------------------------------------------------------------------------
# TF FRAME IDs
# ---------------------------------------------------------------------------
FRAME_WORLD       = 'world'
FRAME_ODOM        = 'odom'
FRAME_BASE_LINK   = 'base_link'
FRAME_CAMERA      = 'camera_link'

# Static camera offset relative to base_link
# Adjust if mount is not perfectly centred
CAM_OFFSET_X =  0.05   # m  forward
CAM_OFFSET_Y =  0.00
CAM_OFFSET_Z =  0.10   # m  above base
CAM_ROLL     =  0.0
CAM_PITCH    =  0.0
CAM_YAW      =  0.0

# ---------------------------------------------------------------------------
# APRILTAG CONFIG
# ---------------------------------------------------------------------------
TAG_SIZE_M    = 0.175   # physical side length of tags in arena
TAG_FAMILY    = 'tag36h11'

# Half-size corners in tag-local frame (z=0 plane, centred)
_h = TAG_SIZE_M / 2.0
TAG_OBJ_POINTS = np.array([
    [-_h,  _h, 0],
    [ _h,  _h, 0],
    [ _h, -_h, 0],
    [-_h, -_h, 0],
], dtype=np.float64)

# ---------------------------------------------------------------------------
# VELOCITY CONSTANTS
# ---------------------------------------------------------------------------
LINEAR_SPEED_FWD =  0.20   # m/s  — colour following
LINEAR_SPEED_BWD = -0.15
ANGULAR_TURN     =  0.50   # rad/s — rotational actions
CREEP_SPEED      =  0.08   # m/s  — pre-turn clearance creep

# ---------------------------------------------------------------------------
# TILE / CLEARANCE
# ---------------------------------------------------------------------------
TILE_LENGTH        = 0.90   # metres
CLEARANCE_CONE_DEG = 15.0   # ±deg narrow cone
CREEP_CHECK_DT     = 0.05   # s  (20 Hz)
CLEARANCE_TIMEOUT  = 10.0   # s

# ---------------------------------------------------------------------------
# ACTION CONFIG
# ---------------------------------------------------------------------------
TURN_RIGHT_DEG   = -90.0
TURN_LEFT_DEG    =  90.0
U_TURN_DEG       =  180.0
TAG_COOLDOWN_SEC =  3.0
APPROACH_DIST    =  1.5    # m — bot drives this far forward before turning

# ---------------------------------------------------------------------------
# WAYPOINT DRIVE CONFIG
# ---------------------------------------------------------------------------
WP_POS_TOL    = 0.10   # m   — arrival tolerance for waypoint
WP_TIMEOUT    = 15.0   # s   — give up driving to waypoint after this
WP_CHECK_DT   = 0.05   # s   — control loop period
WP_YAW_GAIN   = 2.0    # P-gain on heading error during wp drive

# ---------------------------------------------------------------------------
# LIDAR SAFETY
# ---------------------------------------------------------------------------
LIDAR_STOP_DIST    = 0.25
LIDAR_SECTOR_FWD   = ( -30,  30)
LIDAR_SECTOR_RIGHT = (-120, -30)
LIDAR_SECTOR_LEFT  = (  30, 120)
LIDAR_SECTOR_UTURN = ( 150, 210)

# Clearance map: turn_deg → (cone_centre_deg, broad_sector)
TURN_CLEARANCE_MAP = {
    TURN_RIGHT_DEG : ( -90.0, LIDAR_SECTOR_RIGHT),
    TURN_LEFT_DEG  : (  90.0, LIDAR_SECTOR_LEFT ),
    U_TURN_DEG     : ( 180.0, LIDAR_SECTOR_UTURN),
}

# ---------------------------------------------------------------------------
# TEMPLATE MATCHING
# ---------------------------------------------------------------------------
TEMPLATE_PATH            = Path(__file__).parent / 'artpark_logo.png'
TEMPLATE_MATCH_THRESHOLD = 0.55
TEMPLATE_SCALES          = np.arange(0.3, 1.2, 0.1)
EXIT_DEBOUNCE_FRAMES     = 8

# ---------------------------------------------------------------------------
# HSV COLOUR RANGES
# ---------------------------------------------------------------------------
GREEN_HSV_LOW   = np.array([40,  80,  80],  dtype=np.uint8)
GREEN_HSV_HIGH  = np.array([85, 255, 255],  dtype=np.uint8)
ORANGE_HSV_LOW  = np.array([ 5, 120, 120],  dtype=np.uint8)
ORANGE_HSV_HIGH = np.array([22, 255, 255],  dtype=np.uint8)
MIN_BLOB_AREA        = 500
SIDE_THRESHOLD_RATIO = 0.15


# ===========================================================================
# State machine
# ===========================================================================
class BotState(Enum):
    FOLLOWING = auto()
    CLEARING  = auto()   # driving to WP1 / waiting for turn clearance
    TURNING   = auto()   # executing yaw rotation toward WP2
    COOLDOWN  = auto()


# ===========================================================================
# Data classes
# ===========================================================================
@dataclass
class ColourBlob:
    centroid:   Tuple[int, int]
    area:       float
    distance_m: Optional[float] = None

@dataclass
class TemplateResult:
    on_tile:    bool
    confidence: float
    match_loc:  Optional[Tuple[int, int]] = None
    match_size: Optional[Tuple[int, int]] = None

@dataclass
class TileTracker:
    tiles_passed:        int  = 0
    on_tile:             bool = False
    tile_done:           bool = False
    _below_thresh_count: int  = field(default=0, repr=False)

    def update(self, result: TemplateResult) -> None:
        self.tile_done = False
        if result.on_tile:
            self.on_tile = True
            self._below_thresh_count = 0
        else:
            if self.on_tile:
                self._below_thresh_count += 1
                if self._below_thresh_count >= EXIT_DEBOUNCE_FRAMES:
                    self.on_tile             = False
                    self._below_thresh_count = 0
                    self.tiles_passed       += 1
                    self.tile_done           = True


# ===========================================================================
# ROS2 Node
# ===========================================================================
class ArtParkNavNode(Node):

    def __init__(self):
        super().__init__('artpark_nav_node')

        self.declare_parameter('active_colour', 'GREEN')
        self.declare_parameter('show_debug',    True)
        self.declare_parameter('template_path', str(TEMPLATE_PATH))

        self.active_colour = self.get_parameter('active_colour').value.upper()
        self.show_debug    = self.get_parameter('show_debug').value

        # ── State ──────────────────────────────────────────────────────
        self.bot_state       = BotState.FOLLOWING
        self._state_lock     = threading.Lock()
        self._cooldown_until = 0.0

        # ── Sensor snapshots ───────────────────────────────────────────
        self._depth_lock  = threading.Lock()
        self._depth_frame: Optional[np.ndarray] = None

        self._lidar_lock = threading.Lock()
        self._lidar_msg: Optional[LaserScan] = None

        self._cam_info_lock = threading.Lock()
        self._camera_matrix: Optional[np.ndarray] = None   # 3×3
        self._dist_coeffs:   Optional[np.ndarray] = None   # 1×5

        self._debug_lock  = threading.Lock()
        self._debug_frame: Optional[np.ndarray] = None

        # ── TF2 ───────────────────────────────────────────────────────
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Broadcaster for dynamic transforms (odom → base_link each odom msg)
        self._tf_broadcaster = TransformBroadcaster(self)

        # Static broadcaster: world → odom (once) + base_link → camera_link (once)
        self._static_broadcaster = StaticTransformBroadcaster(self)
        self._broadcast_static_transforms()

        # ── Subsystems ─────────────────────────────────────────────────
        self.bridge       = CvBridge()
        self.tile_tracker = TileTracker()

        tmpl_path = self.get_parameter('template_path').value
        self._template_gray, self._template_scales_cache = self._load_template(tmpl_path)

        self._apriltag_detector = None
        self._init_apriltag_detector()
        def _log_aruco(self, tag_id: int, tvec: np.ndarray, confidence: float = 0.0):
            """
            Append a detection entry to aruco_detections.log.
            Format:
                [2026-04-18 12:34:56.789] TAG=0  dist=1.23m  tvec=(x, y, z)  action=TURN RIGHT
            """
            ACTION_LABELS = {
                0: 'TURN RIGHT',
                1: 'TURN LEFT',
                2: 'FOLLOW GREEN',
                3: 'U-TURN',
                4: 'FOLLOW ORANGE',
            }
            action   = ACTION_LABELS.get(tag_id, f'UNKNOWN TAG {tag_id}')
            dist_m   = float(np.linalg.norm(tvec))
            tx, ty, tz = float(tvec[0]), float(tvec[1]), float(tvec[2])
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S') + f'.{int(time.time() % 1 * 1000):03d}'

            line = (
                f'[{timestamp}]  '
                f'TAG={tag_id}  '
                f'action={action:<14}  '
                f'dist={dist_m:.3f}m  '
                f'tvec=({tx:+.3f}, {ty:+.3f}, {tz:+.3f})\n'
            )

            try:
                with open(ARUCO_LOG_PATH, 'a') as f:
                    f.write(line)
            except Exception as e:
                self.get_logger().warn(f'ArUco log write failed: {e}')

            self.get_logger().info(f'[ARUCO LOG] {line.strip()}')

            # ── Publisher / Subscribers ────────────────────────────────────
            self.cmd_pub = self.create_publisher(Twist, CMD_TOPIC, 10)

            self.create_subscription(Image,      RGB_TOPIC,         self._rgb_cb,      10)
            self.create_subscription(Image,      DEPTH_TOPIC,       self._depth_cb,    10)
            self.create_subscription(LaserScan,  LIDAR_TOPIC,       self._lidar_cb,    10)
            self.create_subscription(Odometry,   ODOM_TOPIC,        self._odom_cb,     10)
            self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self._cam_info_cb,  1)

            if self.show_debug:
                self.create_timer(1.0 / 30.0, self._display_timer_cb)

            self.get_logger().info(
                f'\nArtPark Nav Node started'
                f'\n  Odom    : {ODOM_TOPIC}'
                f'\n  CamInfo : {CAMERA_INFO_TOPIC}'
                f'\n  Mode    : {self.active_colour}'
            )

    # ======================================================================
    # Static TF: world → odom  and  base_link → camera_link
    # ======================================================================

    def _broadcast_static_transforms(self):
        now = self.get_clock().now().to_msg()
        transforms = []

        # ── world → odom (identity — world origin = odom origin at start) ──
        t_world = TransformStamped()
        t_world.header.stamp         = now
        t_world.header.frame_id      = FRAME_WORLD
        t_world.child_frame_id       = FRAME_ODOM
        t_world.transform.rotation.w = 1.0   # identity quaternion
        transforms.append(t_world)

        # ── base_link → camera_link (forward-facing, centred) ─────────
        t_cam = TransformStamped()
        t_cam.header.stamp         = now
        t_cam.header.frame_id      = FRAME_BASE_LINK
        t_cam.child_frame_id       = FRAME_CAMERA
        t_cam.transform.translation.x = CAM_OFFSET_X
        t_cam.transform.translation.y = CAM_OFFSET_Y
        t_cam.transform.translation.z = CAM_OFFSET_Z
        q = quaternion_from_euler(CAM_ROLL, CAM_PITCH, CAM_YAW)
        t_cam.transform.rotation.x = q[0]
        t_cam.transform.rotation.y = q[1]
        t_cam.transform.rotation.z = q[2]
        t_cam.transform.rotation.w = q[3]
        transforms.append(t_cam)

        self._static_broadcaster.sendTransform(transforms)
        self.get_logger().info(
            'Static TFs broadcast: world→odom, base_link→camera_link'
        )

    # ======================================================================
    # Sensor callbacks
    # ======================================================================

    def _odom_cb(self, msg: Odometry):
        """Re-broadcast odom → base_link so TF2 can chain the full tree."""
        t = TransformStamped()
        t.header.stamp         = msg.header.stamp
        t.header.frame_id      = FRAME_ODOM
        t.child_frame_id       = FRAME_BASE_LINK
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation       = msg.pose.pose.orientation
        self._tf_broadcaster.sendTransform(t)

    def _cam_info_cb(self, msg: CameraInfo):
        with self._cam_info_lock:
            if self._camera_matrix is None:
                self._camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
                self._dist_coeffs   = np.array(msg.d, dtype=np.float64)
                self.get_logger().info(
                    f'Camera intrinsics received: '
                    f'fx={self._camera_matrix[0,0]:.1f} '
                    f'fy={self._camera_matrix[1,1]:.1f}'
                )

    def _depth_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            with self._depth_lock:
                self._depth_frame = frame
        except Exception as e:
            self.get_logger().warn(f'Depth decode error: {e}')

    def _lidar_cb(self, msg: LaserScan):
        with self._lidar_lock:
            self._lidar_msg = msg

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
        with self._lidar_lock:
            lidar_snapshot = self._lidar_msg
        with self._cam_info_lock:
            K    = self._camera_matrix.copy() if self._camera_matrix is not None else None
            dist = self._dist_coeffs.copy()   if self._dist_coeffs   is not None else None

        twist, debug_frame = self._process_frame(
            frame, depth_snapshot, lidar_snapshot, K, dist
        )

        with self._state_lock:
            current_state = self.bot_state
            if current_state == BotState.COOLDOWN and time.time() > self._cooldown_until:
                self.bot_state = BotState.FOLLOWING
                current_state  = BotState.FOLLOWING
                self.get_logger().info('Cooldown complete — resuming FOLLOWING')

        if current_state == BotState.FOLLOWING:
            self.cmd_pub.publish(twist)

        if self.show_debug and debug_frame is not None:
            with self._debug_lock:
                self._debug_frame = debug_frame

    # ======================================================================
    # Display timer
    # ======================================================================

    def _display_timer_cb(self):
        with self._debug_lock:
            frame = self._debug_frame
        if frame is not None:
            cv2.imshow('ArtPark Nav [sim]', frame)
            cv2.waitKey(1)

    # ======================================================================
    # Main per-frame pipeline
    # ======================================================================

    def _process_frame(
        self,
        frame:  np.ndarray,
        depth:  Optional[np.ndarray],
        lidar:  Optional[LaserScan],
        K:      Optional[np.ndarray],
        dist:   Optional[np.ndarray],
    ) -> Tuple[Twist, Optional[np.ndarray]]:

        debug = frame.copy() if self.show_debug else None
        h, w  = frame.shape[:2]
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Tile tracking ──────────────────────────────────────────────
        tmpl_result = self._run_template_match(gray)
        self.tile_tracker.update(tmpl_result)
        if self.tile_tracker.tile_done:
            self.get_logger().info(
                f'✓ Tile DONE — total: {self.tile_tracker.tiles_passed}'
            )
        if debug is not None:
            self._draw_tile_info(debug, tmpl_result)

        # ── AprilTag detection (FOLLOWING state only) ──────────────────
        with self._state_lock:
            can_act = self.bot_state == BotState.FOLLOWING

        if can_act and K is not None:
            tag = self._detect_apriltag_with_pose(gray, K, dist, debug)
            if tag is not None:
                threading.Thread(
                    target=self._execute_tag_action,
                    args=(tag['id'], tag['tvec'], tag['rvec'], K, dist),
                    daemon=True,
                ).start()

        # ── Colour blobs ───────────────────────────────────────────────
        green_blob  = self._detect_blob(
            hsv, GREEN_HSV_LOW,  GREEN_HSV_HIGH,  'GREEN',  debug, depth)
        orange_blob = self._detect_blob(
            hsv, ORANGE_HSV_LOW, ORANGE_HSV_HIGH, 'ORANGE', debug, depth)
        twist = self._blobs_to_twist(green_blob, orange_blob, h)
        if depth is not None and twist.linear.x > 0:
            twist = self._apply_depth_guard(twist, depth)

        # ── Debug overlay ──────────────────────────────────────────────
        if debug is not None:
            with self._state_lock:
                state_str = self.bot_state.name
            self._draw_overlay(debug, f'{state_str} | {self.active_colour}', twist)
            if lidar is not None:
                self._draw_lidar_arcs(debug, lidar)

        return twist, debug

    # ======================================================================
    # APRILTAG DETECTION WITH POSE ESTIMATION (solvePnP)
    # ======================================================================

    def _detect_apriltag_with_pose(
        self,
        gray:  np.ndarray,
        K:     np.ndarray,
        dist:  Optional[np.ndarray],
        debug: Optional[np.ndarray],
    ) -> Optional[dict]:
        """
        Detect the largest AprilTag, run solvePnP to get its 6-DoF pose
        in the camera frame.

        Returns dict with keys:
            id    : int
            center: (cx, cy) pixels
            tvec  : np.ndarray (3,1)  — translation in camera frame
            rvec  : np.ndarray (3,1)  — rotation vector in camera frame
        """
        corners_list, ids = self._raw_detect(gray)
        if ids is None or len(ids) == 0:
            return None

        # Pick largest detection by corner area
        best_idx = int(np.argmax([
            cv2.contourArea(c.reshape(4, 2)) for c in corners_list
        ]))
        tag_id      = int(ids[best_idx])
        tag_corners = corners_list[best_idx].reshape(4, 2).astype(np.float64)

        # Image points: TL, TR, BR, BL  (matches TAG_OBJ_POINTS order)
        img_pts = tag_corners

        dist_coeffs = dist if dist is not None else np.zeros(5)
        success, rvec, tvec = cv2.solvePnP(
            TAG_OBJ_POINTS, img_pts, K, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not success:
            return None

        cx = int(tag_corners[:, 0].mean())
        cy = int(tag_corners[:, 1].mean())

        if debug is not None:
            ACTION_LABELS = {
                0: 'TURN RIGHT', 1: 'TURN LEFT',
                2: 'FOLLOW GREEN', 3: 'U-TURN', 4: 'FOLLOW ORANGE',
            }
            pts = tag_corners.astype(int)
            cv2.polylines(debug, [pts.reshape(1, 4, 2)], True, (255, 0, 255), 2)
            cv2.circle(debug, (cx, cy), 6, (255, 0, 255), -1)
            dist_m = float(np.linalg.norm(tvec))
            label  = ACTION_LABELS.get(tag_id, f'TAG {tag_id}')
            cv2.putText(debug, f'{label} {dist_m:.2f}m',
                        (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 255), 2)
            # Draw axes
            cv2.drawFrameAxes(debug, K, dist_coeffs, rvec, tvec,
                               TAG_SIZE_M * 0.5)

        return {'id': tag_id, 'center': (cx, cy), 'tvec': tvec, 'rvec': rvec}

    def _raw_detect(
        self, gray: np.ndarray
    ) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """Returns (corners_list, ids) using pupil_apriltags or ArUco fallback."""
        if self._apriltag_detector is not None:
            try:
                detections = self._apriltag_detector.detect(gray)
                if not detections:
                    return [], None
                corners_list = [
                    d.corners.reshape(1, 4, 2).astype(np.float32)
                    for d in detections
                ]
                ids = np.array([[d.tag_id] for d in detections])
                return corners_list, ids
            except Exception as e:
                self.get_logger().warn(f'pupil_apriltags error: {e}')

        # ArUco fallback
        try:
            aruco_dict   = cv2.aruco.getPredefinedDictionary(
                cv2.aruco.DICT_APRILTAG_36h11
            )
            det = cv2.aruco.ArucoDetector(
                aruco_dict, cv2.aruco.DetectorParameters()
            )
            corners_list, ids, _ = det.detectMarkers(gray)
            if ids is None:
                return [], None
            return list(corners_list), ids
        except Exception as e:
            self.get_logger().warn(f'ArUco error: {e}')
            return [], None

    # ======================================================================
    # TAG-POSE → WORLD GOAL
    # ======================================================================

    def _tag_pose_to_world_goal(
        self,
        tag_id: int,
        tvec:   np.ndarray,
        rvec:   np.ndarray,
    ) -> Optional[Tuple[float, float]]:
        """
        Convert the tag pose (camera frame) → world frame via TF2,
        then compute a goal point 1 tile away in the intended turn direction.

        Returns (goal_x, goal_y) in world frame, or None on TF failure.
        """
        # ── Step 1: build tag pose as PoseStamped in camera frame ─────
        tag_pose_cam = PoseStamped()
        tag_pose_cam.header.frame_id = FRAME_CAMERA
        tag_pose_cam.header.stamp    = self.get_clock().now().to_msg()

        tag_pose_cam.pose.position.x = float(tvec[0])
        tag_pose_cam.pose.position.y = float(tvec[1])
        tag_pose_cam.pose.position.z = float(tvec[2])

        # Convert rvec → quaternion
        rot_mat, _ = cv2.Rodrigues(rvec)
        rot4 = np.eye(4)
        rot4[:3, :3] = rot_mat
        from tf_transformations import quaternion_from_matrix
        q = quaternion_from_matrix(rot4)
        tag_pose_cam.pose.orientation.x = q[0]
        tag_pose_cam.pose.orientation.y = q[1]
        tag_pose_cam.pose.orientation.z = q[2]
        tag_pose_cam.pose.orientation.w = q[3]

        # ── Step 2: transform tag pose to world frame ──────────────────
        try:
            tag_pose_world = self._tf_buffer.transform(
                tag_pose_cam, FRAME_WORLD,
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
        except Exception as e:
            self.get_logger().warn(f'TF transform failed: {e}')
            return None

        tx = tag_pose_world.pose.position.x
        ty = tag_pose_world.pose.position.y

        # ── Step 3: get tag's facing direction in world (its +X axis) ──
        tq = tag_pose_world.pose.orientation
        _, _, tag_yaw = euler_from_quaternion([tq.x, tq.y, tq.z, tq.w])

        # ── Step 4: offset goal 1 tile from tag in turn direction ──────
        GOAL_OFFSET_MAP = {
            0: ( math.sin(tag_yaw), -math.cos(tag_yaw)),   # tag's -Y (right turn)
            1: (-math.sin(tag_yaw),  math.cos(tag_yaw)),   # tag's +Y (left  turn)
            3: ( math.cos(tag_yaw),  math.sin(tag_yaw)),   # tag's +X (U-turn, back)
        }

        offset = GOAL_OFFSET_MAP.get(tag_id)
        if offset is None:
            # Colour-switch tags — no spatial goal needed
            return None

        goal_x = tx + offset[0] * TILE_LENGTH
        goal_y = ty + offset[1] * TILE_LENGTH

        self.get_logger().info(
            f'[TAG {tag_id}] Tag world pos=({tx:.2f},{ty:.2f})  '
            f'tag_yaw={math.degrees(tag_yaw):.1f}°  '
            f'goal=({goal_x:.2f},{goal_y:.2f})'
        )
        return (goal_x, goal_y)

    def _get_bot_world_pose(self) -> Optional[Tuple[float, float, float]]:
        """
        Returns (x, y, yaw) of base_link in world frame.
        Uses TF2 buffer.
        """
        try:
            tf = self._tf_buffer.lookup_transform(
                FRAME_WORLD, FRAME_BASE_LINK,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
            bx  = tf.transform.translation.x
            by  = tf.transform.translation.y
            rot = tf.transform.rotation
            _, _, yaw = euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
            return (bx, by, yaw)
        except Exception as e:
            self.get_logger().warn(f'bot pose lookup failed: {e}')
            return None

    # ======================================================================
    # TAG ACTION DISPATCHER  (two-waypoint system)
    # ======================================================================

    def _execute_tag_action(
        self,
        tag_id: int,
        tvec:   np.ndarray,
        rvec:   np.ndarray,
        K:      Optional[np.ndarray],
        dist:   Optional[np.ndarray],
    ):
        """
        Two-waypoint dispatcher:
            WP1 = bot_pos + APPROACH_DIST in bot's current forward heading
            WP2 = WP1 + TILE_LENGTH in the tag-action direction

        Sequence:
            1. Claim state
            2. Snapshot bot pose → compute WP1
            3. Compute WP2 from tag TF (or fallback nominal angle)
            4. Drive straight to WP1
            5. At WP1: check clearance, rotate to face WP2
            6. Cooldown
        """
        with self._state_lock:
            if self.bot_state != BotState.FOLLOWING:
                return
            self.bot_state = BotState.CLEARING

        # ── Colour-switch tags: no driving needed ─────────────────────
        COLOUR_TAGS = {2: 'GREEN', 4: 'ORANGE'}
        if tag_id in COLOUR_TAGS:
            handler = {
                2: self._tag_follow_green,
                4: self._tag_follow_orange,
            }.get(tag_id)
            if handler:
                handler()
            with self._state_lock:
                self.bot_state       = BotState.COOLDOWN
                self._cooldown_until = time.time() + TAG_COOLDOWN_SEC
            return

        # ── Get bot's current world pose ──────────────────────────────
        bot_pose = self._get_bot_world_pose()
        if bot_pose is None:
            self.get_logger().warn(
                f'[TAG {tag_id}] Cannot get bot pose — aborting.'
            )
            with self._state_lock:
                self.bot_state = BotState.FOLLOWING
            return

        bx, by, current_yaw = bot_pose

        # ── WP1: APPROACH_DIST straight ahead in bot's current heading ─
        wp1_x = bx + APPROACH_DIST * math.cos(current_yaw)
        wp1_y = by + APPROACH_DIST * math.sin(current_yaw)
        self.get_logger().info(
            f'[TAG {tag_id}] WP1=({wp1_x:.2f},{wp1_y:.2f})  '
            f'bot=({bx:.2f},{by:.2f})  yaw={math.degrees(current_yaw):.1f}°'
        )

        # ── WP2: action-direction offset from WP1 ─────────────────────
        world_goal = self._tag_pose_to_world_goal(tag_id, tvec, rvec)
        wp2 = self._compute_wp2(tag_id, wp1_x, wp1_y, world_goal, current_yaw)
        self.get_logger().info(f'[TAG {tag_id}] WP2={wp2}')

        # ── Drive to WP1 ──────────────────────────────────────────────
        self.get_logger().info(f'[TAG {tag_id}] Driving to WP1…')
        reached = self._drive_to_waypoint(wp1_x, wp1_y)
        if not reached:
            self.get_logger().warn(
                f'[TAG {tag_id}] Failed to reach WP1 — aborting.'
            )
            with self._state_lock:
                self.bot_state = BotState.FOLLOWING
            return

        # ── Turn toward WP2 ───────────────────────────────────────────
        TURN_DEG_MAP = {
            0: TURN_RIGHT_DEG,
            1: TURN_LEFT_DEG,
            3: U_TURN_DEG,
        }
        fallback_deg = TURN_DEG_MAP.get(tag_id, TURN_RIGHT_DEG)

        handler_map = {
            0: lambda: self._tag_turn_right(wp2),
            1: lambda: self._tag_turn_left(wp2),
            3: lambda: self._tag_uturn(wp2),
        }
        handler = handler_map.get(tag_id)
        if handler is None:
            self.get_logger().warn(f'[TAG {tag_id}] No turn handler — ignoring.')
            with self._state_lock:
                self.bot_state = BotState.FOLLOWING
            return

        self.get_logger().info(
            f'[TAG {tag_id}] At WP1 — executing turn toward WP2…'
        )
        handler()

        with self._state_lock:
            self.bot_state       = BotState.COOLDOWN
            self._cooldown_until = time.time() + TAG_COOLDOWN_SEC
        self.get_logger().info(
            f'[TAG {tag_id}] Complete — cooldown {TAG_COOLDOWN_SEC}s'
        )

    # ======================================================================
    # WAYPOINT HELPERS
    # ======================================================================

    def _compute_wp2(
        self,
        tag_id:      int,
        wp1_x:       float,
        wp1_y:       float,
        world_goal:  Optional[Tuple[float, float]],
        current_yaw: float,
    ) -> Optional[Tuple[float, float]]:
        """
        Compute WP2 = WP1 + TILE_LENGTH in the action direction.

        If TF succeeded (world_goal not None):
            derive unit direction vector from WP1 toward world_goal,
            place WP2 = WP1 + unit_vec * TILE_LENGTH.

        If TF failed:
            fall back to nominal turn angle applied to current_yaw.

        Returns (wp2_x, wp2_y) or None if bot pose also unavailable.
        """
        # ── TF-based direction ────────────────────────────────────────
        if world_goal is not None:
            dx   = world_goal[0] - wp1_x
            dy   = world_goal[1] - wp1_y
            dist = math.hypot(dx, dy)
            if dist > 0.01:
                ux    = dx / dist
                uy    = dy / dist
                wp2_x = wp1_x + ux * TILE_LENGTH
                wp2_y = wp1_y + uy * TILE_LENGTH
                self.get_logger().info(
                    f'  WP2 via TF direction: ({wp2_x:.2f},{wp2_y:.2f})'
                )
                return (wp2_x, wp2_y)

        # ── Fallback: nominal turn angle from current_yaw ─────────────
        FALLBACK_OFFSETS = {
            0: TURN_RIGHT_DEG,
            1: TURN_LEFT_DEG,
            3: U_TURN_DEG,
        }
        offset_deg = FALLBACK_OFFSETS.get(tag_id, 0.0)
        target_yaw = current_yaw + math.radians(offset_deg)
        wp2_x      = wp1_x + math.cos(target_yaw) * TILE_LENGTH
        wp2_y      = wp1_y + math.sin(target_yaw) * TILE_LENGTH
        self.get_logger().info(
            f'  WP2 via fallback {offset_deg:.0f}°: ({wp2_x:.2f},{wp2_y:.2f})'
        )
        return (wp2_x, wp2_y)

    def _drive_to_waypoint(
        self,
        goal_x: float,
        goal_y: float,
    ) -> bool:
        """
        Drive straight toward (goal_x, goal_y) in world frame using a
        P-controller on heading error plus constant forward speed.
        LiDAR forward guard aborts if wall is too close.

        Returns True if goal reached within WP_POS_TOL, False on
        timeout or LiDAR abort.
        """
        deadline = time.time() + WP_TIMEOUT
        self.get_logger().info(
            f'  Driving to WP ({goal_x:.2f},{goal_y:.2f})  '
            f'tol={WP_POS_TOL}m  timeout={WP_TIMEOUT}s'
        )

        while time.time() < deadline:
            # ── Current pose ──────────────────────────────────────────
            bot_pose = self._get_bot_world_pose()
            if bot_pose is None:
                time.sleep(WP_CHECK_DT)
                continue

            bx, by, current_yaw = bot_pose
            dist_to_goal = math.hypot(goal_x - bx, goal_y - by)

            # ── Arrival check ─────────────────────────────────────────
            if dist_to_goal <= WP_POS_TOL:
                self._publish_stop()
                self.get_logger().info(
                    f'  ✓ Reached WP — remaining={dist_to_goal:.3f}m'
                )
                time.sleep(0.1)
                return True

            # ── LiDAR forward safety guard ────────────────────────────
            with self._lidar_lock:
                scan = self._lidar_msg
            if scan is not None:
                fwd_min = self._lidar_sector_min(
                    scan, LIDAR_SECTOR_FWD[0], LIDAR_SECTOR_FWD[1]
                )
                if fwd_min <= LIDAR_STOP_DIST:
                    self._publish_stop()
                    self.get_logger().warn(
                        f'  LiDAR abort — wall at {fwd_min:.2f}m '
                        f'while driving to WP'
                    )
                    return False

            # ── Heading P-controller ──────────────────────────────────
            required_yaw = math.atan2(goal_y - by, goal_x - bx)
            yaw_error    = required_yaw - current_yaw
            # Wrap to (−π, π]
            yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi

            twist           = Twist()
            twist.linear.x  = LINEAR_SPEED_FWD
            twist.angular.z = max(
                -ANGULAR_TURN,
                min(ANGULAR_TURN, WP_YAW_GAIN * yaw_error)
            )

            self.cmd_pub.publish(twist)
            time.sleep(WP_CHECK_DT)

        self._publish_stop()
        self.get_logger().warn(
            f'  _drive_to_waypoint timeout after {WP_TIMEOUT}s'
        )
        return False

    # ======================================================================
    # TAG HANDLERS
    # ======================================================================

    def _tag_turn_right(self, wp2: Optional[Tuple[float, float]]):
        """Tag 0: compute delta_yaw to WP2, check clearance, rotate."""
        self.get_logger().info('[TAG 0] TURN RIGHT')
        delta_yaw = self._compute_delta_yaw(wp2, fallback_deg=TURN_RIGHT_DEG)
        self.get_logger().info(f'  delta_yaw = {math.degrees(delta_yaw):.1f}°')
        if self._wait_for_clearance(TURN_RIGHT_DEG):
            with self._state_lock:
                self.bot_state = BotState.TURNING
            self._action_rotate_by(delta_yaw)
        else:
            self.get_logger().warn('[TAG 0] Clearance failed — aborted.')

    def _tag_turn_left(self, wp2: Optional[Tuple[float, float]]):
        """Tag 1: compute delta_yaw to WP2, check clearance, rotate."""
        self.get_logger().info('[TAG 1] TURN LEFT')
        delta_yaw = self._compute_delta_yaw(wp2, fallback_deg=TURN_LEFT_DEG)
        self.get_logger().info(f'  delta_yaw = {math.degrees(delta_yaw):.1f}°')
        if self._wait_for_clearance(TURN_LEFT_DEG):
            with self._state_lock:
                self.bot_state = BotState.TURNING
            self._action_rotate_by(delta_yaw)
        else:
            self.get_logger().warn('[TAG 1] Clearance failed — aborted.')

    def _tag_uturn(self, wp2: Optional[Tuple[float, float]]):
        """Tag 3: compute delta_yaw to WP2, check clearance, rotate."""
        self.get_logger().info('[TAG 3] U-TURN')
        delta_yaw = self._compute_delta_yaw(wp2, fallback_deg=U_TURN_DEG)
        self.get_logger().info(f'  delta_yaw = {math.degrees(delta_yaw):.1f}°')
        if self._wait_for_clearance(U_TURN_DEG):
            with self._state_lock:
                self.bot_state = BotState.TURNING
            self._action_rotate_by(delta_yaw)
        else:
            self.get_logger().warn('[TAG 3] Clearance failed — aborted.')

    def _tag_follow_green(self):
        self._action_set_colour('GREEN')

    def _tag_follow_orange(self):
        self._action_set_colour('ORANGE')

    # ======================================================================
    # DELTA YAW COMPUTATION
    # ======================================================================

    def _compute_delta_yaw(
        self,
        wp2:          Optional[Tuple[float, float]],
        fallback_deg: float,
    ) -> float:
        """
        Compute the rotation (radians) the bot must turn to face WP2.

        If WP2 is None or bot pose unavailable, falls back to the nominal
        turn angle (e.g. −90° for right turn).

        Returns delta_yaw in radians, wrapped to (−π, π].
        """
        if wp2 is not None:
            bot_pose = self._get_bot_world_pose()
            if bot_pose is not None:
                bx, by, current_yaw = bot_pose
                gx, gy              = wp2

                required_yaw = math.atan2(gy - by, gx - bx)
                delta        = required_yaw - current_yaw

                # Wrap to (−π, π]
                delta = (delta + math.pi) % (2 * math.pi) - math.pi

                self.get_logger().info(
                    f'  bot=({bx:.2f},{by:.2f}) yaw={math.degrees(current_yaw):.1f}°  '
                    f'WP2=({gx:.2f},{gy:.2f})  '
                    f'required={math.degrees(required_yaw):.1f}°  '
                    f'delta={math.degrees(delta):.1f}°'
                )
                return delta

        # Fallback: use nominal angle
        self.get_logger().warn(
            f'  WP2 unavailable — using fallback {fallback_deg:.0f}°'
        )
        return math.radians(fallback_deg)

    # ======================================================================
    # PRE-TURN CLEARANCE LOOP
    # ======================================================================

    def _wait_for_clearance(self, turn_deg: float) -> bool:
        """
        Creep forward until LiDAR confirms >= TILE_LENGTH in the intended direction.

        PRIMARY check  : narrow cone ±CLEARANCE_CONE_DEG around intended heading
        SECONDARY check: broad sector arc for that direction
        CREEP GUARD    : broad forward arc — abort if wall <= LIDAR_STOP_DIST ahead

        Returns True (clear, proceed) or False (wall-blocked / timeout).
        """
        entry = TURN_CLEARANCE_MAP.get(turn_deg)
        if entry is None:
            return True   # no map entry → proceed without check

        cone_centre_deg, broad_sector = entry
        cone_lo = cone_centre_deg - CLEARANCE_CONE_DEG
        cone_hi = cone_centre_deg + CLEARANCE_CONE_DEG

        deadline = time.time() + CLEARANCE_TIMEOUT
        attempt  = 0

        self.get_logger().info(
            f'  Clearance check: cone [{cone_lo:.0f}°,{cone_hi:.0f}°] '
            f'broad {broad_sector}  need ≥{TILE_LENGTH}m'
        )

        while time.time() < deadline:
            with self._lidar_lock:
                scan = self._lidar_msg
            if scan is None:
                time.sleep(CREEP_CHECK_DT)
                continue

            cone_min  = self._lidar_sector_min(scan, cone_lo, cone_hi)
            broad_min = self._lidar_sector_min(scan, broad_sector[0], broad_sector[1])
            attempt  += 1

            if attempt % 20 == 0:
                self.get_logger().info(
                    f'  [{attempt}] cone={cone_min:.2f}m  broad={broad_min:.2f}m'
                )

            if cone_min >= TILE_LENGTH and broad_min >= TILE_LENGTH:
                self.get_logger().info(
                    f'  ✓ Clearance OK — cone={cone_min:.2f}m broad={broad_min:.2f}m'
                )
                self._publish_stop()
                time.sleep(0.1)
                return True

            # Forward wall guard
            fwd_min = self._lidar_sector_min(
                scan, LIDAR_SECTOR_FWD[0], LIDAR_SECTOR_FWD[1]
            )
            if fwd_min <= LIDAR_STOP_DIST:
                self.get_logger().warn(
                    f'  Wall ahead at {fwd_min:.2f}m — cannot creep. Abort.'
                )
                self._publish_stop()
                return False

            # Creep forward
            creep = Twist()
            creep.linear.x = CREEP_SPEED
            self.cmd_pub.publish(creep)
            time.sleep(CREEP_CHECK_DT)

        self._publish_stop()
        self.get_logger().warn(f'  Clearance timeout after {CLEARANCE_TIMEOUT}s.')
        return False

    # ======================================================================
    # ROTATION ACTION
    # ======================================================================

    def _action_rotate_by(self, delta_yaw: float):
        """
        Rotate the bot by delta_yaw radians.
        Positive = CCW (left), negative = CW (right).
        LiDAR broad-arc abort on every publish cycle.
        """
        direction  = 1.0 if delta_yaw >= 0 else -1.0
        target_rad = abs(delta_yaw)
        speed      = ANGULAR_TURN

        # Pick the relevant broad sector based on turn direction
        if direction < 0:
            _, broad_sector = TURN_CLEARANCE_MAP.get(
                TURN_RIGHT_DEG, (None, LIDAR_SECTOR_RIGHT)
            )
        elif abs(math.degrees(delta_yaw)) >= 170:
            _, broad_sector = TURN_CLEARANCE_MAP.get(
                U_TURN_DEG, (None, LIDAR_SECTOR_UTURN)
            )
        else:
            _, broad_sector = TURN_CLEARANCE_MAP.get(
                TURN_LEFT_DEG, (None, LIDAR_SECTOR_LEFT)
            )

        twist = Twist()
        twist.angular.z = direction * speed

        dt         = 0.05
        start_time = time.time()
        duration   = target_rad / speed

        self.get_logger().info(
            f'Rotating {math.degrees(delta_yaw):+.1f}° '
            f'at {speed:.2f} rad/s (est {duration:.1f}s)'
        )

        while (time.time() - start_time) < duration:
            with self._lidar_lock:
                scan = self._lidar_msg
            if scan is not None:
                min_d = self._lidar_sector_min(
                    scan, broad_sector[0], broad_sector[1]
                )
                if min_d < LIDAR_STOP_DIST:
                    self.get_logger().warn(
                        f'LiDAR abort! {min_d:.2f}m in {broad_sector}'
                    )
                    self._publish_stop()
                    return
            self.cmd_pub.publish(twist)
            time.sleep(dt)

        self._publish_stop()
        time.sleep(0.1)

    # ======================================================================
    # COLOUR SWITCH
    # ======================================================================

    def _action_set_colour(self, colour: str):
        self.active_colour = colour
        self.set_parameters([
            rclpy.parameter.Parameter(
                'active_colour',
                rclpy.parameter.Parameter.Type.STRING,
                colour,
            )
        ])
        self.get_logger().info(f'Colour mode → {colour}')

    # ======================================================================
    # LIDAR HELPERS
    # ======================================================================

    def _lidar_sector_min(
        self,
        scan:             LaserScan,
        sector_start_deg: float,
        sector_end_deg:   float,
    ) -> float:
        angle_min = scan.angle_min
        angle_inc = scan.angle_increment
        ranges    = np.array(scan.ranges, dtype=np.float32)
        ranges    = np.where(np.isfinite(ranges), ranges, scan.range_max)

        def to_idx(a: float) -> int:
            return max(0, min(len(ranges) - 1,
                              int((a - angle_min) / angle_inc)))

        s_rad = math.radians(sector_start_deg)
        e_rad = math.radians(sector_end_deg)

        if sector_start_deg <= sector_end_deg:
            seg = ranges[to_idx(s_rad): to_idx(e_rad) + 1]
        else:
            # Wrap-around (e.g. 150° → 210°)
            seg = np.concatenate([
                ranges[to_idx(s_rad): to_idx(math.radians(180)) + 1],
                ranges[to_idx(math.radians(-180)): to_idx(
                    e_rad - math.radians(360)) + 1],
            ])

        return float(np.min(seg)) if seg.size > 0 else float('inf')

    # ======================================================================
    # HELPERS
    # ======================================================================

    def _publish_stop(self):
        self.cmd_pub.publish(Twist())

    # ======================================================================
    # AprilTag detector init
    # ======================================================================

    def _init_apriltag_detector(self):
        try:
            from pupil_apriltags import Detector
            self._apriltag_detector = Detector(
                families=TAG_FAMILY, nthreads=2, quad_decimate=1.0
            )
            self.get_logger().info('AprilTag: pupil_apriltags')
        except ImportError:
            self._apriltag_detector = None
            self.get_logger().warn('pupil_apriltags not found — ArUco fallback')

    # ======================================================================
    # Template matching
    # ======================================================================

    def _load_template(
        self, path: str
    ) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            self.get_logger().warn(f'Template not found: {path}')
            return None, []
        if len(img.shape) == 3 and img.shape[2] == 4:
            alpha   = img[:, :, 3:4].astype(np.float32) / 255.0
            rgb     = img[:, :, :3].astype(np.float32)
            img_bgr = (rgb * alpha + np.ones_like(rgb) * 255 * (1 - alpha)
                       ).astype(np.uint8)
        else:
            img_bgr = img[:, :, :3] if len(img.shape) == 3 else img
        base_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = base_gray.shape
        scaled_cache = []
        for s in TEMPLATE_SCALES:
            sw, sh = int(w * s), int(h * s)
            if sw >= 10 and sh >= 10:
                scaled_cache.append(cv2.resize(base_gray, (sw, sh)))
        self.get_logger().info(
            f'Template: {w}×{h}px  {len(scaled_cache)} scales'
        )
        return base_gray, scaled_cache

    def _run_template_match(self, gray: np.ndarray) -> TemplateResult:
        if not self._template_scales_cache:
            return TemplateResult(on_tile=False, confidence=0.0)
        fh, fw = gray.shape
        best   = (0.0, None, None)
        for tmpl in self._template_scales_cache:
            th, tw = tmpl.shape
            if th > fh or tw > fw:
                continue
            res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
            _, v, _, loc = cv2.minMaxLoc(res)
            if v > best[0]:
                best = (v, loc, (tw, th))
        conf, loc, size = best
        on_tile = conf >= TEMPLATE_MATCH_THRESHOLD
        return TemplateResult(on_tile=on_tile, confidence=conf,
                               match_loc=loc  if on_tile else None,
                               match_size=size if on_tile else None)

    # ======================================================================
    # Colour blob detection
    # ======================================================================

    def _detect_blob(
        self, hsv, low, high, label, debug, depth=None
    ) -> Optional[ColourBlob]:
        mask = cv2.inRange(hsv, low, high)
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
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
        distance_m = None
        if depth is not None and debug is not None:
            dh, dw = depth.shape[:2]
            sx = max(0, min(int(cx * dw / debug.shape[1]), dw - 1))
            sy = max(0, min(int(cy * dh / debug.shape[0]), dh - 1))
            patch = depth[max(0, sy-2):sy+3, max(0, sx-2):sx+3]
            valid = patch[np.isfinite(patch)]
            if valid.size > 0:
                distance_m = float(np.median(valid))
        if debug is not None:
            col = (0, 220, 0) if label == 'GREEN' else (0, 140, 255)
            cv2.drawContours(debug, [largest], -1, col, 2)
            cv2.circle(debug, (cx, cy), 9, col, -1)
            dt = f'{distance_m:.2f}m' if distance_m and np.isfinite(distance_m) \
                 else f'{area:.0f}px'
            cv2.putText(debug, f'{label} {dt}',
                        (cx+12, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        return ColourBlob(centroid=(cx, cy), area=area, distance_m=distance_m)

    # ======================================================================
    # Blob → Twist
    # ======================================================================

    def _blobs_to_twist(self, green, orange, frame_h) -> Twist:
        twist     = Twist()
        primary   = green  if self.active_colour == 'GREEN' else orange
        secondary = orange if self.active_colour == 'GREEN' else green
        threshold = frame_h * SIDE_THRESHOLD_RATIO
        if primary is None and secondary is None:
            twist.linear.x = LINEAR_SPEED_FWD * 0.5
        elif primary is None:
            twist.linear.x = LINEAR_SPEED_FWD * 0.4
        elif secondary is None:
            twist.linear.x = LINEAR_SPEED_FWD
        else:
            dy = primary.centroid[1] - secondary.centroid[1]
            twist.linear.x = (
                LINEAR_SPEED_FWD if abs(dy) < threshold or dy > 0
                else LINEAR_SPEED_BWD
            )
        return twist

    def _apply_depth_guard(self, twist: Twist, depth: np.ndarray) -> Twist:
        h, w  = depth.shape[:2]
        strip = depth[int(h*0.4):int(h*0.7), int(w*0.4):int(w*0.6)]
        strip = np.where(np.isfinite(strip), strip, 10.0)
        return Twist() if float(np.nanmin(strip)) < 0.3 else twist

    # ======================================================================
    # Debug drawing
    # ======================================================================

    def _draw_tile_info(self, frame, result):
        h = frame.shape[0]
        if result.on_tile and result.match_loc and result.match_size:
            x, y   = result.match_loc
            mw, mh = result.match_size
            cv2.rectangle(frame, (x, y), (x+mw, y+mh), (255, 255, 0), 2)
            cv2.putText(frame, f'TILE {result.confidence:.2f}',
                        (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        col  = (0, 255, 255) if self.tile_tracker.on_tile else (180, 180, 180)
        done = '  ← DONE!' if self.tile_tracker.tile_done else ''
        on   = 'ON TILE' if self.tile_tracker.on_tile else 'between tiles'
        cv2.putText(frame,
                    f'Tiles: {self.tile_tracker.tiles_passed}  [{on}]{done}',
                    (8, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    def _draw_overlay(self, frame, label, twist):
        text = f'{label} | lin={twist.linear.x:+.2f}  ang={twist.angular.z:+.2f}'
        col  = ((0,255,0) if twist.linear.x > 0 else
                (0,0,255) if twist.linear.x < 0 else (0,200,255))
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0,0,0), -1)
        cv2.putText(frame, text, (8, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

    def _draw_lidar_arcs(self, frame, scan):
        h, w   = frame.shape[:2]
        cx, cy = w-80, h-80
        radius = 60
        sectors = {
            'F': (LIDAR_SECTOR_FWD,   (255, 255,   0)),
            'R': (LIDAR_SECTOR_RIGHT, (  0,   0, 255)),
            'L': (LIDAR_SECTOR_LEFT,  (  0, 255,   0)),
            'U': (LIDAR_SECTOR_UTURN, (  0, 165, 255)),
        }
        cv2.circle(frame, (cx, cy), radius, (60, 60, 60), 1)
        for lbl, (sector, colour) in sectors.items():
            min_d = self._lidar_sector_min(scan, sector[0], sector[1])
            if max(0.0, 1.0 - min_d/2.0) > 0.1:
                cv2.ellipse(frame, (cx, cy), (radius-5, radius-5),
                            0, int(-sector[1]), int(-sector[0]), colour, 3)
            mid = math.radians((sector[0]+sector[1])/2.0)
            tx = cx + int((radius+10)*math.sin(mid))
            ty = cy - int((radius+10)*math.cos(mid))
            cv2.putText(frame, f'{min_d:.1f}m',
                        (tx-15, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)


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
        node.cmd_pub.publish(Twist())
        node.get_logger().info('Safety stop. Shutting down.')
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()