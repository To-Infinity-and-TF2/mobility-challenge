#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist, Point

from cv_bridge import CvBridge

import cv2
import numpy as np

from message_filters import Subscriber, ApproximateTimeSynchronizer


class LogoDepthFollower(Node):

    def __init__(self):
        super().__init__('logo_depth_follower')

        self.bridge = CvBridge()
        self.camera_info = None

        # --- Subscribers (SYNCED RGB + DEPTH) ---
        self.rgb_sub = Subscriber(self, Image, '/camera/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/depth_image_raw')

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.synced_callback)

        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera_info',
            self.camera_info_cb,
            10
        )

        # --- Publishers ---
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.dist_pub = self.create_publisher(Point, '/logo_distance', 10)

        # --- HSV Ranges (TUNE THESE) ---
        self.lower_green = np.array([40, 50, 50])
        self.upper_green = np.array([80, 255, 255])

        self.lower_orange = np.array([5, 100, 100])
        self.upper_orange = np.array([20, 255, 255])

        # Mode (can be overridden by AprilTag later)
        self.follow_mode = "green"

        self.get_logger().info("✅ Logo Depth Follower Node Started")

    # -----------------------------
    def camera_info_cb(self, msg):
        self.camera_info = msg

    # -----------------------------
    def synced_callback(self, rgb_msg, depth_msg):

        if self.camera_info is None:
            self.get_logger().warn("Waiting for camera info...")
            return

        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')

        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

        # --- Masks ---
        green_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        orange_mask = cv2.inRange(hsv, self.lower_orange, self.upper_orange)

        # --- Morphological cleanup ---
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)

        # --- Detect blobs ---
        green_center, green_bbox = self.get_blob(green_mask, rgb, (0, 255, 0))
        orange_center, orange_bbox = self.get_blob(orange_mask, rgb, (0, 140, 255))

        twist = Twist()

        h, w, _ = rgb.shape
        center_x = w // 2

        target_detected = False

        # ===============================
        # 🎯 MAIN DETECTION LOGIC
        # ===============================
        if green_center and orange_center:

            if self.follow_mode == "green":
                cx, cy = green_center
                bbox = green_bbox
            else:
                cx, cy = orange_center
                bbox = orange_bbox

            xmin, ymin, xmax, ymax = bbox

            # --- Depth ROI ---
            roi = depth[ymin:ymax, xmin:xmax].flatten()
            roi = roi[np.isfinite(roi)]
            roi = roi[(roi > 0.2) & (roi < 10.0)]

            if roi.size > 30:
                z = float(np.median(roi))  # distance

                # --- Camera projection ---
                fx = self.camera_info.k[0]
                fy = self.camera_info.k[4]
                cx_cam = self.camera_info.k[2]
                cy_cam = self.camera_info.k[5]

                x = (cx - cx_cam) * z / fx
                y = (cy - cy_cam) * z / fy

                # --- Publish distance ---
                point = Point(x=x, y=y, z=z)
                self.dist_pub.publish(point)

                # --- Visualization ---
                cv2.putText(rgb, f"Z: {z:.2f}m",
                            (cx, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)

                # ===============================
                # 🚗 CONTROL SYSTEM
                # ===============================
                error_x = cx - center_x

                # Angular control (P-controller)
                twist.angular.z = -0.003 * error_x

                # Deadband (avoid jitter)
                if abs(error_x) < 20:
                    twist.angular.z = 0.0

                # Speed control based on distance
                if z > 1.0:
                    twist.linear.x = 0.25
                elif z > 0.5:
                    twist.linear.x = 0.15
                else:
                    twist.linear.x = 0.05

                target_detected = True

        # ===============================
        # 🔄 FALLBACK (SEARCH MODE)
        # ===============================
        if not target_detected:
            twist.linear.x = 0.05
            twist.angular.z = 0.3

        # ===============================
        # 🚀 ALWAYS PUBLISH CMD_VEL
        # ===============================
        self.cmd_pub.publish(twist)

        # ===============================
        # 🖥️ DEBUG WINDOWS
        # ===============================
        cv2.imshow("Detection", rgb)
        cv2.imshow("Green Mask", green_mask)
        cv2.imshow("Orange Mask", orange_mask)
        cv2.waitKey(1)

    # -----------------------------
    def get_blob(self, mask, frame, color):

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None

        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) < 500:
            return None, None

        x, y, w, h = cv2.boundingRect(largest)

        cx = x + w // 2
        cy = y + h // 2

        # Draw
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.circle(frame, (cx, cy), 5, color, -1)

        return (cx, cy), (x, y, x + w, y + h)


# -----------------------------
def main(args=None):
    rclpy.init(args=args)

    node = LogoDepthFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()