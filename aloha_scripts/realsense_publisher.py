#!/usr/bin/env python3
import sys
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class RealsensePublisher(Node):
    def __init__(self, camera_index=None):
        cam_name = f'camera_{camera_index}'
        super().__init__(f'realsense_publisher_{cam_name}')
        
        # Broadcast exactly what snn_aloha expects
        topic_name = f'{cam_name}/image_raw'
        self.publisher_ = self.create_publisher(Image, topic_name, 10)
        self.bridge = CvBridge()
        
        v4l2_mapping = {
            '0': 10,   
            '1': 6,   
            '2': 18,  
            '3': 26   
        }
        
        target_port = v4l2_mapping.get(str(camera_index), 0)
        self.get_logger().info(f"Opening hardware device: /dev/video{target_port} for {cam_name}")
        
        self.cap = cv2.VideoCapture(target_port, cv2.CAP_V4L2)
        
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            self.get_logger().error(f"CRITICAL: Could not open connection to /dev/video{target_port}!")
        
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)

    def timer_callback(self):
        if not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if ret and frame is not None:
            try:
                # Check if OpenCV's backend already auto-decoded the frame into 3-channel BGR
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    color_image = frame  # It is already in perfect color, no conversion needed!
                else:
                    # If it somehow returns raw 2-channel YUYV bytes, manually decode it
                    color_image = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)
                
                # Convert the OpenCV frame matrix into an official ROS 2 Image Message
                img_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="rgb8")
                img_msg.header.frame_id = "camera_link"
                img_msg.header.stamp = self.get_clock().now().to_msg()
                
                # Publish to the network
                self.publisher_.publish(img_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to process and publish image matrix: {str(e)}")

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
            self.get_logger().info("Released camera handle successfully.")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    camera_index = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        camera_index = sys.argv[1]

    node = RealsensePublisher(camera_index=camera_index)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()