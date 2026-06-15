import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np

class StreamDebugger(Node):
    def __init__(self):
        super().__init__('stream_debugger')
        self.camera_names = ['camera_0', 'camera_1', 'camera_2', 'camera_3']
        self.frame_counts = {name: 0 for name in self.camera_names}
        self.sample_shapes = {name: None for name in self.camera_names}
        self.sample_means = {name: None for name in self.camera_names}
        
        self.subs = []
        for name in self.camera_names:
            topic = f"/{name}/image_raw"
            self.subs.append(
                self.create_subscription(
                    Image, topic, lambda msg, n=name: self.cb(msg, n), 10
                )
            )
        
        # Print status updates every 2 seconds
        self.timer = self.create_timer(2.0, self.print_report)
        self.get_logger().info("--- Stream Diagnostic Started ---")

    def cb(self, msg, name):
        self.frame_counts[name] += 1
        # Quick manual parsing of the raw data buffer array
        if self.sample_shapes[name] is None:
            self.sample_shapes[name] = (msg.height, msg.width)
            # Check the actual raw data values inside the message payload buffer
            raw_bytes = np.frombuffer(msg.data, dtype=np.uint8)
            if len(raw_bytes) > 0:
                self.sample_means[name] = np.mean(raw_bytes)

    def print_report(self):
        print("\n=== LIVE REALSENSE TOPIC REPORT ===")
        for name in self.camera_names:
            cnt = self.frame_counts[name]
            shape = self.sample_shapes[name]
            mean_val = self.sample_means[name]
            
            print(f"[{name}] Total Frames Received: {cnt}")
            if cnt > 0:
                print(f"   -> Frame Resolution: {shape}")
                print(f"   -> Brightness Mean (0-255): {mean_val:.2f}")
                if mean_val == 0.0:
                    print("   ❌ ALERT: Node is transmitting pure zero/black frames!")
            else:
                print("   ❌ ALERT: No messages received! Subscriber is starved.")
        print("====================================")

def main():
    rclpy.init()
    node = StreamDebugger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()