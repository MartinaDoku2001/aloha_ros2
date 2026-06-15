import numpy as np
import time
import rclpy
from rclpy.node import Node
from collections import deque
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from interbotix_xs_msgs.msg import JointSingleCommand, JointGroupCommand
from cv_bridge import CvBridge
from constants import DT

import time
from collections import deque
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageRecorder(Node):
    def __init__(self, init_node=True, is_debug=False):
        # Force a unique node identification
        super().__init__('image_recorder')
        self.is_debug = is_debug
        self.bridge = CvBridge()
        
        # 1. Update these to match your exact camera_0, camera_1 topology
        self.camera_names = ['camera_0', 'camera_1', 'camera_2', 'camera_3']
        self.camera_subscribers = {}

        for cam_name in self.camera_names:
            setattr(self, f'{cam_name}_image', None)
            setattr(self, f'{cam_name}_secs', None)
            setattr(self, f'{cam_name}_nsecs', None)
            
            # 2. Map dynamically using a uniform clean lambda callback structure
            # This points exactly to your active topic: /camera_X/image_raw
            topic_name = f"/{cam_name}/image_raw"
            
            self.camera_subscribers[cam_name] = self.create_subscription(
                Image,
                topic_name,
                lambda msg, name=cam_name: self.image_cb(name, msg),
                10
            )
            self.get_logger().info(f"ImageRecorder successfully subscribed to: {topic_name}")
            
            if self.is_debug:
                setattr(self, f'{cam_name}_timestamps', deque(maxlen=50))

        time.sleep(0.5)

    def image_cb(self, cam_name, data):
        # Convert ROS message to OpenCV image matrix
        # Note: 'passthrough' preserves whatever formatting (RGB/BGR/Mono) the hardware sends
        self.get_logger().debug(f"Received frame from {cam_name}")
        setattr(self, f'{cam_name}_image', self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough'))
        setattr(self, f'{cam_name}_secs', data.header.stamp.sec)
        setattr(self, f'{cam_name}_nsecs', data.header.stamp.nanosec)
        
        if self.is_debug:
            getattr(self, f'{cam_name}_timestamps').append(data.header.stamp.sec + data.header.stamp.nanosec * 1e-9)

    def get_images(self):
        image_dict = dict()
        for cam_name in self.camera_names:
            img = getattr(self, f'{cam_name}_image')
            if img is None:
                # Provide a black fallback image if the camera frame drops to prevent h5py crashes
                img = np.zeros((480, 640, 3), dtype=np.uint8)
            image_dict[cam_name] = img
        return image_dict

    def print_diagnostics(self):
        if not self.is_debug:
            self.get_logger().warn("Diagnostics require is_debug=True initialization")
            return
            
        def dt_helper(l):
            l = np.array(l)
            if len(l) < 2: return 1.0 # Avoid division by zero
            diff = l[1:] - l[:-1]
            return np.mean(diff)
            
        for cam_name in self.camera_names:
            timestamps = getattr(self, f'{cam_name}_timestamps')
            if len(timestamps) > 1:
                image_freq = 1 / dt_helper(timestamps)
                self.get_logger().info(f'{cam_name} {image_freq=:.2f}')
            else:
                self.get_logger().info(f'{cam_name} image_freq=0.00 (No data received)')
        self.get_logger().info('')
        
class Recorder(Node):
    def __init__(self, side, init_node=True, is_debug=False):
        super().__init__(f'recorder_{side}')
        self.secs = None
        self.nsecs = None
        self.qpos = None
        self.qvel = None
        self.effort = None
        self.arm_command = None
        self.gripper_command = None
        self.is_debug = is_debug

        self.joint_state_subscriber = self.create_subscription(
            JointState,
            f"/puppet_{side}/joint_states",
            self.puppet_state_cb,
            10
        )
        self.arm_command_subscriber = self.create_subscription(
            JointGroupCommand,
            f"/puppet_{side}/commands/joint_group",
            self.puppet_arm_commands_cb,
            10
        )
        self.gripper_command_subscriber = self.create_subscription(
            JointSingleCommand,
            f"/puppet_{side}/commands/joint_single",
            self.puppet_gripper_commands_cb,
            10
        )

        if self.is_debug:
            self.joint_timestamps = deque(maxlen=50)
            self.arm_command_timestamps = deque(maxlen=50)
            self.gripper_command_timestamps = deque(maxlen=50)
        time.sleep(0.1)

    def puppet_state_cb(self, data):
        self.qpos = data.position
        self.qvel = data.velocity
        self.effort = data.effort
        self.data = data
        if self.is_debug:
            self.joint_timestamps.append(time.time())

    def puppet_arm_commands_cb(self, data):
        self.arm_command = data.cmd
        if self.is_debug:
            self.arm_command_timestamps.append(time.time())

    def puppet_gripper_commands_cb(self, data):
        self.get_logger().info(f"Received JointState message: {data}")
        self.gripper_command = data.cmd
        if self.is_debug:
            self.gripper_command_timestamps.append(time.time())

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)

        joint_freq = 1 / dt_helper(self.joint_timestamps)
        arm_command_freq = 1 / dt_helper(self.arm_command_timestamps)
        gripper_command_freq = 1 / dt_helper(self.gripper_command_timestamps)

        self.get_logger().info(f'{joint_freq=:.2f}')
        self.get_logger().info(f'{arm_command_freq=:.2f}')
        self.get_logger().info(f'{gripper_command_freq=:.2f}')

def get_arm_joint_positions(bot):
    return bot.arm.core.joint_states.position[:6]

def get_arm_gripper_positions(bot):
    joint_position = bot.gripper.core.joint_states.position[6]
    return joint_position

def move_arms(bot_list, target_pose_list, move_time=1):
    num_steps = int(move_time / DT)
    curr_pose_list = [get_arm_joint_positions(bot) for bot in bot_list]
    traj_list = [np.linspace(curr_pose, target_pose, num_steps) for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            bot.arm.set_joint_positions(traj_list[bot_id][t], blocking=False)
        time.sleep(DT)

def move_grippers(bot_list, target_pose_list, move_time):
    gripper_command = JointSingleCommand(name="gripper")
    num_steps = int(move_time / DT)
    curr_pose_list = [get_arm_gripper_positions(bot) for bot in bot_list]
    traj_list = [np.linspace(curr_pose, target_pose, num_steps) for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            gripper_command.cmd = traj_list[bot_id][t]
            bot.gripper.core.pub_single.publish(gripper_command)
        time.sleep(DT)

def setup_puppet_bot(bot):
    bot.core.robot_reboot_motors("single", "gripper", True)
    bot.core.robot_set_operating_modes("group", "arm", "position")
    bot.core.robot_set_operating_modes("single", "gripper", "current_based_position")
    torque_on(bot)

def setup_master_bot(bot):
    bot.core.robot_set_operating_modes("group", "arm", "pwm")
    bot.core.robot_set_operating_modes("single", "gripper", "current_based_position")
    torque_off(bot)

def set_standard_pid_gains(bot):
    bot.core.robot_set_motor_registers("group", "arm", 'Position_P_Gain', 800)
    bot.core.robot_set_motor_registers("group", "arm", 'Position_I_Gain', 0)

def set_low_pid_gains(bot):
    bot.core.robot_set_motor_registers("group", "arm", 'Position_P_Gain', 100)
    bot.core.robot_set_motor_registers("group", "arm", 'Position_I_Gain', 0)

def torque_off(bot):
    bot.core.robot_torque_enable("group", "arm", False)
    bot.core.robot_torque_enable("single", "gripper", False)

def torque_on(bot):
    bot.core.robot_torque_enable("group", "arm", True)
    bot.core.robot_torque_enable("single", "gripper", True)
