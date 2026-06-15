import time
import numpy as np
import collections
import dm_env

from constants import DT, START_ARM_POSE, MASTER_GRIPPER_JOINT_NORMALIZE_FN, PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN
from constants import PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
from robot_utils import Recorder, ImageRecorder
from robot_utils import setup_master_bot, setup_puppet_bot, move_arms, move_grippers
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand

from interbotix_common_modules.common_robot.robot import (
    get_interbotix_global_node, 
    create_interbotix_global_node,
    robot_startup
)

class RealEnv:
    def __init__(self, init_node, setup_robots=True):
        try:
            global_node = get_interbotix_global_node()
            if global_node is None:
                global_node = create_interbotix_global_node()
        except Exception:
            global_node = create_interbotix_global_node()

        # Initialize ALL 4 physical arms
        self.puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name='puppet_left', node=global_node)
        self.puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name='puppet_right', node=global_node)
        self.master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper", robot_name='master_left', node=global_node)
        self.master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper", robot_name='master_right', node=global_node)

        if init_node:
            robot_startup(global_node)

        if setup_robots:
            self.setup_robots()

        # Map recorders to BOTH physical channels
        self.recorder_left = Recorder('left', init_node=False)
        self.recorder_right = Recorder('right', init_node=False)
        self.image_recorder = ImageRecorder(init_node=False)
        self.gripper_command = JointSingleCommand(name="gripper")

    def setup_robots(self):
        setup_puppet_bot(self.puppet_bot_left)
        setup_puppet_bot(self.puppet_bot_right)

    def get_qpos(self):
        left_qpos_raw = self.recorder_left.qpos
        right_qpos_raw = self.recorder_right.qpos
        
        if left_qpos_raw is None: left_qpos_raw = [0.0] * 8
        if right_qpos_raw is None: right_qpos_raw = [0.0] * 8
            
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[7])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[7])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    def get_qvel(self):
        left_qvel_raw = self.recorder_left.qvel
        right_qvel_raw = self.recorder_right.qvel
        
        if left_qvel_raw is None: left_qvel_raw = [0.0] * 8
        if right_qvel_raw is None: right_qvel_raw = [0.0] * 8
            
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[7])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[7])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    def get_effort(self):
        left_effort_raw = self.recorder_left.effort
        right_effort_raw = self.recorder_right.effort
        
        if left_effort_raw is None: left_effort_raw = [0.0] * 7
        if right_effort_raw is None: right_effort_raw = [0.0] * 7
            
        return np.concatenate([left_effort_raw[:7], right_effort_raw[:7]])

    def get_images(self):
        return self.image_recorder.get_images()

    def set_gripper_pose(self, left_gripper_desired_pos_normalized, right_gripper_desired_pos_normalized):
        # Process both grippers
        self.gripper_command.cmd = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(left_gripper_desired_pos_normalized)
        self.puppet_bot_left.gripper.core.pub_single.publish(self.gripper_command)

        self.gripper_command.cmd = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(right_gripper_desired_pos_normalized)
        self.puppet_bot_right.gripper.core.pub_single.publish(self.gripper_command)

    def _reset_joints(self):
        pose_with_flip = list(START_ARM_POSE[:6])
        pose_with_flip[1] = -pose_with_flip[1]
        pose_with_flip[2] = -pose_with_flip[2]
        
        # Both targets are physical Puppets, so both need the flipped frame profile
        move_arms(
            [self.puppet_bot_left, self.puppet_bot_right], 
            [pose_with_flip, pose_with_flip], 
            move_time=1
        )

    def _reset_gripper(self):
        move_grippers([self.puppet_bot_left, self.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)
        move_grippers([self.puppet_bot_left, self.puppet_bot_right], [PUPPET_GRIPPER_JOINT_CLOSE] * 2, move_time=1)

    def get_observation(self):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos()
        obs['qvel'] = self.get_qvel()
        obs['effort'] = self.get_effort()
        obs['images'] = self.get_images()
        return obs

    def get_reward(self):
        return 0

    def reset(self, fake=False):
        if not fake:
            self.puppet_bot_left.core.robot_reboot_motors("single", "gripper", True)
            self.puppet_bot_right.core.robot_reboot_motors("single", "gripper", True)
            self._reset_joints()
            self._reset_gripper()
        return dm_env.TimeStep(step_type=dm_env.StepType.FIRST, reward=self.get_reward(), discount=None, observation=self.get_observation())

    def step(self, action):
        state_len = int(len(action) / 2)
        left_action = action[:state_len]
        right_action = action[state_len:]

        left_arm_action = list(left_action[:6])
        right_arm_action = list(right_action[:6])

        # Apply the model-based inversion to BOTH physical puppet arms (VX300s)
        left_arm_action[1] = -left_arm_action[1]
        left_arm_action[2] = -left_arm_action[2]

        right_arm_action[1] = -right_arm_action[1]
        right_arm_action[2] = -right_arm_action[2]

        self.puppet_bot_left.arm.set_joint_positions(left_arm_action, blocking=False)
        self.puppet_bot_right.arm.set_joint_positions(right_arm_action, blocking=False)
        self.set_gripper_pose(left_action[-1], right_action[-1])
        
        # time.sleep(DT)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation()
        )

def get_action(master_bot_left, master_bot_right):
    action = np.zeros(14)
    action[:6] = master_bot_left.core.joint_states.position[:6]
    action[6] = MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_left.core.joint_states.position[6])
    action[7:13] = master_bot_right.core.joint_states.position[:6]
    action[13] = MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_right.core.joint_states.position[6])
    return action

def make_real_env(init_node, setup_robots=True):
    return RealEnv(init_node, setup_robots)


def test_real_teleop():
    """Test bimanual teleoperation using the updated global node schema."""
    onscreen_render = True
    render_cam = 'cam_left_wrist'

    global_node = create_interbotix_global_node()

    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', node=global_node)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'master_right', node=global_node)
    
    robot_startup(global_node)
    
    setup_master_bot(master_bot_left)
    setup_master_bot(master_bot_right)

    # setup the environment
    env = make_real_env(init_node=False)
    ts = env.reset(fake=True)
    episode = [ts]
    
    if onscreen_render:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam])
        plt.ion()

    for t in range(1000):
        action = get_action(master_bot_left, master_bot_right)
        ts = env.step(action)
        episode.append(ts)

        if onscreen_render:
            plt_img.set_data(ts.observation['images'][render_cam])
            plt.pause(DT)
        else:
            time.sleep(DT)


if __name__ == '__main__':
    test_real_teleop()