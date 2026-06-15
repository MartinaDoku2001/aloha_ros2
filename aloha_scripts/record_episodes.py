import os
import time
import h5py
import argparse
import numpy as np
from tqdm import tqdm
import threading
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

# Import pynput for non-blocking keyboard listening
from pynput import keyboard

from constants import DT, START_ARM_POSE, TASK_CONFIGS
from constants import MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE, PUPPET_GRIPPER_JOINT_OPEN
from robot_utils import Recorder, ImageRecorder, get_arm_gripper_positions
from robot_utils import move_arms, torque_on, torque_off, move_grippers
from real_env import make_real_env, get_action

from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
    robot_startup
)

# Global flag to track early termination
break_episode = False

def on_press(key):
    global break_episode
    try:
        # You can change 'q' to any key you prefer
        if key.char == 'q':
            print("\n[INFO] 'q' pressed! Ending episode early...")
            break_episode = True
    except AttributeError:
        pass

def opening_ceremony(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right):
    """ Move all robots to a pose where it is easy to start demonstration """
    puppet_bot_left.core.robot_reboot_motors("single", "gripper", True)
    puppet_bot_left.core.robot_set_operating_modes("group", "arm", "position")
    puppet_bot_left.core.robot_set_operating_modes("single", "gripper", "current_based_position")
    master_bot_left.core.robot_set_operating_modes("group", "arm", "position")
    master_bot_left.core.robot_set_operating_modes("single", "gripper", "position")
    
    puppet_bot_right.core.robot_reboot_motors("single", "gripper", True)
    puppet_bot_right.core.robot_set_operating_modes("group", "arm", "position")
    puppet_bot_right.core.robot_set_operating_modes("single", "gripper", "current_based_position")
    master_bot_right.core.robot_set_operating_modes("group", "arm", "position")
    master_bot_right.core.robot_set_operating_modes("single", "gripper", "position")

    torque_on(puppet_bot_left)
    torque_on(master_bot_left)
    torque_on(puppet_bot_right)
    torque_on(master_bot_right)

    pose_no_flip = list(START_ARM_POSE[:6])
    pose_with_flip = list(START_ARM_POSE[:6])
    pose_with_flip[1] = -pose_with_flip[1]
    pose_with_flip[2] = -pose_with_flip[2]
    
    move_arms(
        [master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], 
        [pose_no_flip, pose_with_flip, pose_no_flip, pose_with_flip], 
        move_time=1.5
    )
    
    move_grippers(
        [master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], 
        [MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE, MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE], 
        move_time=0.5
    )

    master_bot_left.core.robot_torque_enable("single", "gripper", False)
    master_bot_right.core.robot_torque_enable("single", "gripper", False)
    
    print(f'Close the right master gripper to start')
    close_thresh = -0.03
    pressed = False
    
    while not pressed:
        gripper_pos_right = get_arm_gripper_positions(master_bot_right)
        print(f"\rWaiting for right trigger... Gripper Pos: {gripper_pos_right:.4f}", end="")
        if gripper_pos_right < close_thresh: 
            pressed = True
        time.sleep(DT)
        
    torque_off(master_bot_left)
    torque_off(master_bot_right)
    print(f'\nStarted!')

def capture_one_episode(dt, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite):
    global break_episode
    break_episode = False # Reset flag for new episode

    print(f'Dataset name: {dataset_name}')
    env = make_real_env(init_node=False, setup_robots=True)

    try:
        global_node = get_interbotix_global_node()
    except NameError:
        global_node = None
    if global_node is not None:
        robot_startup(global_node)

    recording_executor = MultiThreadedExecutor()
    if hasattr(env, 'recorder_left') and isinstance(env.recorder_left, Node):
        recording_executor.add_node(env.recorder_left)
    if hasattr(env, 'recorder_right') and isinstance(env.recorder_right, Node):
        recording_executor.add_node(env.recorder_right)
    if hasattr(env, 'image_recorder') and isinstance(env.image_recorder, Node):
        recording_executor.add_node(env.image_recorder)
        
    rec_spin_thread = threading.Thread(target=recording_executor.spin, daemon=True)
    rec_spin_thread.start()

    master_bot_left = env.master_bot_left
    master_bot_right = env.master_bot_right
    puppet_bot_left = env.puppet_bot_left
    puppet_bot_right = env.puppet_bot_right

    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path + '.hdf5') and not overwrite:
        print(f'Dataset already exists... Hint: set overwrite to True.')
        return False

    opening_ceremony(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right)
    print("---opening")

    ts = env.reset(fake=True)
    timesteps = [ts]
    actions = []
    actual_dt_history = []
    
    # --- START KEYBOARD LISTENER ---
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    print(">> Press 'q' at any time to finish the episode early. <<")

    for t in tqdm(range(max_timesteps)):
        if break_episode:
            break  # Break out of the loop early

        t0 = time.time()
        action = get_action(master_bot_left, master_bot_right)
        t1 = time.time()
        
        ts = env.step(action)
        t2 = time.time()
        
        timesteps.append(ts)
        actions.append(action)
        actual_dt_history.append([t0, t1, t2])
        
        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    # Stop keyboard listener
    listener.stop()

    freq_mean = print_dt_diagnosis(actual_dt_history)
    if freq_mean < 42:
        print("Warning: Step frequency too low, episode might be unhealthy.")
    
    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
    }
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    # Count how many steps were actually recorded
    actual_timesteps = len(actions)

    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])
        data_dict['/action'].append(action)
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

    t0 = time.time()
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        root.attrs['sim'] = False
        obs = root.create_group('observations')
        image = obs.create_group('images')
        
        # CRITICAL FIX: Use 'actual_timesteps' instead of 'max_timesteps' 
        # so HDF5 allocations match the shortened data array shapes.
        for cam_name in camera_names:
            _ = image.create_dataset(cam_name, (actual_timesteps, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3))
        _ = obs.create_dataset('qpos', (actual_timesteps, 14))
        _ = obs.create_dataset('qvel', (actual_timesteps, 14))
        _ = obs.create_dataset('effort', (actual_timesteps, 14))
        _ = root.create_dataset('action', (actual_timesteps, 14))

        for name, array in data_dict.items():
            root[name][...] = array
            
    print(f'Saving Complete ({actual_timesteps}/{max_timesteps} steps): {time.time() - t0:.1f} secs')
    return True

def main(args):
    try:
        global_node = get_interbotix_global_node()
    except NameError:
        global_node = create_interbotix_global_node()
        
    if global_node is None:
        global_node = create_interbotix_global_node()
        
    task_config = TASK_CONFIGS[args['task_name']]
    dataset_dir = task_config['dataset_dir']
    max_timesteps = task_config['episode_len']
    camera_names = task_config['camera_names']

    if args['episode_idx'] is not None:
        episode_idx = args['episode_idx']
    else:
        episode_idx = get_auto_index(dataset_dir)
    overwrite = True

    dataset_name = f'episode_{episode_idx}'
    print(dataset_name + '\n')
    
    while True:
        is_healthy = capture_one_episode(DT, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite)
        if is_healthy:
            break
        else:
            print("Episode capture failed or was unhealthy. Retrying...")

def get_auto_index(dataset_dir, dataset_name_prefix='', data_suffix='hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")

def print_dt_diagnosis(actual_dt_history):
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]

    dt_mean = np.mean(total_time)
    freq_mean = 1 / dt_mean
    print(f'Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3f} Step env: {np.mean(step_env_time):.3f}')
    return freq_mean

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=None, required=False)
    main(vars(parser.parse_args()))
    import os
    os._exit(0)