import time
import sys
import IPython
e = IPython.embed

from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
from constants import    get_master2puppet_joint_target, MASTER2PUPPET_JOINT_FN, DT, START_ARM_POSE, MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE
from robot_utils import torque_on, torque_off, move_arms, move_grippers, get_arm_gripper_positions
from interbotix_common_modules.common_robot.robot import (
 
    create_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)

PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -1.2500  # <-- Changed from -0.6213 to allow full close
PUPPET_GRIPPER_MAX_STEP = 0.08 # Max rad distance the gripper can travel per DT step

def prep_robots(master_bot, puppet_bot):
    # reboot gripper motors, and set operating modes for all motors
    puppet_bot.core.robot_reboot_motors("single", "gripper", True)
    puppet_bot.core.robot_set_operating_modes("group", "arm", "position")
    puppet_bot.core.robot_set_operating_modes("single", "gripper", "current_based_position")
    master_bot.core.robot_set_operating_modes("group", "arm", "position")
    master_bot.core.robot_set_operating_modes("single", "gripper", "position")
    torque_on(puppet_bot)
    torque_on(master_bot)

    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    combined_pos = [[p for p in start_arm_qpos]]
    combined_pos.append([p for p in start_arm_qpos])
    # -- flip the joint angles
    combined_pos[1][1] = -combined_pos[1][1]
    combined_pos[1][2] = -combined_pos[1][2]
    move_arms([master_bot, puppet_bot], combined_pos, move_time=1)

    # move grippers to starting position
    move_grippers([master_bot, puppet_bot], [MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE], move_time=0.5)


def press_to_start(master_bot):
    master_bot.core.robot_torque_enable("single", "gripper", False)
    print(f'Close the gripper to start')
    close_thresh = -0.03
    pressed = False
    while not pressed:  
        gripper_pos = get_arm_gripper_positions(master_bot)
        if gripper_pos < close_thresh:
            pressed = True
        time.sleep(DT/10)
    torque_off(master_bot)
    print(f'Started!')


def teleop(robot_side):
    """ A standalone function for experimenting with teleoperation. No data recording. """
    global_node = create_interbotix_global_node()
    puppet_bot = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_{robot_side}', node=global_node)
    master_bot = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper", robot_name=f'master_{robot_side}', node=global_node)
    robot_startup(global_node)

    prep_robots(master_bot, puppet_bot)
    press_to_start(master_bot)

    # 2. INITIALIZE TRACKING & GRASP STATES BEFORE THE LOOP
    gripper_command = JointSingleCommand(name="gripper")
    
    prev_master_gripper_signal = None
    object_grasped = False
    
    # Establish our working command baseline by reading the initial position of the puppet
    puppet_gripper_joint_command = get_arm_gripper_positions(puppet_bot)

    print("\nEntering Advanced Teleoperation Loop...")
    try:
        while True:
            # --- Arm Position Synchronization ---
            master_state_joints = master_bot.core.joint_states.position[:6]
            master_state_joints[1] = -master_state_joints[1]
            master_state_joints[2] = -master_state_joints[2]
            puppet_bot.arm.set_joint_positions(master_state_joints, blocking=False)
            
            # --- Advanced Gripper Architecture Loop ---
            # Read the current master joint input position
            master_gripper_signal = get_arm_gripper_positions(master_bot)
            
            if prev_master_gripper_signal is None:
                master_is_closing = False
                master_is_opening = False
            else:
                delta = master_gripper_signal - prev_master_gripper_signal
                # Checking movement direction based on your master calibration direction curves
                master_is_closing = delta < -0.002
                master_is_opening = delta > 0.002
                
            prev_master_gripper_signal = master_gripper_signal
            
            # Run the input calculation through your master-to-puppet conversion function
            puppet_gripper_joint_target = get_master2puppet_joint_target(master_gripper_signal, robot_side)
            
            # Safety clamp raw target before evaluating contact logic
            puppet_gripper_joint_target = max(PUPPET_GRIPPER_JOINT_CLOSE, min(PUPPET_GRIPPER_JOINT_OPEN, puppet_gripper_joint_target))
            
            # Read real-time physical feedback position of the puppet gripper
            actual_puppet_gripper_pos = get_arm_gripper_positions(puppet_bot)
            
            # Tracking error evaluation: Check if mechanical blockages are stopping progress
            block_error = actual_puppet_gripper_pos - puppet_gripper_joint_command
            target_delta = abs(puppet_gripper_joint_target - puppet_gripper_joint_command)
            
            # Release state check: Did user intentionally open hand wide to break an active grasp?
            if object_grasped and master_is_opening and target_delta > 0.1:
                object_grasped = False
                print("\n[gripper] Grasp released; resuming tracking.")

            # Calculate the command adjustments based on target directions
            if puppet_gripper_joint_target > puppet_gripper_joint_command:
                # Hand is opening: Always follow master tracking without constraint
                puppet_gripper_joint_command = min(
                    puppet_gripper_joint_command + PUPPET_GRIPPER_MAX_STEP,
                    puppet_gripper_joint_target,
                )
            else:
                # Hand is closing: Evaluate for physical collision checks
                if not object_grasped and block_error > 0.15 and master_is_closing:
                    object_grasped = True
                    print("\n[gripper] Contact detected! Freezing closing loop to protect hardware.")
                    # Snap command targets to actual position offsets to drop electrical winding strain
                    puppet_gripper_joint_command = max(
                        PUPPET_GRIPPER_JOINT_CLOSE, 
                        min(PUPPET_GRIPPER_JOINT_OPEN, actual_puppet_gripper_pos - 0.05)
                    )
                    
                if object_grasped:
                    # Hold absolute position freeze inside closing branch while object is grasped
                    pass
                else:
                    # No obstruction: step downward toward closed targets smoothly
                    puppet_gripper_joint_command = max(
                        puppet_gripper_joint_command - PUPPET_GRIPPER_MAX_STEP,
                        puppet_gripper_joint_target,
                    )
            
            # Final output safety boundaries clamp
            puppet_gripper_joint_command = max(PUPPET_GRIPPER_JOINT_CLOSE, min(PUPPET_GRIPPER_JOINT_OPEN, puppet_gripper_joint_command))
            
            # Display real-time telemetry metrics directly to terminal stream
            print(f"\r[{robot_side.upper()}] Master Signal: {master_gripper_signal:.3f} | Commanded Target: {puppet_gripper_joint_command:.3f} | Object Grasped: {object_grasped}", end="")
            
            # Stream the validated step out directly over your active publisher channel
            gripper_command.cmd = puppet_gripper_joint_command
            puppet_bot.gripper.core.pub_single.publish(gripper_command)
            
            # Sleep step sequence interval
            time.sleep(DT)

    except KeyboardInterrupt:
        pass
    finally:
        print('\nTeleop stopped. Disabling torque safely on all physical master and puppet joints...')
        torque_off(puppet_bot)
        torque_off(master_bot)
        print('Wipe complete. Device hardware handles freed.')


if __name__=='__main__':
    side = sys.argv[1]
    teleop(side)