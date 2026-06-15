import sys
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from robot_utils import move_arms, torque_on
# 1. Import the new required ROS 2 global node utilities
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)

def main():
    # 2. Instantiate the required shared ROS 2 node
    global_node = create_interbotix_global_node()

    # 3. Pass 'node=global_node' to every arm instead of 'init_node'
    # puppet_bot_left  = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name='puppet_left', node=global_node)
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name='puppet_right', node=global_node)
    #master_bot_left  = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper", robot_name='master_left', node=global_node)
    #master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper", robot_name='master_right', node=global_node)

    # 4. Start up the node communication layer safely
    robot_startup(global_node)

    # --- Your original logic resumes here perfectly ---
    # all_bots = [puppet_bot_left, puppet_bot_right]
    all_bots = [puppet_bot_right]
    for bot in all_bots:
        torque_on(bot)

    puppet_sleep_position = (0, -1.7, 1.55, 0.12, 0.65, 0)
    master_sleep_position = (0, -1.1, 1.24, 0, -0.24, 0)
    move_arms(all_bots, [puppet_sleep_position] * 1, move_time=2)

    # 5. Cleanly shut down the node when finished
    robot_shutdown(global_node)

if __name__ == '__main__':
    main()