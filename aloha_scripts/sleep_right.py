import sys
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from robot_utils import move_arms, torque_on
# Import the required ROS 2 global node managers
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)

def main():
    # 1. Create the shared ROS 2 node required by the new library version
    global_node = create_interbotix_global_node()

    print("Initializing robot arms...")
    # 2. Pass 'node=global_node' to every arm instead of using 'init_node'
    # puppet_bot_left  = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name='puppet_left', node=global_node)
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name='puppet_right', node=global_node)
    # master_bot_left  = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper", robot_name='master_left', node=global_node)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper", robot_name='master_right', node=global_node)

    # 3. Start up the node communication layer
    robot_startup(global_node)

    # 4. Run the original student's torque logic
    print("Torqueing on puppet arms...")
    all_bots = [ puppet_bot_right]
    for bot in all_bots:
        torque_on(bot)

    # 5. Execute the original custom sleep positions
    print("Moving arms to custom resting positions...")
    puppet_sleep_position = (0, -1.7, 1.55, 0.12, 0.65, 0)
    master_sleep_position = (0, -1.1, 1.24, 0, -0.24, 0)
    
    # Move the puppets down 8only 
    move_arms(all_bots, [puppet_sleep_position] * 1, move_time=2)
    
    # Optional: If you also want the master arms to tuck away using their custom position,
    # you can uncomment the two lines below:
    # for bot in [master_bot_left, master_bot_right]: torque_on(bot)
    move_arms([ master_bot_right], [master_sleep_position] * 1, move_time=2)

    # 6. Cleanly close out the ROS 2 session
    robot_shutdown(global_node)
    print("Arms successfully at rest!")

if __name__ == '__main__':
    main()
