from modules.robot import RobotInstruction
from modules.kalman_filter import KalmanFilter
from modules.global_navigation import GlobalNavigation, update_target_index
from modules.motor import PIDController
from modules.cv import (
    MapInfo,
    Camera,
    get_robot_info,
)
from modules.utils import CancellationToken
from modules.visualizer import Visualizer
from modules.local_navigation import LocalNavigation
import time
import numpy as np


class Controller:
    def __init__(
        self,
        camera: Camera,
        robot: RobotInstruction,
        kalman: KalmanFilter,
        pid: PIDController,
        visualizer: Visualizer,
        local_navigation: LocalNavigation,
        global_navigation: GlobalNavigation,
        map_info: MapInfo,
        threshold_distance_to_target_cm: float = 2,
        motor_speed_multiplier: float = 0.35
    ):
        """
        Initializing the controller with the given parameters.

        Args:
            camera: Camera used for mapping and detecting arucos.
            robot: Robot instruction for controlling the robot.
            kalman: Kalman filter for state estimation.
            pid: PID controller.
            visualizer: Visualizer for displaying information.
            local_navigation: Local navigation for path planning.
            global_navigation: Global navigation object for path planning.
            map_info: Map information object containing map details.
            threshold_distance_to_target_cm: Threshold distance to the target in centimeters.
            motor_speed_multiplier: Multiplier for motor speed.
        """
        self.camera = camera
        self.robot = robot
        self.kalman = kalman
        self.pid = pid
        self.visualizer = visualizer
        self.local_navigation = local_navigation
        self.global_navigation = global_navigation
        self.map_info = map_info
        self.target_index = 0
        self.current_position = np.array([0, 0])
        self.current_theta = 0
        self.threshold_distance_to_target_cm = threshold_distance_to_target_cm
        self.motor_speed_multiplier = motor_speed_multiplier

        start_position = None
        print("Waiting for the robot to be visible...")
        while start_position is None:
            robot_info = get_robot_info(
                self.camera.get_frame(),
                self.map_info.original_width,
                self.map_info.original_height
            )
            if robot_info.robot_available:
                start_position = (robot_info.robot_x, robot_info.robot_y)
        self.current_position = start_position
        print("Waiting for the robot to be connected...")

        self.robot.connect()

        print("Controller initialized")

    def __del__(self):
        self.robot.disconnect()

    def update_position(self, frame, motor_speeds, t_since_last_update):
        """
        Updates the position of the robot.

        Args:
            frame: Current frame from the camera.
            motor_speeds: Left and right motor speeds.
            t_since_last_update: Time elapsed since the last update in seconds.

        Returns:
            bool: Indicating whether the robot was hijacked.
        """

        left_speed, right_speed = motor_speeds

        robot_info = get_robot_info(
            frame,
            self.map_info.original_width,
            self.map_info.original_height
        )

        self.visualizer.set_robot_pose_observation(
            robot_info.robot_available,
            robot_info.robot_x,
            robot_info.robot_y,
            robot_info.robot_theta
        )

        self.visualizer.set_target_pose(
            robot_info.goal_available,
            robot_info.goal_x,
            robot_info.goal_y,
            robot_info.goal_theta
        )

        # Kalman filter assumes theta=arctan(y/x), while the robot gives theta=arctan(x/y)
        # Therefore, we need to swap x and y when passing to the kalman filter
        new_x_cm, new_y_cm, new_theta, maybe_hijack = self.kalman.get_prediction(
            last_x=self.current_position[0] * self.map_info.cm_per_pixel,
            last_y=self.current_position[1] * self.map_info.cm_per_pixel,
            last_theta=self.current_theta,
            measure_x=robot_info.robot_x * self.map_info.cm_per_pixel,
            measure_y=robot_info.robot_y * self.map_info.cm_per_pixel,
            measure_theta=robot_info.robot_theta,
            command_vl=left_speed * self.motor_speed_multiplier,
            command_vr=right_speed * self.motor_speed_multiplier,
            dt=t_since_last_update,
            aruco_available=robot_info.robot_available
        )
        new_x = new_x_cm / self.map_info.cm_per_pixel
        new_y = new_y_cm / self.map_info.cm_per_pixel

        self.current_position = np.array([new_x, new_y])
        self.current_theta = new_theta

        self.visualizer.set_robot_pose_prediction(
            new_x,
            new_y,
            new_theta,
            self.kalman.last_P / self.map_info.cm_per_pixel**2
        )
        self.visualizer.set_original_map(
            self.map_info.transform_frame(frame)
        )

        return maybe_hijack

    def run(self, target, cancellation_token: CancellationToken):
        """
        Main control loop for navigating the robot towards the target position.

        Args:
            target: Target position.
            cancellation_token: Token to signal cancellation of the operation.

        Steps:
            1. Initializing target index and calculating path to the target.
            2. If no path is found, it prints a message and returns.
            3. Converting path to centimeters and setting planned path in the visualizer.
            4. Capture initial frame from the camera and updates the robot's position.
            5. Loop to continuously update the robot's position and control its movement:
                - Checks for cancellation.
                - Captures a new frame and updates the robot's position.
                - If hijacking is detected, recalculates the path and resets the target index.
                - Updates the target index based on the current position.
                - Sets the next path index and updates the visualizer.
                - If the target index is None, prints a message and breaks the loop.
                - Calculates the wheel speeds using the PID controller.
                - Locks the robot to update motor speeds and detect obstacles.
                - Adjusts the motor speeds based on local navigation and obstacle detection.
            6. Finally, prints a message indicating the robot is stopping.
        """

        self.target_index = 0
        path = self.global_navigation.get_path(start=self.current_position[::-1], end=target)

        if path is None:
            print("No path found")
            return

        path_in_cm = path * self.map_info.cm_per_pixel
        self.visualizer.set_planned_path(path)

        frame = self.camera.get_frame()
        self.update_position(frame, (0, 0), 0)

        last_time = time.time()

        with self.robot.lock_thymio(variable_to_wait={"motor.left.speed", "motor.right.speed", "prox.horizontal"}) as _:
            motor_speeds = self.robot.get_motorspeeds()

        try:
            while self.target_index is not None:

                if cancellation_token.is_cancelled():
                    break

                this_time = time.time()

                frame = self.camera.get_frame()
                maybe_hijack = self.update_position(
                    frame, motor_speeds, this_time - last_time
                )

                if maybe_hijack:
                    print("Hijacking the robot")
                    path = self.global_navigation.get_path(start=self.current_position[::-1], end=target)
                    path_in_cm = path * self.map_info.cm_per_pixel
                    self.target_index = 0
                    self.visualizer.set_planned_path(path)

                last_time = this_time

                self.target_index = update_target_index(
                    self.current_position * self.map_info.cm_per_pixel,
                    path_in_cm,
                    self.target_index,
                    self.threshold_distance_to_target_cm,
                )

                self.visualizer.set_next_path_index(self.target_index)
                self.visualizer.display()

                if self.target_index is None:
                    print("Reached the end of the path")
                    break

                target_position = path_in_cm[self.target_index]

                left_speed, right_speed = self.pid.calculate_wheel_speeds(
                    current_position_cm=self.current_position * self.map_info.cm_per_pixel,
                    target_position_cm=target_position,
                    current_angle=self.current_theta
                )

                with self.robot.lock_thymio(variable_to_wait={"motor.left.speed", "motor.right.speed", "prox.horizontal"}) as _:
                    prox = self.robot.detect_obstacles()
                    left_speed, right_speed = self.local_navigation.update_speed(
                        [left_speed, right_speed], prox)

                    motor_speeds = self.robot.get_motorspeeds()

                    if abs(motor_speeds[0] - left_speed) / (abs(motor_speeds[0]) + 1) > 0.1 or abs(motor_speeds[1] - right_speed) / (abs(motor_speeds[1]) + 1) > 0.1:
                        self.robot.set_motorspeeds([left_speed, right_speed])
        finally:
            print("Stopping the robot")
            # self.robot.stop_moving()

    def stop(self):
        self.robot.connect()
        self.robot.stop_moving()
        self.robot.disconnect()
