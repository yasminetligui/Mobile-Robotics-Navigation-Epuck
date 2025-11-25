from threading import Thread
import time
from modules.cv import Camera, get_map_info, get_robot_info
from modules.controller import Controller
from modules.global_navigation import GlobalNavigation
from modules.kalman_filter import KalmanFilter
from modules.local_navigation import LocalNavigation
from modules.motor import PIDController
from modules.robot import RobotInstruction
from modules.utils import CancellationToken
from modules.visualizer import Visualizer
from definitions import *

class MainLogic() :

    def __init__(self) :
        self.camera = Camera()
        self.robot = RobotInstruction()
        self.robot.disable_comments()
        self.pid = PIDController(
            kp_linear=KP_LINEAR, ki_linear=KI_LINEAR, kd_linear=KD_LINEAR,
            kp_angular=KP_ANGULAR, ki_angular=KI_ANGULAR, kd_angular=KD_ANGULAR,
            angular_speed_correction=ANGULAR_SPEED_CORRECTION,
            max_speed=MAX_ROBOT_SPEED
        )
        self.local_navigation = LocalNavigation(
            avoiding_period=AVOIDING_PERIOD,
            detection_distance=OBSTACLE_DETECTION_MAX,
            start_avoidance_thres=START_AVOIDANCE_DIST,
            stop_avoidance_thres=STOP_AVOIDANCE_DIST,
            max_speed=MAX_ROBOT_SPEED
        )
        self.map_info = get_map_info(
            self.camera.get_frame(),
            target_scale=MAP_TARGET_SCALE,
            descretize_scale=DESCRITIZE_SCALE,
            descretize_threshold=30,        # This number needs to be changed based on the environment
            aruco_cm=ARUCO_SIZE_CM
        )
        self.visualizer = Visualizer(
            original_map=self.map_info.original_map,
            grid_map=self.map_info.descretized_map,
            descretize_scale=self.map_info.descretize_scale,
            robot_width=ROBOT_RADIUS_CM,
        )
        self.global_navigation = GlobalNavigation(
            self.map_info.original_map,
            DESCRITIZE_SCALE,
            DESCRITIZE_THRESHOLD,
            ROBOT_RADIUS_CM / self.map_info.cm_per_pixel,
            REMOVE_SMALL_OBSTACLES_IN_CM2 // (self.map_info.cm_per_pixel ** 2),
        )
        self.kalman = KalmanFilter(
            t_frame=SHUTTER_SPEED,              # ~0.01 s
            d_pixel=self.map_info.cm_per_pixel, # ~0.1 cm
            d_robot=ROBOT_RADIUS_CM,            # ~10 cm
            d_aruco=ARUCO_SIZE_CM,              # ~5 cm
            v_error=MOTOR_SPEED_ERROR,          # ~5% error
        )
        self.controller = Controller(
            camera=self.camera,
            robot=self.robot,
            kalman=self.kalman,
            pid=self.pid,
            visualizer=self.visualizer,
            local_navigation=self.local_navigation,
            global_navigation=self.global_navigation,
            map_info=self.map_info,
            threshold_distance_to_target_cm=THRESHOLD_DISTANCE_TO_TARGET_CM,
            motor_speed_multiplier=MOTOR_SPEED_TO_CM
        )
    
    def get_checkpoints(self):
        """
        Retrieves the list of checkpoints detected by the robot's camera.
        This method captures a frame from the robot's camera and processes it to 
        extract information about the robot's environment, including the list of 
        checkpoints. It asserts that at least one checkpoint is detected.

        Returns:
            list: A list of checkpoints detected by the robot's camera.
        Raises:
            AssertionError: If no checkpoints are detected.
        """

        robot_info = get_robot_info(
            self.camera.get_frame(),
            self.map_info.original_width,
            self.map_info.original_height
        )

        assert len(robot_info.checkpoint_list) > 0, "No checkpoints detected"
        return robot_info.checkpoint_list
    
    def listen_for_target(self):
        """
        Continuously listens for a new target position from the camera feed and updates the next target if a significant change is detected.
        This method runs an infinite loop where it captures frames from the camera, extracts robot information, and checks if a new goal is available. If the new goal position differs significantly from the last target, it updates the next target and cancels the current path.
        
        Attributes:
            last_target (list): The last known target coordinates [x, y].
        Updates:
            self.next_target (tuple): The next target coordinates (y, x).
            self.cancel_token.cancel(): Cancels the current path.
        Note:
            The loop includes a sleep interval of 1 second to prevent excessive CPU usage.
        """

        last_target = [-1, -1]

        while True:
            frame = self.camera.get_frame()
            robot_info = get_robot_info(
                frame,
                self.map_info.original_width,
                self.map_info.original_height
            )

            pixel_threshold = TARGET_CHANGE_THRESHOLD / self.map_info.cm_per_pixel
            if robot_info.goal_available and (abs(robot_info.goal_x - last_target[0]) > pixel_threshold or abs(robot_info.goal_y - last_target[1]) > pixel_threshold):
                last_target = (robot_info.goal_x, robot_info.goal_y)
                
                # set next target
                self.next_target = (robot_info.goal_y, robot_info.goal_x)
                # cancel current path
                self.cancel_token.cancel()
            else:
                time.sleep(1)
    
    def get_next_waypoint(self):
        """
        Generator function to get the next waypoint for the robot to navigate to.

        This function yields the next target waypoint if `self.next_target` is set.
        If `self.next_target` is None, it yields the next checkpoint from the `self.checkpoints` list,
        cycling through the list in a circular manner.

        Yields:
            tuple: The next waypoint or checkpoint coordinates.
        """
        while True:
            if self.next_target is not None:
                target = self.next_target
                self.next_target = None
                yield target
            else:
                next_checkpoint = self.checkpoints[self.next_checkpoint_index]
                self.next_checkpoint_index = (self.next_checkpoint_index + 1) % len(self.checkpoints)
                yield next_checkpoint

    def run(self):
        """
        Executes the main logic for navigating through checkpoints.

        This method performs the following steps:
        1. Retrieves the list of checkpoints.
        2. Initializes the next checkpoint index and target.
        3. Starts a listener thread to listen for target updates.
        4. Iterates through the waypoints and navigates to each target.
        5. Prints the current position and the next target.
        6. Creates a cancellation token and runs the controller to move to the next target.

        If an exception occurs, it ensures the robot stops moving.

        Raises:
            Any exceptions that occur during the execution of the method.
        """
        try:
            self.checkpoints = self.get_checkpoints()
            self.next_checkpoint_index = 0

            self.next_target = None
            listener_thread = Thread(target=self.listen_for_target)
            listener_thread.start()

            for next_target in self.get_next_waypoint():
                print("Current position", self.controller.current_position)
                print("Going to", next_target)

                self.cancel_token = CancellationToken()
                self.controller.run(next_target, self.cancel_token)
        finally:
            self.robot.stop_moving()
