
import enum
from multiprocessing import allow_connection_pickling
from threading import Lock
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from typing import Any, Optional, Tuple, List
import time

class MapInfo() :

    raw_image: np.ndarray
    original_map: np.ndarray
    descretized_map: np.ndarray
    original_width: int
    original_height: int
    descretized_width: int
    descretized_height: int
    descretize_scale: int
    cm_per_pixel: float
    transform: Any
    aruco_corners: List[np.ndarray]

    def transform_frame(self, frame) :
        """
        Applies a perspective transformation to the given frame.

        Parameters:
        frame (numpy.ndarray): The input image/frame to be transformed.

        Returns:
        numpy.ndarray: The transformed image/frame.
        """

        return cv2.warpPerspective(frame, self.transform, (self.original_width, self.original_height))
    
class RobotInfo() :
    robot_available: bool
    robot_x: float
    robot_y: float
    robot_theta: float
    goal_available: bool
    goal_x: float
    goal_y: float
    goal_theta: float
    checkpoint_available: bool
    checkpoint_list: np.ndarray

class Camera() :

    def __init__(self) :

        self.vc = cv2.VideoCapture(0)
        time.sleep(1)
        rval, frame = self.vc.read()

        self.lock = Lock()
    
    def get_frame(self) :
        """
        Captures a frame from the video capture device. This function is thread-safe.

        This method acquires a lock before attempting to read a frame from the video capture device.
        If the device is opened successfully, it reads a frame, converts its color from BGR to RGB,
        releases the lock, and returns the frame. If the device is not opened, it prints an error
        message, releases the lock, and returns None.

        The lock is to prevent race from two different threads.

        Returns:
            numpy.ndarray: The captured frame with colors converted from BGR to RGB, or None if the
                           video capture device is not opened.
        """
        self.lock.acquire()
        if self.vc.isOpened() :
            rval, frame = self.vc.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.lock.release()
            return frame
        else :
            print("Camera not opened")
            self.lock.release()
            return None

    def get_frame_when_esc(self) :
        """
        Capture and return a video frame when the ESC key is pressed.
        This method opens a video capture window and continuously reads frames from the video capture device.
        It displays each frame in a window named "preview". When the ESC key (key code 27) is pressed, the
        current frame is returned. If the video capture device is not opened, the method returns None.
        
        Returns:
            frame (numpy.ndarray): The video frame captured when the ESC key is pressed.
            None: If the video capture device is not opened.
        """

        cv2.namedWindow("preview")
        while self.vc.isOpened() :
            rval, frame = self.vc.read()
            cv2.imshow("preview", frame)
            key = cv2.waitKey(20)
            if key == 27:   # exit on ESC
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame
        
        return None

def display_image(image, title = None):
    """
    Display image using matplotlib.

    Parameters:
        image (numpy.ndarray): The image to display.
        title (str, optional): The title of the image. Default is None.
    """
    plt.imshow(image, interpolation='nearest')
    plt.axis('off')  # Hide axes

    if title is not None:
        plt.title(title)
    plt.show()

def read_image(path) :
    """
    Read image from the path.

    Parameters:
        path (str): The path to the image file.
    """
    return cv2.imread(path)

def generate_aruco(id = 0, size = 100) :
    """
    Generate aruco images.

    Parameters:
        id (int): The id of the aruco marker.
        size (int): The size of the aruco marker.
    """

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

    marker_id = id
    marker_size = size  # Size in pixels
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

    return marker_image

def _get_aruco_pose(frame) :
    """
    Detects ArUco markers in a given frame and returns the frame with detected markers drawn, 
    along with the corners and ids of the detected markers.

    Parameters:
        frame (numpy.ndarray): The input image frame in which to detect ArUco markers.

    Returns:
        tuple: A tuple containing:
            - draw_frame (numpy.ndarray): The input frame with detected markers drawn.
            - corners (list): A list of corners of the detected markers.
            - ids (numpy.ndarray): The ids of the detected markers.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, rejected = detector.detectMarkers(gray)

    # print("Detected markers:", ids)
    draw_frame = frame.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(draw_frame, corners, ids)
    return draw_frame, corners, ids
    
def capture_map(path, map_id) :
    """
    Captures an image from the camera and saves it as a map image file.
    
    Parameters:
        path (str): The directory path where the map image will be saved.
        map_id (int): The identifier for the map image file.
    Raises:
        Exception: If the image could not be written to the specified path.
    Returns:
        None
    """


    camera = Camera()
    frame = camera.get_frame_when_esc()
    plt.imshow(frame, interpolation='nearest')
    if not cv2.imwrite(os.path.join(path, f"map{map_id}.png"), frame) :
        raise Exception("Could not write image")
    plt.axis('off')  # Hide axes
    plt.title(f'map')
    plt.show()

def _get_persepective_transform(camera_pts, scaled_width=None, scaled_height=None, target_scale=None) :
    """
    Computes the perspective transform matrix and optionally scales the width and height.

    Parameters:
    camera_pts (numpy.ndarray): A 4x2 array of points representing the coordinates in the camera view.
    scaled_width (int, optional): The desired width of the transformed image. If not provided, it will be calculated based on target_scale.
    scaled_height (int, optional): The desired height of the transformed image. If not provided, it will be calculated based on target_scale.
    target_scale (float, optional): The scale factor to compute the scaled width and height if they are not provided.

    Returns:
    tuple: A tuple containing:
        - persp_trans (numpy.ndarray): The perspective transform matrix.
        - scaled_width (int): The width of the transformed image.
        - scaled_height (int): The height of the transformed image.
    """

    measured_height = np.linalg.norm(camera_pts[0] - camera_pts[3])
    measured_width = np.linalg.norm(camera_pts[1] - camera_pts[3])

    if not scaled_width or not scaled_height :
        assert(target_scale is not None)
        scaled_height = int(target_scale * measured_height / (measured_height + measured_height))
        scaled_width = int(target_scale * measured_width / (measured_height + measured_height))
        # print("Generating new width and height:", scaled_width, scaled_height)

    target_pts = np.array(
        [[0, scaled_height], [scaled_width, 0], [scaled_width, scaled_height], [0, 0]]
    ).astype(np.float32)

    persp_trans = cv2.getPerspectiveTransform(camera_pts, target_pts)

    return persp_trans, scaled_width, scaled_height


def _descretize_map(frame, grid_size, threshold = 100) :
    """
    Discretizes a given frame into a grid based on the specified grid size and threshold.

    Parameters:
        frame (numpy.ndarray): The input image/frame to be discretized.
        grid_size (int): The size of each grid cell.
        threshold (int, optional): The threshold value to determine the binary state of each grid cell. Default is 100.

    Returns:
        numpy.ndarray: A 2D array representing the discretized grid where each cell is either 0 or 1.
    """

    height, width, _ = frame.shape
    grid_height = height // grid_size
    grid_width = width // grid_size

    grid = np.zeros((grid_height, grid_width))

    for i in range(grid_height) :
        for j in range(grid_width) :
            grid[i, j] = np.mean(frame[i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size])
            if grid[i, j] < threshold :
                grid[i, j] = 0
            else :
                grid[i, j] = 1
    return grid

def _get_robot_and_goal_pose(image, scaled_width, scaled_height) :
    """
    Detects the robot and goal poses from an image using ArUco markers.

    Parameters:
        image (numpy.ndarray): The input image containing ArUco markers.
        scaled_width (int): The width to scale the image to.
        scaled_height (int): The height to scale the image to.

    Returns:
        tuple: A tuple containing:
            - robot_available (bool): Whether the robot marker was detected.
            - robot_center_y (float): The y-coordinate of the robot's center.
            - robot_center_x (float): The x-coordinate of the robot's center.
            - robot_theta (float): The orientation of the robot in radians.
            - goal_available (bool): Whether the goal marker was detected.
            - goal_center_y (float): The y-coordinate of the goal's center.
            - goal_center_x (float): The x-coordinate of the goal's center.
            - goal_theta (float): The orientation of the goal in radians.
            - map_available (bool): Whether the map markers were detected.
            - checkpoint_list (numpy.ndarray): An array of waypoints detected.
            - scaled_width (int): The scaled width of the image.
            - scaled_height (int): The scaled height of the image.
    """

    frame, corners, ids = _get_aruco_pose(image)

    robot_available = False
    goal_available = False
    robot_corners = np.zeros((1, 4, 2))
    goal_corners = np.zeros((1, 4, 2))
    map_corners = np.zeros((4, 4, 2))
    waypoint_corners = []
    corners_get = 0
    if ids is not None :
        for i, ids in enumerate(ids) :
            if ids[0] == 4 :
                robot_corners = corners[i]
                robot_available = True
            elif ids[0] == 5:
                goal_corners = corners[i]
                goal_available = True
            elif ids[0] < 4 :
                map_corners[ids[0]] = corners[i]
                corners_get += 1
            elif ids[0] >=6 : # Way points
                waypoint_corners.append((ids[0].item(), corners[i]))
    
    if corners_get < 4 :
        print("WANRNING! Could not detect all markers")
        return (
            False,
            0,
            0,
            0,
            False,
            0,
            0,
            0,
            False,
            None,
            scaled_width,
            scaled_height
        )
        # raise Exception("Could not detect all markers")
    
    # print("Waypoints:", waypoint_corners)
    if len(waypoint_corners) > 0 :

        waypoint_corners = sorted(waypoint_corners, key=lambda x: x[0])
        waypoint_corners = np.concatenate([val for key, val in waypoint_corners])
    
    camera_pts = np.array(
        [
            map_corners[0][1],
            map_corners[1][3],
            map_corners[2][0],
            map_corners[3][2]
        ]
    ).astype(np.float32)
    
    persp_transform, scaled_width, scaled_height = _get_persepective_transform(
        camera_pts,
        scaled_width=scaled_width,
        scaled_height=scaled_height
    )
    
    robot_corners = cv2.perspectiveTransform(robot_corners, persp_transform)[0]
    goal_corners = cv2.perspectiveTransform(goal_corners, persp_transform)[0]
    if len(waypoint_corners) > 0:
        waypoint_corners = cv2.perspectiveTransform(waypoint_corners, persp_transform)
    else :
        waypoint_corners = np.zeros((0, 2))
    
    robot_center = np.mean(robot_corners, axis=0)
    x_axis = robot_corners[1] - robot_corners[2]
    theta = np.arctan2(x_axis[1], x_axis[0])

    goal_center = np.mean(goal_corners, axis=0)
    x_axis = goal_corners[1] - goal_corners[2]
    goal_theta = np.arctan2(x_axis[1], x_axis[0])

    checkpoint_list = np.mean(waypoint_corners, axis=1)

    return (
        robot_available,
        robot_center[1].item(),
        robot_center[0].item(),
        theta.item(),
        goal_available,
        goal_center[1].item(),
        goal_center[0].item(),
        goal_theta.item(),
        True,
        checkpoint_list,
        scaled_width,
        scaled_height
    )

def _remove_arucos(
    map_frame: np.ndarray,
    aruco_corners: np.ndarray,
) :
    """
    Removes ArUco markers from a given map frame by filling their regions with white color.

    Parameters:
        map_frame (np.ndarray): The input image/frame from which ArUco markers need to be removed.
        aruco_corners (np.ndarray): An array of ArUco marker corner points.

    Returns:
        np.ndarray: The modified map frame with ArUco markers removed.
    """
    map_frame = map_frame.astype(np.uint8)
    for aruco in aruco_corners :
        cv2.fillPoly(map_frame, np.int32([aruco]), color=(255, 255, 255))
    return map_frame

def get_map_info(
        map_frame: np.ndarray,
        target_scale=2048,
        descretize_scale=6,
        descretize_threshold=30,
        aruco_cm=5
    ) -> MapInfo :

    """
    Processes a map frame to extract and transform map information using ArUco markers.
    
    Parameters:
        map_frame (np.ndarray): The input map frame as a numpy array.
        target_scale (int, optional): The target scale for the perspective transformation. Default is 2048.
        descretize_scale (int, optional): The scale factor for discretizing the map. Default is 6.
        descretize_threshold (int, optional): The threshold value for discretizing the map. Default is 30.
        aruco_cm (int, optional): The size of the ArUco marker in centimeters. Default is 5.
    
    Returns:
        MapInfo: An object containing various information about the processed map, including the original map,
                    discretized map, transformation matrix, and ArUco marker corners.
    
    Raises:
        Exception: If less than 4 ArUco markers are detected in the map frame.
    """


    masked_frame, corners, ids = _get_aruco_pose(map_frame.copy())

    if len(ids) < 4 :
        raise Exception("Could not detect all markers")

    corners_sorted = np.zeros((4, 4, 2))
    vis = [0, 0, 0, 0]
    for i in range(len(ids)) :
        if ids[i][0] < 4 :
            corners_sorted[ids[i][0]] = corners[i]
            vis[ids[i][0]] = 1
        
    for i in range(4) :
        if vis[i] == 0 :
            raise Exception("Could not detect all markers")
    
    camera_pts = np.array(
        [
            corners_sorted[0][1],
            corners_sorted[1][3],
            corners_sorted[2][0],
            corners_sorted[3][2]
        ]
    ).astype(np.float32)

    persp_trans, scaled_width, scaled_height = _get_persepective_transform(
        camera_pts,
        target_scale=target_scale,
    )

    original_map = cv2.warpPerspective(map_frame, persp_trans, (scaled_width, scaled_height))

    corners_sorted_transformed = cv2.perspectiveTransform(corners_sorted, persp_trans)

    map_corner_diag = corners_sorted_transformed[:, 0] - corners_sorted_transformed[:, 2]
    map_corner_dist = np.linalg.norm(map_corner_diag, axis=1) / np.sqrt(2)
    cm_per_pixel = aruco_cm / np.mean(map_corner_dist)

    all_corners_transformed = cv2.perspectiveTransform(np.concatenate(corners), persp_trans)

    map_info = MapInfo()
    map_info.raw_image = map_frame
    map_info.original_map = _remove_arucos(original_map, all_corners_transformed)
    map_info.original_width = scaled_width
    map_info.original_height = scaled_height
    map_info.descretized_map = _descretize_map(original_map, descretize_scale, threshold=descretize_threshold)
    map_info.descretized_height = map_info.descretized_map.shape[0]
    map_info.descretized_width = map_info.descretized_map.shape[1]
    map_info.descretize_scale = descretize_scale
    map_info.cm_per_pixel = cm_per_pixel
    map_info.transform = persp_trans
    map_info.aruco_corners = all_corners_transformed

    return map_info


def get_robot_info(image: np.ndarray, scaled_width: int, scaled_height: int) -> RobotInfo:
    """
    Extracts and returns information about the robot and its goal from the given image.

    Parameters:
        image (np.ndarray): The input image from which to extract robot and goal information.
        scaled_width (int): The scaled width of the image.
        scaled_height (int): The scaled height of the image.

    Returns:
        RobotInfo: An object containing information about the robot's position, orientation, 
                   goal position, goal orientation, and checkpoints.
    """

    (
        robot_available,
        robot_center_x, 
        robot_center_y,
        robot_theta,
        goal_available,
        goal_center_x,
        goal_center_y,
        goal_theta,
        checkpoint_available,
        checkpoint_list,
        scaled_width,
        scaled_height
    ) = _get_robot_and_goal_pose(image, scaled_width, scaled_height)

    robot_info = RobotInfo()
    robot_info.robot_available = robot_available
    robot_info.robot_x = robot_center_x
    robot_info.robot_y = robot_center_y
    robot_info.robot_theta = robot_theta
    robot_info.goal_available = goal_available
    robot_info.goal_x = goal_center_x
    robot_info.goal_y = goal_center_y
    robot_info.goal_theta = goal_theta
    robot_info.checkpoint_available = checkpoint_available
    robot_info.checkpoint_list = checkpoint_list

    return robot_info

if __name__ == "__main__" :

    # capture_map("./resources/map_captures", 4)
    # map_frame = cv2.imread("../resources/map_captures/map3.png")
    # map_scaled = _transform_map(map_frame)
    # map_descretized = _descretize_map(map_scaled, 6)

    # plt.imshow(map_descretized, interpolation='nearest')
    # plt.axis('off')  # Hide axes
    # plt.title(f'map')
    # plt.show()
    # plt.imshow(map_scaled, interpolation='nearest')
    # plt.axis('off')  # Hide axes
    # plt.title(f'map')
    # plt.show()

    # aruco6 = generate_aruco(6, 640)
    # aruco7 = generate_aruco(7, 640)
    # aruco8 = generate_aruco(8, 640)
    # cv2.imwrite("resources/aruco/marker6.png", aruco6)
    # cv2.imwrite("resources/aruco/marker7.png", aruco7)
    # cv2.imwrite("resources/aruco/marker8.png", aruco8)

    # map_frame = cv2.imread("./resources/map_captures/map5.jpeg")
    # map_info = get_map_info(map_frame, target_scale=2048)
    # robot_info = get_robot_info(map_frame, scaled_width=map_info.original_width, scaled_height=map_info.original_height)
    # print(robot_info.checkpoint_list)
    capture_map("./resources/map_captures", 6)
    pass
