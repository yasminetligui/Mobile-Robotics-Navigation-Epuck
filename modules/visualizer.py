import cv2
import numpy as np


class Visualizer():

    def __init__(
        self,
        original_map,
        grid_map,
        descretize_scale=6,
        robot_width=10,
        cm_per_pixel=0.1,
    ):
        """
        Initialization of the Visualizer class with the provided parameters.

        Args:
            original_map: Original map as a numpy array.
            grid_map: Grid map as a numpy array.
            descretize_scale: Scale for discretizing the map.
            robot_width: Width of the robot.
            cm_per_pixel: Conversion factor from centimeters to pixels.
        """

        self.original_map = original_map
        self.grid_map = grid_map
        self.descretize_scale = descretize_scale
        self.map_height = original_map.shape[0]
        self.map_width = original_map.shape[1]
        self.robot_observation_available = False
        self.robot_observation_x = 0
        self.robot_observation_y = 0
        self.robot_observation_theta = 0
        self.robot_pred_x = 0
        self.robot_pred_y = 0
        self.robot_pred_theta = 0
        self.target_available = False
        self.target_x = 0
        self.target_y = 0
        self.target_theta = 0
        self.next_path_index = 0
        self.planned_path = []
        self.observed_path = []
        self.observed_path_available = []
        self.predicted_path = []
        self.raw_frame = None
        self.covariance_matrix = np.array([[0,0,0],[0,0,0],[0,0,0]])
        self.show_path_length = 20

        self.robot_width = robot_width
        self.cm_per_pixel = cm_per_pixel
        pass

    def set_original_map(self, original_map):

        self.original_map = original_map

    def set_robot_pose_observation(self, available, x, y, theta):
        """
        Sets the OBSERVED pose of the robot and updates the observed path.

        Parameters:
        available: Indicates if the robot observation is available.
        x: X coordinate of observed position.
        y: Y coordinate of observed position.
        theta: Orientation of the observed position in radians.
        """

        self.robot_observation_available = available
        self.robot_observation_x = x
        self.robot_observation_y = y
        self.robot_observation_theta = theta
        self.observed_path.append((x, y))
        self.observed_path_available.append(available)

    def set_robot_pose_prediction(self, x, y, theta, cov_matrix):
        """
        Sets the PREDICTED pose of the robot and updates the predicted path.
        Sets the covariance matrix obtained from Kalman filter, used for error visualization.

        Parameters:
        x: X coordinate of predicted position.
        y: Y coordinate of predicted position.
        theta: Orientation of the predicted position in radians.
        cov_matrix: Covariance matrix representing the uncertainty of the prediction.
        """

        self.robot_pred_x = x
        self.robot_pred_y = y
        self.robot_pred_theta = theta
        self.predicted_path.append((x, y))
        self.covariance_matrix = cov_matrix

    def set_target_pose(self, available, x, y, theta):
        """
        Set the target (invader) position, used for visualization.

        Parameters:
        available: Indicates target is available.
        x: X coordinate of the target position.
        y: Y coordinate of the target position.
        theta: Orientation of the target in radians.
        """

        self.target_available = available
        self.target_x = x
        self.target_y = y
        self.target_theta = theta

    def set_planned_path(self, path):

        self.planned_path = path

    def set_next_path_index(self, index):

        if index is not None:
            self.next_path_index = index

    def _draw_text(self, image, text, position, delta=(0, 0), color = (120, 120, 120)):
        """
        Draws text below the specified position.

        Args:
            image: Map on which to draw the text.
            text: text string to be drawn.
            position: (x, y) coordinates for the center of the text.
            delta: (x, y) offset to adjust the text position.
            color: Color of the text in BGR format.
        """

        font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
        font_scale = 1  # Font scale
        thickness = 3  # Thickness of the text
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness)

        cv2.putText(
            image,
            text,
            (position[0] - text_width // 2 + delta[0],
             position[1] + text_height // 2 + delta[1]),
            font,
            font_scale,
            color,
            thickness
        )

    def _draw_linear_uncertainty(self,image, position, theta, cov_matrix, alpha=0.5):
        """
        Draws an uncertainty ellipse at the given position based on the covariance matrix obtained from Kalman filter.
        Uses eigenvalue decomposition of numpay library to draw the ellipse in its accurate rotation.

        Parameters:
            image: Map on which to draw the text.
            position: (x, y) coordinates for the center of the text.
            theta: Orientation angle of the robot in radians.
            cov_matrix: 3x3 covariance matrix representing uncertainty.
            alpha: Transparency factor of the ellipse.
        """

        overlay = image.copy()  # Create an overlay for transparency

        eig_vals, eig_vecs = np.linalg.eig(cov_matrix[:2, :2])  # Eigen decomposition of 2x2 submatrix
        eig_vals = np.nan_to_num(eig_vals)
        eig_vecs = np.nan_to_num(eig_vecs)
        major_axis_idx = np.argmax(eig_vals)
        angle = np.degrees(np.arctan2(eig_vecs[major_axis_idx, 1], eig_vecs[major_axis_idx, 0])) # rotation of ellipse

        # Computation for different cofidence intervals
        for std_multiplier, color in zip([15, 10, 5], [(205, 250, 255),(122, 160, 255), (28, 28, 183)]):

            major_axis_length = int(std_multiplier * np.sqrt(eig_vals[major_axis_idx]))  # Convert variance to standard deviation
            minor_axis_length = int(std_multiplier * np.sqrt(eig_vals[1 - major_axis_idx]))


            cv2.ellipse(overlay, position, (major_axis_length, minor_axis_length),angle, 0, 360, color, -1)
            
        cv2.addWeighted(overlay, alpha, image, 1-alpha, 0, image)

    def _draw_angular_uncertainty(self, image, position, theta, cov_matrix, line_length, alpha=0.5):
        """
        Draws an uncertainty angle at the given position based on the covariance matrix obtained from Kalman filter.
        Uses only the variance of theta.

        Parameters:
            image: Map on which to draw the text.
            position: (x, y) coordinates for the center of the text.
            theta: Orientation angle of the robot in radians.
            cov_matrix: 3x3 covariance matrix representing uncertainty.
            line_length: Length of the lines representing the uncertainty.
            alpha: Transparency factor of the ellipse.
        """

        overlay = image.copy()  # Create an overlay for transparency
        
        var_theta = cov_matrix[2, 2]
        var_theta = np.nan_to_num(var_theta) + 0.001
        std_dev_theta = np.sqrt(var_theta)

        # Computation for different cofidence intervals
        for std_multiplier, color in zip([3, 2, 1], [(205, 250, 255),(122, 160, 255), (28, 28, 183)]):

            theta_min = theta - std_multiplier * std_dev_theta
            theta_max = theta + std_multiplier * std_dev_theta

            num_points = 100
            angle = np.linspace(theta_min, theta_max, num_points)
            
            arc_points = [(int(position[0] + line_length * np.cos(t)),
                           int(position[1] + line_length * np.sin(t)))
                           for t in angle]
            
            arc_points.insert(0, position)  # Start line to the center
            arc_points.append(position)    # End line to the center

            arc_polygon = np.array(arc_points, dtype=np.int32)
            cv2.fillPoly(overlay, [arc_polygon], color)

        cv2.addWeighted(overlay, alpha, image, 1-alpha, 0, image)


    def _draw_robot_uncertainty(self, image, position, theta, robot_width_pixel, color, cov_matrix=None):
        """
        Draws the robot's linear and angular uncertainty on the map.
        """

        line_length = 1.5 * robot_width_pixel
        line_end = (int(position[0] + line_length * np.cos(theta)), 
                    int(position[1] + line_length * np.sin(theta)))

        if cov_matrix is not None:
            # Draw the angular uncertainty
            self._draw_angular_uncertainty(image, position, theta, cov_matrix, line_length, alpha=0.5)

            # Draw linear uncertainty
            self._draw_linear_uncertainty(image, position, theta, cov_matrix, alpha=0.5)


    def _draw_target(self, image, position, robot_width_pixel, color):
        """
        Draws flag pattern representing the invader at the given position.
        """

        flag_size = robot_width_pixel
        square_size = (flag_size // 4)
        flag_top_left = (position[0] - flag_size // 2, position[1] - flag_size // 2)
        flag_bottom_right = (flag_top_left[0] + flag_size, flag_top_left[1] + flag_size)

        # Draw flag pattern
        for i in range(4):
            for j in range(4):
                square_color = (0, 0, 0) if (i + j) % 2 == 0 else (255, 255, 255) # alternating black and white
                top_left = (flag_top_left[0] + j * square_size, flag_top_left[1] + i * square_size)
                bottom_right = (top_left[0] + square_size, top_left[1] + square_size)
                cv2.rectangle(image, top_left, bottom_right, square_color, -1)

        # Border arond flag
        cv2.rectangle(image, flag_top_left, flag_bottom_right, color, 2)

    def _draw_path_checkpoint(self, image, next_path_xy):
        """
        This function draws a cross and a circle at the given positon to 
        indicate a waypoint on the path.
        """

        cross_size = 15

        cv2.line(image,
                 (next_path_xy[0] - cross_size, next_path_xy[1] + cross_size), 
                 (next_path_xy[0] + cross_size, next_path_xy[1] - cross_size), 
                 (255,255,255),
                 4)
        
        cv2.line(image,
                 (next_path_xy[0] + cross_size, next_path_xy[1] + cross_size), 
                 (next_path_xy[0] - cross_size, next_path_xy[1] - cross_size), 
                 (255,255,255),
                 4)
        
        cv2.circle(image,next_path_xy,10,(255, 255, 255), -1)

    def _add_label_box(self, image):
        """
        Adds label box at the top right of the map.
        Text inside the box: 'Kalman Prediction' and 'Camera Observation'.
        """

        h, w, _ = image.shape  # image dimensions

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_lines = ['Kalman Prediction', 'Camera Observation']
        font_scale = 1
        thickness = 2
        padding = 20
        line_spacing = 10  # Space between lines

        # Calculate the total height of the text block
        text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[
            0] for line in text_lines]
        text_block_width = max([size[0] for size in text_sizes])
        text_block_height = sum([size[1]
                                for size in text_sizes]) + line_spacing

        # Box dimensions
        box_x1 = w - text_block_width - 3 * padding  # Top-left
        box_y1 = padding                             # Top-left
        box_x2 = w - padding                         # Bottom-right
        box_y2 = text_block_height + 3 * padding     # Bottom-right

        cv2.rectangle(image, (box_x1, box_y1),
                      (box_x2, box_y2), (255, 255, 255), -1)
        cv2.rectangle(image, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), 2)

        # Add each line of text inside the box
        text_x = box_x1 + padding
        text_y = box_y1 + padding
        text_colors = [(0, 0, 255), (0, 255, 0)]

        for size, line, color in zip(text_sizes, text_lines, text_colors):
            text_y += size[1]  # Add the height of the current line
            cv2.putText(image, line, (text_x, text_y),
                        font, font_scale, color, thickness)
            text_y += line_spacing  # Add spacing between lines

    def display(self):
        """
        Displays the current state of the robot and its environment on the map.

        The following elements are drawn on the map:
        - Robot observation (if available)
        - Robot prediction
        - Target position (if available)
        - Next path checkpoint (if available)
        - Future planned path
        - Observed path
        - Predicted path
        """

        map_image = self.original_map.copy().astype(np.uint8)
        map_image = cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB)

        robot_width_pixel = int(self.robot_width / self.cm_per_pixel)
        text_delta = (0, robot_width_pixel+10)

        # invert x and y
        prediction_xy = (int(self.robot_pred_y), int(self.robot_pred_x))
        observation_xy = (int(self.robot_observation_y), int(self.robot_observation_x))
        target_xy = (int(self.target_y), int(self.target_x))

        # draw robot prediction
        self._draw_robot_uncertainty(map_image, prediction_xy, self.robot_pred_theta, robot_width_pixel, color=(128,128,255),
                                  cov_matrix=self.covariance_matrix)

        # draw target if available
        if self.target_available:
            self._draw_target(map_image, target_xy, robot_width_pixel, color=(255, 0, 0))
            self._draw_text(map_image, "invader", target_xy, delta=text_delta, color=(255, 0, 0))

        # draw next path checkpoint
        if self.next_path_index < len(self.planned_path):
            next_path_xy = (int(self.planned_path[self.next_path_index][1]),
                            int(self.planned_path[self.next_path_index][0]))
            self._draw_path_checkpoint(map_image,next_path_xy)
            self._draw_text(map_image, "next waypoint", next_path_xy, delta=(
                0, robot_width_pixel//2), color=(255, 255, 255))
            dist = np.linalg.norm(
                next_path_xy-np.array([self.robot_pred_y, self.robot_pred_x])) * self.cm_per_pixel
            self._draw_text(map_image, "distance: {:.2f}".format(dist), next_path_xy,
                            delta=(0, robot_width_pixel//2+40), color=(255, 255, 255))
        
        # future path
        if self.next_path_index+1 < len(self.planned_path):
            for cur, nxt in zip(self.planned_path[self.next_path_index:-1], self.planned_path[self.next_path_index+1:]):
                cv2.line(
                    map_image,
                    (int(cur[1]), int(cur[0])),
                    (int(nxt[1]), int(nxt[0])),
                    (255, 255, 0),
                    2
                )
        
        # draw observed path
        for cur, nxt, cur_avail, nxt_avail in zip(
                self.observed_path[-self.show_path_length:][:-1],
                self.observed_path[-self.show_path_length:][1:],
                self.observed_path_available[-self.show_path_length:][:-1],
                self.observed_path_available[-self.show_path_length:][1:]):
            if not cur_avail or not nxt_avail:
                continue
            cv2.line(
                map_image,
                (int(cur[1]), int(cur[0])),
                (int(nxt[1]), int(nxt[0])),
                (0, 255, 0),
                3
            )
        
        # draw predicted path
        for cur, nxt in zip(self.predicted_path[-self.show_path_length:][:-1], self.predicted_path[-self.show_path_length:][1:]):
            cv2.line(
                map_image,
                (int(cur[1]), int(cur[0])),
                (int(nxt[1]), int(nxt[0])),
                (128, 128, 255),
                3
            )

        self._add_label_box(map_image)

        cv2.imshow("Robot Map", map_image)
        cv2.waitKey(1)  # Allows OpenCV to process events



if __name__ == "__main__":
    # Create a blank map (a simple black image)
    map_width, map_height = 1500, 1500
    original_map = np.zeros((map_height, map_width, 3), dtype=np.uint8)
    grid_map = np.zeros((map_height, map_width), dtype=np.uint8)

    # Initialize the visualizer
    visualizer = Visualizer(
        original_map=original_map,
        grid_map=grid_map,
        descretize_scale=6,
        robot_width=10,
        cm_per_pixel=0.1,
    )

    # Set the robot's initial observed position and orientation
    visualizer.set_robot_pose_observation(
        available=True,
        x=250,
        y=250,
        theta=np.pi / 4
    )

    # Set the robot's predicted position and orientation
    visualizer.set_robot_pose_prediction(
        x=580,
        y=580,
        theta=np.pi / 3,
        cov_matrix=np.array([[0,0,0],[0,0,0],[0,0,0]])
    )

    # Set a target position
    visualizer.set_target_pose(
        available=True,
        x=800,
        y=1000,
        theta=0.1
    )

    # Set a planned path (list of (x, y) tuples)
    planned_path = [(300, 1200), (300, 1200), (350, 350), (400, 400)]
    visualizer.set_planned_path(planned_path)

    # Set the next path index
    visualizer.set_next_path_index(1)  # Highlight the second point in the path

    # Simulate the visualization over a few frames
    for i in range(10):
        # Update the robot's observed and predicted positions slightly to simulate movement
        visualizer.set_robot_pose_observation(
            available=True,
            x=250 + i * 100,
            y=250 + i * 10,
            theta=np.pi / 4
        )
        visualizer.set_robot_pose_prediction(
            x=580 + i * 10,
            y=580 + i * 100,
            theta=np.pi / 3 + np.pi/5,
            cov_matrix=np.array([[100,0,0],[0,500,0],[0,0,4*np.pi/1]])
        )

        # Display the updated map
        visualizer.display()

        # Add a small delay to see the updates
        cv2.waitKey(1000)
