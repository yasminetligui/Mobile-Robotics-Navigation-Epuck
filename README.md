# Team Project for Mobile Robots 2024 Fall

## API Lists

### CV

- class Camera() :
  - get_frame() -> array : Get a frame from camera directly, and return the frame as an array.
  - get_frame_when_esc() -> array : Display the video until esc is touched, and return the frame as an array.
- display_image(frame) : Display an image.
- generate_aruco(id, size) : Generate an aruco marker with specific index and pixel number.
- get_aruco_pose(frame) -> frame, corners, ids : Detect the aruco markers in the frame. corners are 4 corners of each marker, ids are the index of each marker, respectively.
- capture_map(path, map_id) : Capture a map and save to path with name map{map_id}.
- transform_map(frame, target_scale) -> frame : Use perspective transformation to transform a map to standard shape. target_scale is the expected sum of height and width after transformation.
- descretize_map(frame, grid_size) -> grid : Discritize a map to 0/1.
- get_robot_pose(image) -> success, xy, theta : Get robot aruco pose from image.

## Extended Kalman Filter

- class KalmanFilter(t_frame, d_pixel, d_robot, d_aruco, v_error) :
  - t_frame is the shutter time.
  - d_pixel is the minimum resolution of the camera (in your unit).
  - d_robot is the wheel distance of the robot (in your unit).
  - d_aruco is the size of aruco markers (in your unit).
  - v_error is the relative error of wheel speed measurement (in %).
  - base_kalman_filter(...) -> [X, P] : The function to run the kalman filter given all parameters.
  - get_prediction(...) -> [x, y, theta] : The function to predict current robot state, given last state and current measurement.
