import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt

class KalmanFilter() :

    def __init__(
        self,
        t_frame: float,
        d_pixel: float,
        d_robot: float,
        d_aruco: float,
        v_error: float,
    ) :
        """
        t_frame: Shutter time,
        d_pixel: Resolution constant,
        d_robot: Robot wheel distance,
        d_aruco: Aruco marker size (describes the theta mesurement error)
        v_error: Velocity measurement error rate (in %)
        Notice: d_pixel, d_robot, d_aruco should be in the same unit (e.g. cm)
        """
        self.t_frame = t_frame
        self.d_pixel = d_pixel
        self.d_robot = d_robot
        self.d_aruco = d_aruco
        self.v_error = v_error
        self.v_error_abs = 0.01
        self.xy_error_abs = 0.05
        self.theta_error_abs = 0.0001
        self.init_P = np.diag([1e10, 1e10, 1e10])
        self.init_x = np.array([[0.0], [0.0], [0.0]])
        self.last_P = self.init_P

    def base_kalman_filter(
        self,
        F_k: np.ndarray,
        B_k: np.ndarray,
        H_k: np.ndarray,
        Q_k: np.ndarray,
        R_k: np.ndarray,
        x_k_1: np.ndarray,
        P_k_1: np.ndarray,
        u_k: np.ndarray,
        z_k: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Base Kalman Filter
        F_k: State Transition Matrix
        B_k: Control Input Matrix
        H_k: Observation Matrix
        Q_k: Process Noise Covariance Matrix
        R_k: Measurement Noise Covariance Matrix
        x_k_1: Initial State Estimate
        P_k_1: Initial Error Covariance Estimate
        u_k: Control Input
        z_k: Measurement
        Returns: x_k, P_k
        """
        x_k_pred = F_k @ x_k_1 + B_k @ u_k
        P_k_pred = F_k @ P_k_1 @ F_k.T + Q_k
        y_k = z_k - H_k @ x_k_pred
        y_k[2] = (y_k[2] % (2 * np.pi) + 3 * np.pi) % (2 * np.pi) - np.pi
        S_k = H_k @ P_k_pred @ H_k.T + R_k
        K_k = P_k_pred @ H_k.T @ np.linalg.inv(S_k)
        x_k = x_k_pred + K_k @ y_k
        x_k[2] = (x_k[2] % (2 * np.pi) + 3 * np.pi) % (2 * np.pi) - np.pi
        P_k = (np.eye(K_k.shape[0]) - K_k @ H_k) @ P_k_pred

        return x_k, P_k

    def get_prediction(
        self,
        last_x: float,
        last_y: float,
        last_theta: float,
        measure_x: float,
        measure_y: float,
        measure_theta: float,
        command_vl: float,
        command_vr: float,
        dt: float,
        aruco_available: bool
    ) :
        """
        last_x: last filtered x position of the robot
        last_y: last filtered y position of the robot
        last_theta: last filtered orientation of the robot (in rads, from x-axis)
        measure_x: new measured x position of the robot
        measure_y: new measured y position of the robot
        command_vl: left wheel velocity
        command_vr: right wheel velocity
        dt: time interval since last measurement
        aruco_available: indicates if the x,y measurement are trustworthy
        """

        # State Transition Matrix

        v_ = (command_vl + command_vr) / 2
        delta_v = command_vr - command_vl
        sigma_vl = np.abs(command_vl) * self.v_error + self.v_error_abs
        sigma_vr = np.abs(command_vr) * self.v_error + self.v_error_abs
        cos_theta = np.cos(last_theta)
        sin_theta = np.sin(last_theta)

        F_k = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        # Control Input Matrix
        B_k = np.array([
            [0.5 * dt * sin_theta, 0.5 * dt * sin_theta],
            [0.5 * dt * cos_theta, 0.5 * dt * cos_theta],
            [dt / self.d_robot, -dt / self.d_robot]
        ])


        # Observation Matrix
        H_k = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        sigma_xx = sin_theta**2 * dt**2 * (sigma_vl**2 + sigma_vr**2) / 4
        sigma_yy = cos_theta**2 * dt**2 * (sigma_vl**2 + sigma_vr**2) / 4
        sigma_xy = sin_theta * cos_theta * dt**2 * (sigma_vl**2 + sigma_vr**2) / 4
        sigma_xt = sin_theta * dt**2 * (sigma_vl**2 - sigma_vr**2) / (2*self.d_robot)
        sigma_yt = cos_theta * dt**2 * (sigma_vl**2 - sigma_vr**2) / (2*self.d_robot)
        sigma_tt = dt**2 * (sigma_vl**2 + sigma_vr**2) / (self.d_robot**2)

        # Process Noise Covariance Matrix
        Q_k = np.array([
            [sigma_xx, sigma_xy, sigma_xt],
            [sigma_xy, sigma_yy, sigma_yt],
            [sigma_xt, sigma_yt, sigma_tt]
        ])
        Q_k = Q_k + np.diag([self.xy_error_abs, self.xy_error_abs, self.theta_error_abs]) * dt**2

        # Measurement Noise Covariance Matrix
        if aruco_available :
            R_k = np.array([
                [v_**2 * self.t_frame**2 + self.d_pixel**2, 0, 0],
                [0, v_**2 * self.t_frame**2 + self.d_pixel**2, 0],
                [0, 0, ((delta_v**2 + v_**2)*self.t_frame**2 + self.d_pixel**2) / self.d_aruco**2]
            ])
        else :
            R_k = np.array([
                [np.inf, 0, 0],
                [0, np.inf, 0],
                [0, 0, np.inf]
            ])
        
        maybe_hijack = False
        if aruco_available and (np.abs(measure_x-last_x) > 10 or np.abs(measure_y-last_y) > 10) :
            self.last_P = self.init_P
            maybe_hijack = True
        
        x_k_1 = np.array([[last_x], [last_y], [last_theta]])
        u_k = np.array([[command_vl], [command_vr]])
        z_k = np.array([[measure_x], [measure_y], [measure_theta]])

        x_k, P_k = self.base_kalman_filter(
            F_k,
            B_k,
            H_k,
            Q_k,
            R_k,
            x_k_1,
            self.last_P,
            u_k,
            z_k
        )

        self.last_P = P_k

        return x_k[0, 0], x_k[1, 0], x_k[2, 0], maybe_hijack

def generate_testcase(d_robot: float, dt: float = 0.1) -> Tuple[list, list, list, list, list, list, list, list] :
    """
    Generates a test case for a mobile robot using a simple motion model with noise.
    Args:
        d_robot (float): The distance between the wheels of the robot.
        dt (float, optional): The time step for each iteration. Defaults to 0.1.
    Returns:
        Tuple[list, list, list, list, list, list, list, list]: 
            - x_real_list (list): The list of true x positions of the robot.
            - y_real_list (list): The list of true y positions of the robot.
            - theta_real_list (list): The list of true orientations of the robot.
            - x_measure_list (list): The list of measured x positions of the robot.
            - y_measure_list (list): The list of measured y positions of the robot.
            - theta_measure_list (list): The list of measured orientations of the robot.
            - vl_list (list): The list of left wheel velocities.
            - vr_list (list): The list of right wheel velocities.
    """


    vl_list = [0]*10 + [1, 1, 1, 1, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.5, 1.5, 1.5, 1.5]
    vr_list = [0]*10 + [1, 1, 1, 1, 1, 0.9, 0.8, 0.7, 0.6, 1.5, 1.5, 1.5, 1.5, 1.5]
    x_real_list = []
    y_real_list = []
    theta_real_list = []
    x_real = 0
    y_real = 0
    theta_real = 0
    x_measure_list = []
    y_measure_list = []
    theta_measure_list = []

    for vl, vr in zip(vl_list, vr_list) :
        
        x_real = x_real + (vl + vr) / 2 * np.cos(theta_real) * dt + np.random.normal(0, 0.02)
        y_real = y_real + (vl + vr) / 2 * np.sin(theta_real) * dt + np.random.normal(0, 0.02)
        theta_real = theta_real + (vr - vl) / d_robot * dt + np.random.normal(0, 0.02)
    
        x_measure_list.append(x_real + np.random.normal(0, 0.02))
        y_measure_list.append(y_real + np.random.normal(0, 0.02))
        theta_measure_list.append(theta_real + np.random.normal(0, 0.02))

        x_real_list.append(x_real)
        y_real_list.append(y_real)
        theta_real_list.append(theta_real)
    
    return x_real_list, y_real_list, theta_real_list, x_measure_list, y_measure_list, theta_measure_list, vl_list, vr_list

if __name__ == "__main__" :

    d_robot = 0.1
    dt = 0.1

    (
        x_real_list,
        y_real_list,
        theta_real_list,
        x_measure_list,
        y_measure_list,
        theta_measure_list,
        vl_list,
        vr_list,
    ) = generate_testcase(d_robot, dt)

    kf = KalmanFilter(
        t_frame=0.01,
        d_pixel=0.01,
        d_robot=d_robot,
        d_aruco=0.05,
        v_error=0.05
    )
    x_pred = -1
    y_pred = -1
    x_pred_list = []
    y_pred_list = []
    theta_pred_list = []

    for x_measure, y_measure, theta_measure, vl, vr in zip(
        x_measure_list,
        y_measure_list,
        theta_measure_list,
        vl_list,
        vr_list
    ) :
        x_pred, y_pred, theta_pred, hijack = kf.get_prediction(
            x_pred,
            y_pred,
            theta_measure,
            x_measure,
            y_measure,
            theta_measure,
            vl,
            vr,
            dt,
            True
        )
        x_pred_list.append(x_pred)
        y_pred_list.append(y_pred)
        theta_pred_list.append(theta_pred)
        # print(f"x_pred: {x_pred}, y_pred: {y_pred}, theta_pred: {theta_pred}")

    plt.plot(x_real_list, y_real_list, label="Real Path")
    plt.plot(x_measure_list, y_measure_list, label="Measured Path")
    plt.plot(x_pred_list, y_pred_list, label="Predicted Path")
    plt.legend()
    plt.show()
