import math
import time

import numpy as np


class PIDController:
    def __init__(self, kp_linear, ki_linear, kd_linear,
                 kp_angular, ki_angular, kd_angular,
                 max_speed=50, angular_speed_correction=0.5):
        """
        Initialise les paramètres du contrôleur PID.
        """
        self.kp_linear = kp_linear
        self.ki_linear = ki_linear
        self.kd_linear = kd_linear
        self.kp_angular = kp_angular
        self.ki_angular = ki_angular
        self.kd_angular = kd_angular
        self.max_speed = max_speed

        self.last_time = time.time()
        self.last_distance_to_target = 0
        self.last_alpha_error = 0
        self.integral_linear = 0
        self.integral_angular = 0
        self.angular_speed_correction = angular_speed_correction

    def calculate_wheel_speeds(self,
                               current_position_cm=[0, 0],
                               target_position_cm=[0, 0],
                               current_angle=0,):

        # Computing the dt
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time  # Mise à jour du timestamp

        # Computing erreurs
        dx = target_position_cm[0] - current_position_cm[0]
        dy = target_position_cm[1] - current_position_cm[1]
        distance_to_target = (dx**2 + dy**2)**0.5  # Distance linéaire
        distance_to_target = np.clip(distance_to_target, 0, 10)
        alpha = math.atan2(dx, dy)  # Angle vers la cible

        alpha_error = ((alpha - current_angle) % (2 * math.pi) +
                       # Erreur angulaire
                       math.pi * 3) % (2 * math.pi) - math.pi
        # print("Alpha error:", alpha_error)
        alpha_error = np.clip(alpha_error, -1, 1)

        # Integrales for the pid
        self.integral_linear += distance_to_target * dt
        self.integral_angular += alpha_error * dt

        # Derivates for the pid
        derivative_linear = (distance_to_target -
                             self.last_distance_to_target) / dt
        derivative_angular = (alpha_error - self.last_alpha_error) / dt

        # Updating the errors for next utilisation of the function
        self.last_distance_to_target = distance_to_target
        self.last_alpha_error = alpha_error

        # Compute linear and angular speed
        linear_speed = (
            self.kp_linear * distance_to_target +
            self.ki_linear * self.integral_linear +
            self.kd_linear * derivative_linear
        )

        angular_speed = (
            self.kp_angular * alpha_error +
            self.ki_angular * self.integral_angular +
            self.kd_angular * derivative_angular
        )

        # Compute speeds of each wheel
        if np.abs(alpha_error) > 0.4:
            ratio = 1
        else:
            ratio = 0.2 + 0.8 * np.abs(alpha_error) / 0.4

        # print(ratio, linear_speed, angular_speed)

        left_speed = (1-ratio) * linear_speed + ratio * angular_speed
        right_speed = (1-ratio) * linear_speed - ratio * angular_speed

        # Limiting speeds
        clipping_ratio = max(max(abs(left_speed), abs(right_speed) + 1) / self.max_speed, 1)
        left_speed = left_speed / clipping_ratio
        right_speed = right_speed / clipping_ratio

        return int(left_speed), int(right_speed)
