import numpy as np
import time


class LocalNavigation:
    
    def __init__(
        self,
        avoiding_period: int = 4,
        detection_distance: int = 5000,
        start_avoidance_thres: int = 2000,
        stop_avoidance_thres: int = 4000,
        max_speed: int = 50
    ) :
        self.avoiding_period = avoiding_period
        self.current_state = "running"
        self.current_state_time = 0
        self.detection_distance = detection_distance
        self.start_avoidance_thres = start_avoidance_thres
        self.stop_avoidance_thres = stop_avoidance_thres
        self.max_speed = max_speed

    def _get_obstacle_status(self, prox_horizontal):

        prox = self.detection_distance-prox_horizontal[:5]
        mn = np.min(prox)
        if mn <= self.start_avoidance_thres:
            return "near_obstacle"
        elif mn <= self.stop_avoidance_thres:
            return "far_obstacle"
        else:
            return "no_obstacle"
    
    def _transition(self, last_state, last_state_time, edge):
        
        if last_state == "running":
            if edge == "near_obstacle":
                return "avoiding", time.time()
            else:
                return "running", time.time()
        elif last_state == "avoiding":
            if edge == "no_obstacle":
                return "back_up", time.time()
            else :
                return "avoiding", last_state_time
        elif last_state == "back_up":
            if edge == "near_obstacle":
                return "avoiding", time.time()
            else :
                if time.time() - last_state_time >= self.avoiding_period:
                    return "running", time.time()
                else:
                    return "back_up", last_state_time
        else :
            raise ValueError("Invalid state")

    def update_speed(self, motor_speeds, prox_horizontal):

        edge = self._get_obstacle_status(prox_horizontal)
        self.current_state, self.current_state_time = self._transition(
            self.current_state,
            self.current_state_time,
            edge
        )
        # print(edge, self.current_state, time.time() - self.current_state_time, prox_horizontal)
        
        if self.current_state == "avoiding":

            obstacle_pos = np.argmin(np.where(prox_horizontal[:5]<=0, np.inf, self.detection_distance-prox_horizontal[:5]))

            if obstacle_pos <= 2:
                motor_speeds[0] = self.max_speed
                motor_speeds[1] = -self.max_speed
            else :
                motor_speeds[0] = -self.max_speed
                motor_speeds[1] = self.max_speed

            return [int(motor_speed) for motor_speed in motor_speeds]
    
        elif self.current_state == "back_up":

            back_up_weight = 1 - (time.time() - self.current_state_time) / self.avoiding_period
            if back_up_weight > 0.5 :
                back_up_weight = 1
            else :
                back_up_weight = 0.75

            motor_speeds[0] = self.max_speed * back_up_weight + motor_speeds[0] * (1 - back_up_weight)
            motor_speeds[1] = self.max_speed * back_up_weight + motor_speeds[1] * (1 - back_up_weight)

            return [int(motor_speed) for motor_speed in motor_speeds]

        else :

            bias = 0

            sensor_influence = {
                0: -2.0,
                1: 0.0,
                2: 0.0,
                3: 0.0,
                4: 2.0
            }

            
            for i, magnitude in enumerate(prox_horizontal[:5]):

                if magnitude != 0:
                    force_magnitude = magnitude / self.detection_distance
                else:
                    force_magnitude = 0
                bias += force_magnitude * sensor_influence[i]
            
            motor_speeds = [np.cos(bias) * motor_speeds[0] - np.sin(bias) * motor_speeds[1], 
                            np.sin(bias) * motor_speeds[0] + np.cos(bias) * motor_speeds[1]]

            motor_speeds[0] = max(min(motor_speeds[0], self.max_speed), -self.max_speed)
            motor_speeds[1] = max(min(motor_speeds[1], self.max_speed), -self.max_speed)

            return [int(motor_speed) for motor_speed in motor_speeds]
