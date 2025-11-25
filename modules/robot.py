import numpy as np
from tdmclient import ClientAsync, aw
from contextlib import contextmanager

class RobotInstruction:

    def __init__(self):
        self.client = ClientAsync()
        self.node = None
        self.verbose = False
        self.in_context_manager = False

    @contextmanager
    def lock_thymio(self, variable_to_wait=None):
        try:
            aw(self.node.lock())
            aw(self.node.wait_for_variables(variable_to_wait))
            yield None
        finally:
            aw(self.node.unlock())

    def connect(self):
        """Connect to the Thymio robot."""
        self.node = aw(self.client.wait_for_node())
        print("Robot connected.")
        return

    def disconnect(self):
        """Unlock the node and disconnect."""
        if self.node is not None:
            self.node = None
            print("Robot disconnected.")
        return
    
    def get_variables(self):
        if self.node is None:
            print("Failed to print variables. Robot not connected.")
            return
        variables = aw(self.node.var_description())
        for variable in variables:
            print(variable)
        return
    
    def enable_comments(self):
        self.verbose = True

    def disable_comments(self):
        self.verbose = False

    def get_motorspeeds(self):
        if self.node is None:
            print("Failed to get motorspeeds. Robot not connected.")
        l_speed = self.node["motor.left.speed"]
        r_speed = self.node["motor.right.speed"]
        motorspeeds = [l_speed,r_speed]
        if self.verbose:
            print(f'The current motor speeds are: {motorspeeds}' )
        return motorspeeds
    
    def set_motorspeeds(self, motor_speeds=[100,100]):
        """Move the robot by choosing left and rigth motor speeds. (default speed = 100)"""

        if self.node is None:
            print("Failed moving forward. Robot not connected.")
            return
        v = {
            "motor.left.target": [motor_speeds[0]],
            "motor.right.target": [motor_speeds[1]]
        }
        aw(self.node.set_variables(v))
        if self.verbose:
            print(f"Set motor speeds at: {motor_speeds}")
        return
    
    def stop_moving(self):
        if self.node is None:
            print("Failed to mobilise. Robot not connected.")
            return
        
        v = {
            "motor.left.target": [0],
            "motor.right.target": [0]
        }
        aw(self.node.lock())
        aw(self.node.set_variables(v))
        aw(self.node.unlock())
        if self.verbose:
            print(f"Robot stopped moving.")
        return
    
    def detect_obstacles(self):
        if self.node is None:
            print("Failed to detect obstacles. Robot not connected.")
        distances = list(self.node["prox.horizontal"])
        if self.verbose:
            print(f'The proximity sensors values are currently: {distances}' )
        return np.array(distances)


if __name__ == "__main__":
    #robot = RobotInstruction()
    #robot.connect()
    #robot.move_forward()
    #robot.set_leds()
    #aw(robot.client.sleep(1))
    #robot.stop()
    #robot.disconnect()
    pass
