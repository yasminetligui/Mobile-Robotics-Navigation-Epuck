import numpy as np
import matplotlib.pyplot as plt

def plot_map(
    map: np.ndarray, 
    robot_pos: tuple = None,
    target_pos: tuple = None,
    waypoints: np.ndarray = None,
    path: np.ndarray = None,
    edges: np.ndarray = None,
    title: str = None):
    """
    Plot the map with the robot position and orientation
    """

    plt.imshow(map, cmap='gray', interpolation='nearest', origin='lower')

    if edges is not None:
        for edge in edges:
            plt.plot([edge[0][1], edge[1][1]], [edge[0][0], edge[1][0]], c='blue', linestyle='dotted')
            
    if waypoints is not None:
        plt.scatter(waypoints[:, 1], waypoints[:, 0], c='blue', marker='o', zorder=5)

    if path is not None:
        plt.plot(path[:, 1], path[:, 0], c='orange', linewidth=3)

    if robot_pos is not None:
        plt.scatter(robot_pos[1], robot_pos[0], c='green', marker='o', s=100, zorder=10)

    if target_pos is not None:
        plt.scatter(target_pos[1], target_pos[0], c='red', marker='o', s=100, zorder=10)
        
    if title is not None:
        plt.title(title)
    
    plt.show()

class CancellationToken:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def is_cancelled(self):
        return self.cancelled
