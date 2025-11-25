from queue import PriorityQueue, Queue
from typing import Tuple
import cv2
import numpy as np

FREE = 1
OBSTACLE = 0


def extract_corners(discretized_map: np.ndarray) -> np.ndarray:
    """
    Extracts corners from a discretized map using the Shi-Tomasi corner detection method.
    Args:
        discretized_map (np.ndarray): A 2D numpy array of type uint8 representing the discretized map.

    Returns:
        np.ndarray: A 2D numpy array of shape (N, 2) containing the coordinates of the detected corners.
                    If no corners are detected, an empty array of shape (0, 2) is returned.
    """
    # extract corners using goodFeaturesToTrack (more here: https://docs.opencv.org/4.x/d4/d8c/tutorial_py_shi_tomasi.html)
    assert (discretized_map.dtype == np.uint8)
    corners = cv2.goodFeaturesToTrack(discretized_map, 50, 0.2, 10)
    if corners is None:
        return np.zeros((0, 2), dtype=int)
    else:
        return corners.astype(int).squeeze()


def extract_regions(discretized_map: np.ndarray) -> None:
    """
    Extracts and labels connected regions of obstacles in a discretized map.
    Args:
        discretized_map (np.ndarray): A 2D numpy array representing the discretized map where obstacles are marked.
    Returns:
        tuple: A tuple containing:
            - range: A range object representing the region IDs.
            - np.ndarray: A 2D numpy array of the same shape as discretized_map with each cell labeled by its region ID.
    """
    grid_height, grid_width = discretized_map.shape
    region_grid = np.zeros((grid_height, grid_width), dtype=int)

    def mark_region(i: int, j: int, region_id: int) -> None:
        """
        Marks a region in the region_grid starting from the given coordinates (i, j) with the specified region_id.
        It uses BFS (Breadth First Search, more info here: https://en.wikipedia.org/wiki/Breadth-first_search) algorithm
        to mark all connected cells in the region.
        Args:
            i (int): The starting x-coordinate in the grid.
            j (int): The starting y-coordinate in the grid.
            region_id (int): The identifier to mark the region with.
        """
        queue = Queue()
        queue.put((i, j))
        while not queue.empty():
            x, y = queue.get()
            if x < 0 or x >= grid_height or y < 0 or y >= grid_width:
                continue
            if region_grid[x, y] != 0 or discretized_map[x, y] != OBSTACLE:
                continue
            region_grid[x, y] = region_id
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    queue.put((x + dx, y + dy))

    next_region_id = 1
    for i in range(1, grid_height - 1):
        for j in range(1, grid_width - 1):
            if region_grid[i, j] == 0 and discretized_map[i, j] == OBSTACLE:
                mark_region(i, j, next_region_id)
                next_region_id += 1

    return range(1, next_region_id), region_grid


def get_distance(a: np.array, b: np.array) -> float:
    """
    Calculate the Euclidean distance between two points.
    Args:
        a (np.array): The first point as a NumPy array.
        b (np.array): The second point as a NumPy array.

    Returns:
        float: The Euclidean distance between points a and b.
    """
    return np.linalg.norm(a - b)


def update_target_index(current_position: np.ndarray, global_path: np.ndarray, current_target_index: int, threshold_distance_to_target_cm: float) -> int:
    """
    Updates the target index in the global path based on the current position and a threshold distance.
    Args:
        current_position (np.ndarray): The current position of the object.
        global_path (np.ndarray): The global path as an array of points.
        current_target_index (int): The current index of the target position in the global path.
        threshold_distance_to_target_cm (float): The distance threshold to determine if the target has been reached.
    Returns:
        int: The updated target index. Returns None if the path is finished.
    """
    distance_to_target = get_distance(
        current_position, global_path[current_target_index])

    if distance_to_target < threshold_distance_to_target_cm:
        if current_target_index == len(global_path) - 1:
            return None  # path is finished

        return current_target_index + 1

    return current_target_index


class Graph:
    def __init__(self, discretized_map: np.ndarray, corners: np.ndarray, thymio_size_in_pixels: int):
        """
        Initializes the GlobalNavigation class with the given parameters.

        Args:
            discretized_map (np.ndarray): The discretized map of the environment.
            corners (np.ndarray): An array of corner points [(y, x)].
            thymio_size_in_pixels (int): The size of the Thymio robot in pixels.
        """
        self.discretized_map = discretized_map
        self.thymio_size_in_pixels = thymio_size_in_pixels
        self.points = corners
        self.graph = self._create_graph(corners)

    def _reachable(self, a: np.ndarray, b: np.ndarray) -> bool:
        """
        Determines if point `b` is reachable from point `a` without encountering obstacles
        by a straight line.
        Args:
            a (np.ndarray): The starting point.
            b (np.ndarray): The destination point.
        Returns:
            bool: True if point `b` is reachable from point `a`, False otherwise.
        """
        def obsactle_around(point: np.ndarray, d: int) -> bool:
            """
            Check if there is an obstacle around the point within the distance `d`.
            Args:
                point (np.ndarray): The point to check.
                d (int): The distance around the point to check.
            Returns:
                bool: True if there is an obstacle around the point, False otherwise.
            """
            x, y = point
            l = max(y-d, 0)
            r = min(y+d, self.discretized_map.shape[0])
            u = max(x-d, 0)
            d = min(x+d, self.discretized_map.shape[1])
            return (self.discretized_map[l:r, u:d] == OBSTACLE).any()

        distance = np.linalg.norm(a - b)
        line = np.linspace(a, b, int(distance), dtype=int)
        d = int(self.thymio_size_in_pixels // 2)
        for point in line:
            if obsactle_around(point, d):
                return False

        return True

    def _create_graph(self, corners: np.ndarray) -> dict:
        """
        Create an undirected graph from an array of corners.
        Args:
            corners (np.ndarray): An array of corner coordinates.
        Returns:
            dict: A dictionary representing the graph.
        """
        graph = {}
        for i in range(len(corners)):
            graph[i] = []
        for i, corner in enumerate(corners):
            for j, other_corner in enumerate(corners):
                if i < j and self._reachable(corner, other_corner):
                    graph[i].append(j)
                    graph[j].append(i)

        return graph

    def get_neighbours(self, point_index: int) -> list:
        """
        Retrieve the neighboring points of a given point in the graph.
        Args:
            point_index (int): The index of the point for which to find the neighbors.
        Returns:
            list: A list of neighboring points' indices.
        """
        return self.graph[point_index]

    def add_point(self, point: np.ndarray) -> Tuple[bool, int]:
        """
        Adds a point to the graph if it is not an obstacle and connects it to all reachable points.
        Args:
            point np.ndarray: The (x, y) coordinates of the point to be added.
        Returns:
            Tuple[bool, int]: A tuple containing a boolean indicating if the point was successfully added,
                              and the index of the point in the graph. If the point is an obstacle or no
                              reachable points are found, returns (False, -1).
        """
        # check if the point is not an obstacle
        x, y = point
        if self.discretized_map[y, x] == OBSTACLE:
            return False, -1

        # find all reachable points
        reachable_points = []
        for i, other_point in enumerate(self.points):
            if self._reachable(point, other_point):
                reachable_points.append(i)

        if not reachable_points:
            return False, -1

        point_index = len(self.points)

        # add the point to the graph
        self.points = np.concatenate((self.points, [point]))
        self.graph[point_index] = reachable_points

        # update the neighbours of the reachable points
        for i in reachable_points:
            self.graph[i].append(point_index)

        return True, point_index

    def remove_point(self, point_index) -> bool:
        """
        Removes a point from the graph and updates the neihbours
        Args:
            point_index (int): The index of the point to be removed.
        Returns:
            bool: True if the point was successfully removed, False if the point_index is out of range.
        """
        if point_index >= len(self.points):
            return False

        for neighbour in self.graph[point_index]:
            self.graph[neighbour].remove(point_index)

        del self.graph[point_index]
        self.points = np.delete(self.points, point_index, axis=0)

        return True

    def get_edges(self) -> np.ndarray:
        """
        Get the edges of the graph.
        Returns:
            np.array: A list of edges, where each edge is a tuple of two points.
        """
        edges = []
        for i, neighbours in self.graph.items():
            for j in neighbours:
                if i < j:
                    edges.append([self.points[i], self.points[j]])
        return np.array(edges)


class GlobalNavigation:
    def __init__(
        self,
        map_image: np.ndarray,
        discretize_scale: int,
        discretize_threshold: int,
        thymio_size_in_pixels: int,
        small_obstacle_area_in_pixels: int = 10
    ):
        """
        Initializes the GlobalNavigation class with the given parameters.
        Args:
            map_image (np.ndarray): The map image as a numpy array.
            discretize_scale (int): The scale to discretize the map.
            discretize_threshold (int): The threshold value to discretize the map.
            thymio_size_in_pixels (int): The size of the Thymio robot in pixels.
            small_obstacle_area_in_pixels (int): The area threshold in pixels to remove small obstacles.
        """
        self.map_image = map_image
        self.discretize_scale = discretize_scale

        descritized_thymio_size = int(
            thymio_size_in_pixels // discretize_scale)

        # discretize the map
        self.discretized_map = self._discretize_map(
            map_image, discretize_scale, discretize_threshold)
        self.cleaned_discretized_map = self._remove_small_objects(
            self.discretized_map, small_obstacle_area_in_pixels // (discretize_scale ** 2))

        # expand the map
        # used to increase the expansion size to leave small space for thymio
        expansion_factor = 1.5
        # expansion size is half of the thymio size multiplied by the expansion factor
        expansion_size = int(expansion_factor * descritized_thymio_size // 2)
        self.expanded_map = self._expand_obstacles(
            self.cleaned_discretized_map, expansion_size)

        # extract corners
        self.discretized_corners = extract_corners(self.expanded_map)
        self.corners = self.discretized_corners * discretize_scale

        # create the graph
        self.graph = Graph(self.cleaned_discretized_map,
                           self.discretized_corners, descritized_thymio_size)

    def _discretize_map(self, map_image: np.ndarray, discretize_scale: int, threshold: int) -> np.ndarray:
        """
        Discretizes a given map image into a grid based on the specified scale and threshold.

        Args:
            map_image (np.ndarray): The input map image as a NumPy array.
            discretize_scale (int): The scale factor to sample the map image.
            threshold (int): The threshold value to determine obstacles and free spaces.
        Returns:
            np.ndarray: A 2D grid where each cell is marked as either an obstacle or free space.
        """
        height, width, _ = map_image.shape
        grid_height = height // discretize_scale
        grid_width = width // discretize_scale

        grid = np.zeros((grid_height, grid_width), dtype=np.uint8)

        for i in range(grid_height):
            for j in range(grid_width):
                grid[i, j] = np.mean(
                    map_image[i*discretize_scale:(i+1)*discretize_scale, j*discretize_scale:(j+1)*discretize_scale])
                if grid[i, j] < threshold:
                    grid[i, j] = OBSTACLE
                else:
                    grid[i, j] = FREE

        return grid

    def _remove_small_objects(self, grid: np.ndarray, min_size: int) -> np.ndarray:
        """
        Remove small objects from a grid based on a minimum size threshold.
        A size is calculated as the number of pixels in the object.
        Args:
            grid (np.ndarray): The discretized map grid.
            min_size (int): The minimum size threshold for objects to be retained.
        Returns:
            np.ndarray: A new grid with small objects removed.
        """
        labels, region_grid = extract_regions(grid)
        filtered_grid = np.copy(grid)

        for region_id in labels:
            if np.sum(region_grid == region_id) < min_size:
                filtered_grid[region_grid == region_id] = FREE

        return filtered_grid

    def _expand_obstacles(self, discretized_map: np.ndarray, d: int) -> np.ndarray:
        """
        Expands obstacles in a discretized map by a given distance.
        Args:
            discretized_map (np.ndarray): A numpy array representing the discretized map.
            d (int): The radius by which to expand the obstacles.
        Returns:
            np.ndarray: A numpy array with the expanded obstacles.
        """
        expanded_map = np.ones(discretized_map.shape, dtype=np.uint8) * FREE

        def draw_circle(x, y, r):
            for i in range(x - r, x + r + 1):
                for j in range(y - r, y + r + 1):
                    if 0 <= i < discretized_map.shape[0] and 0 <= j < discretized_map.shape[1]:
                        expanded_map[i, j] = OBSTACLE

        for i in range(discretized_map.shape[0]):
            for j in range(discretized_map.shape[1]):
                if discretized_map[i, j] == 0:
                    draw_circle(i, j, d)

        return expanded_map

    def _discretize_point(self, point: np.ndarray) -> np.ndarray:
        """
        Discretizes a given point to discrete map coord space.
        Args:
            point (np.ndarray): The point to be discretized.
        Returns:
            np.ndarray: The discretized point with integer coordinates.
        """
        return (np.array(point) // self.discretize_scale).astype(int)

    def _normalize_point(self, discrete_point: np.ndarray) -> np.ndarray:
        """
        Normalize a discrete point by scaling it with the discretize_scale attribute. Opposite of _discretize_point.
        Args:
            discrete_point (np.ndarray): The point to be normalized.
        Returns:
            np..ndarray: The normalized point as a numpy array.
        """
        return np.array(discrete_point) * self.discretize_scale

    def get_path(self, start, end):
        """
        Find the shortest path between start and end points

        Args:
            start (tuple): the start point (y, x)
            end (tuple): the end point (y, x)
        """
        discrete_start = self._discretize_point(start)
        discrete_end = self._discretize_point(end)

        start_point_ok, start_index = self.graph.add_point(discrete_start)
        if not start_point_ok:
            print("Start point is an obstacle")

            # find closes point to the start point
            distances = np.linalg.norm(
                self.graph.points - discrete_start, axis=1)
            start_index = np.argmin(distances)

        end_point_ok, end_index = self.graph.add_point(discrete_end)
        if not end_point_ok:
            print("End point is an obstacle")
            return None
        
        # performing Dijkstra's algorithm, more info here: https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm

        distances = np.full(len(self.graph.points), np.inf)
        distances[start_index] = 0    # start point

        visited = np.zeros(len(self.graph.points), dtype=bool)

        predecessor = np.full(len(self.graph.points), -1, dtype=int)
        predecessor[start_index] = start_index

        queue = PriorityQueue()
        queue.put((0, start_index))  # (distance, point)

        while not queue.empty():
            distance, current_index = queue.get()
            if visited[current_index]:
                continue

            visited[current_index] = True

            for neighbour_index in self.graph.get_neighbours(current_index):
                if visited[neighbour_index]:
                    continue

                moving_cost = get_distance(
                    self.graph.points[current_index], self.graph.points[neighbour_index])
                new_distance = distance + moving_cost

                if new_distance < distances[neighbour_index]:
                    predecessor[neighbour_index] = current_index
                    distances[neighbour_index] = new_distance
                    queue.put((new_distance, neighbour_index))

        def backtrack(predecessor: int, start: np.ndarray, end: np.ndarray) -> np.ndarray:
            """
            Backtrack the path from end to start using the predecessor array.
            Args:
                predecessor (np.ndarray): The predecessor array.
                start (np.ndarray): The start point.
                end (np.ndarray): The end point.
            Returns:
                np.ndarray: The path as an array of points [(y, x)].
            """
            path = []
            index = end
            while index != start:
                path.append(index)
                index = predecessor[index]

            path.append(start)
            return path

        def clean_graph():
            if end_point_ok:
                self.graph.remove_point(end_index)
            if start_point_ok:
                self.graph.remove_point(start_index)

        if predecessor[end_index] == -1:
            print("No path found between start and end")
            clean_graph()
            return None

        full_path = backtrack(predecessor, start_index, end_index)

        path_normalized = np.array(
            [self._normalize_point(self.graph.points[i]) for i in full_path])

        clean_graph()

        return path_normalized[::-1, ::-1]  # [(y,x)]
