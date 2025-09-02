import numpy as np
from shapely.geometry import Polygon, Point
from scipy.spatial import ConvexHull

def build_horizontal_polygon(vertices_3d):
    """
    Robustly construct a 2D polygon representing the horizontal footprint of an OBB.
    This is done by projecting all 8 vertices onto the x-y plane and taking their convex hull.
    """
    # Project onto the horizontal plane
    points_2d = np.array([[v[0], v[1]] for v in vertices_3d])  # shape (8, 2)

    # Compute convex hull
    hull = ConvexHull(points_2d)
    polygon_points = points_2d[hull.vertices]

    return Polygon(polygon_points)

def merge_collision_segments(collision_frames):
    """
    Merge multiple consecutive or near-consecutive collision frames into continuous segments.
    Segments are merged if the gap between frames does not exceed 1 frame.
    Returns a list of (start_frame, end_frame) tuples.
    """
    if not collision_frames:
        return []

    collision_frames = sorted(collision_frames)
    segments = []
    start = collision_frames[0]
    prev = start

    for frame in collision_frames[1:]:
        if frame - prev > 1:
            segments.append((start, prev))
            start = frame
        prev = frame

    segments.append((start, prev))
    return segments


def smooth_path(path, window_size=3):
    """
    Smooth a path using simple moving average while keeping the start and end points unchanged.
    
    Parameters:
    - path: List of [x, y] points
    - window_size: Size of the sliding window (must be odd, recommended 3 or 5)

    Returns:
    - Smoothed path (start and end points remain unchanged)
    """
    if len(path) < window_size or window_size < 3:
        return path

    smoothed = [path[0]]  # Keep the starting point
    for i in range(1, len(path) - 1):
        start = max(0, i - window_size // 2)
        end = min(len(path), i + window_size // 2 + 1)
        window = path[start:end]
        avg_x = sum(p[0] for p in window) / len(window)
        avg_y = sum(p[1] for p in window) / len(window)
        smoothed.append((avg_x, avg_y))
    smoothed.append(path[-1])  # Keep the ending point
    return smoothed

def check_collision_2d(waypoints, list_of_obb_vertices):
    """
    Check whether each waypoint intersects with any OBB in the horizontal plane.
    Returns a boolean list of the same length as waypoints.
    """
    polygons = [build_horizontal_polygon(verts) for verts in list_of_obb_vertices]
    collision_flags = []

    for pt in waypoints:
        point_xy = Point(pt[0], pt[1])
        collision = any(polygon.contains(point_xy) for polygon in polygons)
        collision_flags.append(collision)

    return collision_flags

import matplotlib.pyplot as plt

def visualize_gridmap(grid_map, start=None, goal=None, path=None, labeled_objects=None, save_path='test.png'):
    """
    Visualize the GridMap. 
    labeled_objects should be a list of (name, position) tuples.
    """
    occupancy = grid_map.occupancy.T  # Transpose for a more intuitive coordinate system

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(occupancy, cmap='Greys', origin='lower')

    ax.set_xlim(0, grid_map.width)
    ax.set_ylim(0, grid_map.height)

    # Start point
    if start is not None:
        gx, gy = grid_map.world_to_grid(*start)
        ax.plot(gx, gy, 'go', label='Start')

    # Goal point
    if goal is not None:
        gx, gy = grid_map.world_to_grid(*goal)
        ax.plot(gx, gy, 'ro', label='Goal')

    # Path
    if path is not None:
        path_grid = [grid_map.world_to_grid(x, y) for (x, y) in path]
        x_list, y_list = zip(*path_grid)
        ax.plot(x_list, y_list, 'b-', label='Path')

    # Display obstacle labels
    if labeled_objects:
        for name, pos in labeled_objects:
            gx, gy = grid_map.world_to_grid(*pos[:2])
            ax.text(
                gx, gy, name, color='blue', fontsize=8, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.5, boxstyle='round')
            )

    ax.set_title("GridMap Occupancy")
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    ax.set_aspect("equal")
    ax.legend()
    plt.grid(True)
    plt.savefig(save_path)


class GridMap:
    def __init__(self, xlim, ylim, resolution):
        self.resolution = resolution
        self.xlim = xlim
        self.ylim = ylim

        self.width = int((xlim[1] - xlim[0]) / resolution)
        self.height = int((ylim[1] - ylim[0]) / resolution)
        self.occupancy = np.zeros((self.width, self.height), dtype=bool)

    def world_to_grid(self, x, y):
        gx = int((x - self.xlim[0]) / self.resolution)
        gy = int((y - self.ylim[0]) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        x = gx * self.resolution + self.xlim[0] + self.resolution / 2
        y = gy * self.resolution + self.ylim[0] + self.resolution / 2
        return x, y

    def set_obstacle_from_polygon(self, polygon):
        for gx in range(self.width):
            for gy in range(self.height):
                wx, wy = self.grid_to_world(gx, gy)
                if polygon.contains(Point(wx, wy)):
                    self.occupancy[gx, gy] = True

    def is_free(self, gx, gy):
        if 0 <= gx < self.width and 0 <= gy < self.height:
            return not self.occupancy[gx, gy]
        return False
    
    def is_free_with_shape(self, gx, gy, obj_w, obj_h):
        """
        Check whether an object of size obj_w × obj_h (meters), centered at (gx, gy),
        would collide with any obstacles.
        """
        # Convert dimensions from meters to grid cells
        w_cells = int(np.ceil(obj_w / self.resolution))
        h_cells = int(np.ceil(obj_h / self.resolution))

        # Check symmetric range (ensuring center alignment)
        for dx in range(-w_cells // 2, (w_cells + 1) // 2):
            for dy in range(-h_cells // 2, (h_cells + 1) // 2):
                if not self.is_free(gx + dx, gy + dy):
                    return False
        return True


import heapq

def astar(grid_map, start_xy, goal_xy, start_obj_w=None, start_obj_h=None, 
          path_obj_w=None, path_obj_h=None, goal_obj_w=None, goal_obj_h=None, 
          search_radius=100):
    """
    A* path planning algorithm supporting different object sizes at different stages.
    
    Parameters:
    - grid_map: The grid map
    - start_xy: Start point in world coordinates
    - goal_xy: Goal point in world coordinates
    - start_obj_w, start_obj_h: Object size at start (used to validate start point)
    - path_obj_w, path_obj_h: Object size during path planning (usually smaller if object is lifted)
    - goal_obj_w, goal_obj_h: Object size at goal (used to validate goal point)
    - search_radius: Radius to search for alternative points if start/goal is blocked
    """
    start_gx, start_gy = grid_map.world_to_grid(*start_xy)
    goal_gx, goal_gy = grid_map.world_to_grid(*goal_xy)

    # Check if start is collision-free (using start object size)
    start_is_free = False
    if start_obj_w is not None and start_obj_h is not None:
        start_is_free = grid_map.is_free_with_shape(start_gx, start_gy, start_obj_w, start_obj_h)
    else:
        start_is_free = grid_map.is_free(start_gx, start_gy)
    
    # If start is blocked, find the nearest free point
    while not start_is_free:
        alt_start = find_closest_free_point(grid_map, start_gx, start_gy, 
                                          start_obj_w, start_obj_h, search_radius)
        if alt_start is None:
            start_obj_w = max(0, start_obj_w - 0.05)
            start_obj_h = max(0, start_obj_h - 0.05)
            path_obj_w = max(0, path_obj_w - 0.05) if path_obj_w is not None else None
            path_obj_h = max(0, path_obj_h - 0.05) if path_obj_h is not None else None
            continue  # Try with smaller dimensions
        start_is_free = True
        # Update start grid coordinates
        start_gx, start_gy = alt_start
        start_xy = grid_map.grid_to_world(start_gx, start_gy)

    # Check if goal is collision-free (using goal object size)
    goal_is_free = False
    if goal_obj_w is not None and goal_obj_h is not None:
        goal_is_free = grid_map.is_free_with_shape(goal_gx, goal_gy, goal_obj_w, goal_obj_h)
    else:
        goal_is_free = grid_map.is_free(goal_gx, goal_gy)
    
    # Save original goal for later distance calculations
    original_goal_gx, original_goal_gy = goal_gx, goal_gy
    
    if not goal_is_free:
        alt_goal = find_closest_free_point(grid_map, goal_gx, goal_gy, 
                                          goal_obj_w, goal_obj_h, search_radius)
        if alt_goal is not None:
            goal_gx, goal_gy = alt_goal
            goal_is_free = True
        # If no alternative goal is found, keep original goal but mark as blocked

    open_set = []
    heapq.heappush(open_set, (0, (start_gx, start_gy)))

    came_from = {}
    cost_so_far = {(start_gx, start_gy): 0}

    goal_np = np.array([goal_gx, goal_gy])
    directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(1,-1),(-1,1)]

    while open_set:
        _, current = heapq.heappop(open_set)
        
        # If goal is reached
        if current == (goal_gx, goal_gy):
            if goal_is_free:
                break
            else:
                # Goal is blocked; continue searching
                continue

        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            
            # Check free space using path object size (usually smaller)
            is_free = False
            if path_obj_w is not None and path_obj_h is not None:
                is_free = grid_map.is_free_with_shape(nx, ny, path_obj_w, path_obj_h)
            else:
                is_free = grid_map.is_free(nx, ny)
                
            if not is_free:
                continue

            # For diagonal moves, ensure adjacent cells are also free 
            # (prevents cutting through obstacle corners)
            if dx != 0 and dy != 0:
                adj1_free = False
                adj2_free = False
                
                if path_obj_w is not None and path_obj_h is not None:
                    adj1_free = grid_map.is_free_with_shape(current[0] + dx, current[1], path_obj_w, path_obj_h)
                    adj2_free = grid_map.is_free_with_shape(current[0], current[1] + dy, path_obj_w, path_obj_h)
                else:
                    adj1_free = grid_map.is_free(current[0] + dx, current[1])
                    adj2_free = grid_map.is_free(current[0], current[1] + dy)
                
                if not adj1_free or not adj2_free:
                    continue

            new_cost = cost_so_far[current] + np.hypot(dx, dy)
            if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                cost_so_far[(nx, ny)] = new_cost
                priority = new_cost + np.linalg.norm(np.array([nx, ny]) - goal_np)
                heapq.heappush(open_set, (priority, (nx, ny)))
                came_from[(nx, ny)] = current

    # Check if a path to the goal exists
    if (goal_gx, goal_gy) not in came_from and (goal_gx, goal_gy) != (start_gx, start_gy):
        # No path found to goal, pick closest reachable point
        if len(came_from) == 0:
            return None  # No reachable points at all
        
        reachable_points = list(came_from.keys()) + [(start_gx, start_gy)]
        
        # Filter out points that are valid for the goal object size
        valid_endpoints = []
        for point in reachable_points:
            px, py = point
            is_valid = False
            if goal_obj_w is not None and goal_obj_h is not None:
                is_valid = grid_map.is_free_with_shape(px, py, goal_obj_w, goal_obj_h)
            else:
                is_valid = grid_map.is_free(px, py)
            
            if is_valid:
                valid_endpoints.append(point)
        
        # If valid endpoints exist, pick the one closest to the original goal
        if valid_endpoints:
            original_goal_np = np.array([original_goal_gx, original_goal_gy])
            closest_point = min(valid_endpoints, key=lambda p: np.linalg.norm(np.array(p) - original_goal_np))
            goal_gx, goal_gy = closest_point
        else:
            # If no valid endpoint is found, return None
            return None
    
    # Reconstruct path
    path = []
    curr = (goal_gx, goal_gy)
    
    # Handle case where start == goal
    if curr == (start_gx, start_gy):
        return [start_xy]
    
    # Ensure current point exists in came_from (unless it's the start)
    if curr not in came_from:
        return None
    
    while curr != (start_gx, start_gy):
        path.append(grid_map.grid_to_world(*curr))
        if curr not in came_from:
            # Safety check to prevent infinite loop
            return None
        curr = came_from[curr]
    
    path.append(start_xy)
    path.reverse()
    return path


def find_closest_free_point(grid_map, gx, gy, obj_w=None, obj_h=None, search_radius=10):
    """Find the nearest collision-free grid point to (gx, gy)."""
    visited = set()
    queue = [(0, gx, gy)]  # (distance, x, y)
    
    directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(1,-1),(-1,1)]
    
    while queue:
        dist, x, y = heapq.heappop(queue)
        
        if (x, y) in visited:
            continue
        visited.add((x, y))
        
        # Check if current point is collision-free
        is_free = False
        if obj_w is not None and obj_h is not None:
            is_free = grid_map.is_free_with_shape(x, y, obj_w, obj_h)
        else:
            is_free = grid_map.is_free(x, y)
            
        if is_free:
            return (x, y)
        
        # Stop searching if we exceed the radius
        if dist >= search_radius:
            continue
            
        # Add neighboring points to the queue
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            new_dist = dist + np.hypot(dx, dy)
            heapq.heappush(queue, (new_dist, nx, ny))
    
    return None  # No valid point found within search radius


from shapely.geometry import LineString

def simplify_path_rdp(path, epsilon=0.1):
    """
    Simplify a path using the Ramer–Douglas–Peucker (RDP) algorithm,
    while forcing the start and end points to remain unchanged.
    
    Args:
    - path: List of [x, y]
    - epsilon: Simplification tolerance
    
    Returns:
    - Simplified path with start and end points preserved.
    """
    if len(path) <= 2:
        return path

    line = LineString(path)
    simplified = list(line.simplify(epsilon, preserve_topology=False).coords)

    # Force start and end points to match original path (avoid precision drift)
    if not np.allclose(simplified[0], path[0]):
        simplified[0] = tuple(path[0])
    if not np.allclose(simplified[-1], path[-1]):
        simplified[-1] = tuple(path[-1])

    return simplified


def sample_path_by_distance(path, min_dist=0.2):
    """
    Resample the path so that consecutive points are at least min_dist apart.
    Always keeps the start and end points, and removes unnecessary points near them.
    """
    if not path:
        return []

    start = np.array(path[0])
    end = np.array(path[-1])
    sampled = [path[0]]
    last_pt = start

    for pt in path[1:-1]:  # Resample intermediate points
        pt = np.array(pt)
        if np.linalg.norm(pt - last_pt) >= min_dist:
            sampled.append(pt.tolist())
            last_pt = pt

    # === Handle excessive points near the goal ===
    # Remove trailing points too close to the end, keeping only the last valid one
    i = len(sampled) - 1
    while i >= 0:
        dist = np.linalg.norm(np.array(sampled[i]) - end)
        if dist < min_dist / 2:
            i -= 1
        else:
            break
    sampled = sampled[:i + 1]
    sampled.append(path[-1])

    # === Handle excessive points near the start ===
    # Remove points too close to the start, keep only the first valid one
    i = 1
    while i < len(sampled):
        dist = np.linalg.norm(np.array(sampled[i]) - start)
        if dist < min_dist / 2:
            del sampled[i]
        else:
            break

    return sampled


def assign_frame_ids_by_distance(path_2d, start_frame, end_frame):
    """
    Assign frame indices to path points proportionally to their traveled distance.
    
    Args:
    - path_2d: List of [x, y]
    - start_frame: First frame index
    - end_frame: Last frame index
    
    Returns:
    - List of frame IDs corresponding to each point in path_2d
    """
    if not path_2d:
        return []

    # 1. Compute cumulative distances
    dists = [0.0]
    for i in range(1, len(path_2d)):
        dist = np.linalg.norm(np.array(path_2d[i]) - np.array(path_2d[i-1]))
        dists.append(dists[-1] + dist)
    total_dist = dists[-1] if dists[-1] > 0 else 1.0  # Avoid division by zero

    # 2. Map distances to frame IDs
    frame_span = end_frame - start_frame
    frame_ids = [int(round(start_frame + frame_span * (d / total_dist))) for d in dists]

    return frame_ids

