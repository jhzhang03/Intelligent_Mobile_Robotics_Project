"""
In this file, you should implement your own path planning class or function.
Within your implementation, you may call `env.is_collide()` and `env.is_outside()`
to verify whether candidate path points collide with obstacles or exceed the
environment boundaries.

You are required to write the path planning algorithm by yourself. Copying or calling 
any existing path planning algorithms from others is strictly
prohibited. Please avoid using external packages beyond common Python libraries
such as `numpy`, `math`, or `scipy`. If you must use additional packages, you
must clearly explain the reason in your report.
"""
import numpy as np
import matplotlib.pyplot as plt
from flight_environment import FlightEnvironment
import heapq
import numpy as np

class AStarPlanner3D:
    def __init__(self,
                 env,
                 resolution=0.2,
                 safety_margin=0.2,
                 ):
        """
        env: FlightEnvironment
        resolution: grid resolution
        safety_margin: safety margin to the cylindar
        """
        self.env = env
        self.resolution = resolution
        self.safety_margin = safety_margin

        self.neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    self.neighbors.append((dx, dy, dz))

        # edge of env
        self.x_min, self.x_max = 0.0, self.env.env_width
        self.y_min, self.y_max = 0.0, self.env.env_length
        self.z_min, self.z_max = 0.0, self.env.env_height

        # size of grid
        self.nx = int((self.x_max - self.x_min) / self.resolution) + 1
        self.ny = int((self.y_max - self.y_min) / self.resolution) + 1
        self.nz = int((self.z_max - self.z_min) / self.resolution) + 1

    #******************* tools *******************#
    def point_to_grid(self, point):
        x, y, z = point
        ix = int(round((x - self.x_min) / self.resolution))
        iy = int(round((y - self.y_min) / self.resolution))
        iz = int(round((z - self.z_min) / self.resolution))
        return (ix, iy, iz)

    def grid_to_point(self, cell):
        ix, iy, iz = cell
        x = self.x_min + ix * self.resolution
        y = self.y_min + iy * self.resolution
        z = self.z_min + iz * self.resolution
        return (x, y, z)

    def in_bounds(self, cell):
        ix, iy, iz = cell
        return (0 <= ix < self.nx and
                0 <= iy < self.ny and
                0 <= iz < self.nz)

    def is_free(self, point):
        if self.env.is_outside(point):
            return False
        if self.env.is_collide(point, epsilon=self.safety_margin):
            return False
        return True

    #******************* main *******************#
    def heuristic(self, cell, goal_point):
        x, y, z = self.grid_to_point(cell)
        gx, gy, gz = goal_point
        return np.sqrt((x - gx)**2 + (y - gy)**2 + (z - gz)**2)

    def plan(self, start, goal):
        """
        Input:
            start: (x, y, z)
            goal : (x, y, z)
        Output:
            path: [[x0, y0, z0], [x1, y1, z1], ...] or None
        """

        # start at obstacle
        if not self.is_free(start):
            print("Start is in collision or outside environment.")
            return None
        if not self.is_free(goal):
            print("Goal is in collision or outside environment.")
            return None

        start_cell = self.point_to_grid(start)
        goal_cell  = self.point_to_grid(goal)

        # open set: (f, g, (ix, iy, iz))
        open_heap = []
        heapq.heappush(open_heap, (0.0, 0.0, start_cell))

        g_score = {start_cell: 0.0}
        came_from = {}

        closed_set = set()

        while open_heap:
            f_curr, g_curr, cell = heapq.heappop(open_heap)

            if cell in closed_set:
                continue
            if cell == goal_cell:
                return self.reconstruct_path(came_from, cell, start)

            closed_set.add(cell)

            for dx, dy, dz in self.neighbors:
                nbr = (cell[0] + dx, cell[1] + dy, cell[2] + dz)

                if not self.in_bounds(nbr):
                    continue

                # grid to point
                nbr_point = self.grid_to_point(nbr)
                if not self.is_free(nbr_point):
                    continue

                # cost
                step_cost = np.sqrt(dx*dx + dy*dy + dz*dz) * self.resolution
                tentative_g = g_curr + step_cost

                if nbr in g_score and tentative_g >= g_score[nbr]:
                    continue

                g_score[nbr] = tentative_g
                came_from[nbr] = cell

                h = self.heuristic(nbr, goal)
                f = tentative_g + h
                heapq.heappush(open_heap, (f, tentative_g, nbr))

        print("No path found.")
        return None

    def reconstruct_path(self, came_from, current, start_point):
        path_cells = [current]
        while current in came_from:
            current = came_from[current]
            path_cells.append(current)
        path_cells.reverse()

        path = [list(self.grid_to_point(c)) for c in path_cells]

        path[0] = list(start_point)
        return path











