"""
In this file, you should implement your trajectory generation class or function.
Your method must generate a smooth 3-axis trajectory (x(t), y(t), z(t)) that 
passes through all the previously computed path points. A positional deviation 
up to 0.1 m from each path point is allowed.

You should output the generated trajectory and visualize it. The figure must
contain three subplots showing x, y, and z, respectively, with time t (in seconds)
as the horizontal axis. Additionally, you must plot the original discrete path 
points on the same figure for comparison.

You are expected to write the implementation yourself. Do NOT copy or reuse any 
existing trajectory generation code from others. Avoid using external packages 
beyond general scientific libraries such as numpy, math, or scipy. If you decide 
to use additional packages, you must clearly explain the reason in your report.
"""
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt

from flight_environment import FlightEnvironment
from path_planner import AStarPlanner3D


class TrajectoryGenerator:
    def __init__(self, planner: AStarPlanner3D):
        self.planner = planner
        self.trajectory_x = None
        self.trajectory_y = None
        self.trajectory_z = None


    def generate_trajectory(self, path):
        '''
        @input:
            path: An Nx3 array representing the path points.
        @output:
            X-axis, Y-axis, Z-axis trajectories as functions of time.
            shape: (N) where N is the number of time steps.
        '''
        path = np.array(path)
        self.trajectory_x = path[:, 0]
        self.trajectory_y = path[:, 1]
        self.trajectory_z = path[:, 2]
        return self.trajectory_x, self.trajectory_y, self.trajectory_z
    
    
        
    def plot_trajectory(self, trajectory_list: np.ndarray):
        '''
        @input:
            trajectory_list: A [3, N] array representing the trajectory x, y and z points.
        '''
        N = trajectory_list.shape[1]
        time_steps = np.arange(N)

        plt.figure()

        plt.subplot(3, 1, 1)
        plt.plot(time_steps, trajectory_list[0], label='Trajectory X')
        plt.xlabel('Time (s)')
        plt.ylabel('X Position (m)')
        plt.title('X-axis Trajectory')
        plt.grid()
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(time_steps, trajectory_list[1], label='Trajectory Y')
        plt.xlabel('Time (s)')
        plt.ylabel('Y Position (m)')
        plt.title('Y-axis Trajectory')
        plt.grid()
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(time_steps, trajectory_list[2], label='Trajectory Z')
        plt.xlabel('Time (s)')
        plt.ylabel('Z Position (m)')
        plt.title('Z-axis Trajectory')
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.show()

