"""

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""
import os
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

show_animation = True


class ThetaStarPlanner:

    def __init__(
        self, ob, resolution, rr, 
        min_x=None, min_y=None, max_x=None, max_y=None,
        save_animation_to_figs=False,
        fig_dir=None,
        los_mode="column_sample",
        exempt_water_endpoints=False,
    ):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        los_mode: "column_sample" (default, historical) samples one cell per
            column with int() truncation, which lets a segment cut across
            blocked cells -- measured on the AIS maps, 13 path segments crossing
            34-424 m of land each. "supercover" checks every cell the segment
            touches. OPEN_PROBLEMS item 21.
        """
        if los_mode not in ("column_sample", "supercover"):
            raise ValueError(f"unknown los_mode {los_mode!r}")
        self.los_mode = los_mode
        # A buffer wider than a cell can swallow a start or goal that sits in
        # water close to the bank (a berth), and the search then fails. When
        # set, an endpoint node is unblocked if no obstacle point lies within
        # half a cell of it -- i.e. it is blocked only by buffer, not by land.
        self.exempt_water_endpoints = exempt_water_endpoints

        self.ob = np.array(ob, dtype=float)
        assert self.ob.ndim == 2 and self.ob.shape[1] == 2, "obstacles should be a 2D array with shape (N, 2)"
        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = min_x, min_y
        self.max_x, self.max_y = max_x, max_y
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map()

        self.save_animation_to_figs = save_animation_to_figs
        self.fig_dir = fig_dir
        self.i_fig = 0
        self.plt_elements = []

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy, curr_i_fig=None):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        if self.exempt_water_endpoints:
            tree = cKDTree(self.ob)
            for n in (start_node, goal_node):
                if 0 <= n.x < self.x_width and 0 <= n.y < self.y_width and self.obstacle_map[n.x][n.y]:
                    centre = (self.calc_grid_position(n.x, self.min_x), self.calc_grid_position(n.y, self.min_y))
                    if tree.query(centre)[0] > 0.5 * self.resolution:
                        self.obstacle_map[n.x][n.y] = False

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        if self.save_animation_to_figs:
            if curr_i_fig is None:
                self.i_fig = curr_i_fig

        while True:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o]))
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                self.plt_elements.append(
                    plt.plot(
                        self.calc_grid_position(current.x, self.min_x),
                        self.calc_grid_position(current.y, self.min_y), ".y"
                    )[0]
                )
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None]
                )
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

                    if self.save_animation_to_figs:
                        plt.savefig(os.path.join(self.fig_dir, 'frame_{}.png'.format(self.i_fig)), bbox_inches='tight')
                        self.i_fig += 1


            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 np.inf, -1)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node


                parent_id = current.parent_index if current.parent_index != -1 else c_id
                parent = closed_set[parent_id]
                # if there is no line of sight, work as normal A*
                if not self.line_of_sight(parent, node): 
                    curr_path_cost = current.cost + self.motion[i][2]
                    if curr_path_cost < open_set[n_id].cost:
                        open_set[n_id].cost = curr_path_cost
                        open_set[n_id].parent_index = c_id
                # if there is line of sight, use the parent of the current node as the parent of the new node
                else:
                    curr_path_cost = parent.cost + math.hypot(parent.x - node.x, parent.y - node.y)
                    if curr_path_cost < open_set[n_id].cost:
                        open_set[n_id].cost = curr_path_cost
                        open_set[n_id].parent_index = parent_id

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry
    
    def line_of_sight(self, n1, n2):
        """
        Line of sight check
        """
        if self.los_mode == "supercover":
            return self._line_of_sight_supercover(n1, n2)
        x1, y1 = n1.x, n1.y
        x2, y2 = n2.x, n2.y
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        # f = 0

        if dy < dx:
            if x1 > x2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            slope = (y2 - y1) / (x2 - x1)
            for x in range(x1, x2):
                y = int(slope * (x - x1) + y1)
                if self.obstacle_map[x][y]:
                    return False
        else:
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            slope = (x2 - x1) / (y2 - y1)
            for y in range(y1, y2):
                x = int(slope * (y - y1) + x1)
                if self.obstacle_map[x][y]:
                    return False

        return True

    def _line_of_sight_supercover(self, n1, n2):
        """True iff no blocked cell is touched by the segment n1 -> n2.

        Nodes are cell centres at integer indices; cell (i, j) is the unit
        square centred on it. Walk the major axis one cell at a time; within
        each slab [i - 0.5, i + 0.5] the segment spans an interval on the minor
        axis, and every cell overlapping that interval is checked. Touching a
        cell's edge or corner counts, so the test is conservative.
        """
        x1, y1, x2, y2 = n1.x, n1.y, n2.x, n2.y
        steep = abs(y2 - y1) > abs(x2 - x1)
        if steep:
            x1, y1, x2, y2 = y1, x1, y2, x2
        if x1 > x2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0.0
        omap = self.obstacle_map
        for x in range(x1, x2 + 1):
            xa, xb = max(x - 0.5, x1), min(x + 0.5, x2)
            ya, yb = y1 + slope * (xa - x1), y1 + slope * (xb - x1)
            lo, hi = (ya, yb) if ya <= yb else (yb, ya)
            for y in range(math.ceil(lo - 0.5), math.floor(hi + 0.5) + 1):
                cx, cy = (y, x) if steep else (x, y)
                if 0 <= cx < self.x_width and 0 <= cy < self.y_width and omap[cx][cy]:
                    return False
        return True

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self):

        ox, oy = self.ob[:, 0], self.ob[:, 1]
        if self.min_x is None:
            self.min_x = round(min(ox))
        if self.min_y is None:
            self.min_y = round(min(oy))
        if self.max_x is None:
            self.max_x = round(max(ox))
        if self.max_y is None:
            self.max_y = round(max(oy))
        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)

        # Obstacle map generation: a cell is blocked when any obstacle lies
        # within self.rr of its centre. Written as a nearest-neighbour query
        # rather than the equivalent triple loop over
        # (x_width x y_width x len(ob)) -- on a harbour-sized map in metres
        # that product reaches 1e9+ and the loop never finishes.
        xs = self.calc_grid_position(np.arange(self.x_width), self.min_x)
        ys = self.calc_grid_position(np.arange(self.y_width), self.min_y)
        grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
        nearest, _ = cKDTree(np.asarray(self.ob, dtype=float)).query(grid)
        self.obstacle_map = (nearest <= self.rr).reshape(self.x_width, self.y_width)
        # for xi in range(self.x_width):
        #     for yi in range(self.y_width):
        #         ob_status = np.any(
        #             np.linalg.norm(self.ob - np.array([xi, yi]), axis=1) < 1
        #         )
        #         if ob_status:
        #             self.obstacle_map[xi][yi] = True

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion

