# Comments are following a 3x3 example, not a 4x4 which the video showcases.
# Each tile has a number from 1 to 8 which indicates in which position the tile should be in the sorted image.
# state - an array where state[0] tells which of the 9 tiles is currently in the top-left corner,
#                        state[1] which tile is currently in the top-middle...


import heapq
import random
import time
import config
from collections import deque

module_heuristic = __import__('heuristics')


class Algorithm:
    def __init__(self, heuristic=None):
        self.heuristic = heuristic
        self.nodes_evaluated = 0
        self.nodes_generated = 0

    # Returns a list of possible moves, i.e., linearized matrix indices around the empty tile.
    def get_legal_actions(self, state):
        self.nodes_evaluated += 1
        # max_index     - number of tiles in the matrix (e.g. if the matrix is 3x3, max_index is 9).
        # config.N      - number of rows in the matrix.
        # zero_tile_ind - index of the tile in the matrix where the empty tile currently is.
        max_index = len(state)
        zero_tile_ind = state.index(0)
        legal_actions = []
        # If the empty tile isn't in the top row, it's possible to slide down the tile that is above it.
        if 0 <= zero_tile_ind - config.N:
            legal_actions.append(zero_tile_ind - config.N)
        # If the empty tile isn't in the bottom row, it's possible to slide up the tile that is below it.
        if (zero_tile_ind + config.N) < max_index:
            legal_actions.append(zero_tile_ind + config.N)
        # If the empty tile isn't in the far right column, it's possible to slide left the tile that is next to it.
        if (zero_tile_ind + 1) % config.N:
            legal_actions.append(zero_tile_ind + 1)
        # If the empty tile isn't in the far left column, it's possible to slide right the tile that is next to it.
        if zero_tile_ind % config.N:
            legal_actions.append(zero_tile_ind - 1)
        return legal_actions

    # Returns a copy of the state array where the positions of the empty tile and its chosen neighbour are switched.
    def apply_action(self, state, action):
        self.nodes_generated += 1
        copy_state = list(state)
        zero_tile_ind = state.index(0)
        copy_state[action], copy_state[zero_tile_ind] = copy_state[zero_tile_ind], copy_state[action]
        return tuple(copy_state)

    # This function is overriden in the extended classes.
    def get_steps(self, initial_state, goal_state):
        pass

    # This function is called from the main program before even starting the game.
    def get_solution_steps(self, initial_state, goal_state):
        begin_time = time.time()
        solution_actions = self.get_steps(initial_state, goal_state)
        print(f'Execution time in seconds: {(time.time() - begin_time):.2f} | '
              f'Nodes generated: {self.nodes_generated} | '
              f'Nodes evaluated: {self.nodes_evaluated}')
        return solution_actions


# Picks randomly one of the possible moves (switches the positions of the empty tile and one of its neighbours).
class ExampleAlgorithm(Algorithm):
    def get_steps(self, initial_state, goal_state):
        state = initial_state
        solution_actions = []
        while state != goal_state:
            legal_actions = self.get_legal_actions(state)
            action = legal_actions[random.randint(0, len(legal_actions) - 1)]
            solution_actions.append(action)
            state = self.apply_action(state, action)
        return solution_actions


class Node:
    def __init__(self, action, state):
        self.action = action
        self.state = state
        self.parent = None


class BFSAlgorithm(Algorithm):
    def get_steps(self, initial_state, goal_state):
        root = Node(-1, initial_state)
        node = root
        queue = deque([root])
        visited_set = set()

        while node.state != goal_state:
            node = queue.popleft()
            state = node.state

            if state in visited_set:
                continue
            visited_set.add(state)

            for action in self.get_legal_actions(state):
                new_state = self.apply_action(state, action)
                new_node = Node(action, new_state)
                new_node.parent = node
                queue.append(new_node)

        solution_actions = []
        while node != root:
            solution_actions.append(node.action)
            node = node.parent
        solution_actions.reverse()
        return solution_actions


class NodeBF(Node):
    def __init__(self, action, state):
        super().__init__(action, state)
        self.predecessors = set()
        self.heuristic = None

    # Overrides the lt operator for Node objects so that they can be compatible with the heap data structure.
    def __lt__(self, other):
        return self.heuristic < other.heuristic or \
            self.heuristic == other.heuristic and self.state < other.state


class BestFirstAlgorithm(Algorithm):
    def get_steps(self, initial_state, goal_state):
        root = NodeBF(-1, initial_state)
        node = root
        lst = [root]
        heapq.heapify(lst)

        while node.state != goal_state:
            node = heapq.heappop(lst)
            state = node.state

            for action in self.get_legal_actions(state):
                new_state = self.apply_action(state, action)
                if new_state in node.predecessors:
                    continue

                new_node = NodeBF(action, new_state)
                new_node.heuristic = self.heuristic.get_evaluation(new_state)
                new_node.parent = node
                new_node.predecessors = node.predecessors
                new_node.predecessors.add(node.state)
                heapq.heappush(lst, new_node)

        solution_actions = []
        while node != root:
            solution_actions.append(node.action)
            node = node.parent
        solution_actions.reverse()
        return solution_actions


class NodeAStar(Node):
    def __init__(self, action, state):
        super().__init__(action, state)
        self.predecessors = set()
        self.heuristic = None
        self.cumulated_cost = 0

    # Overrides the lt operator for Node objects so that they can be compatible with the heap data structure.
    def __lt__(self, other):
        return self.cumulated_cost < other.cumulated_cost or \
            self.cumulated_cost == other.cumulated_cost and self.state < other.state


class AStarAlgorithm(Algorithm):
    def get_steps(self, initial_state, goal_state):
        root = NodeAStar(-1, initial_state)
        node = root
        node.heuristic = 0
        lst = [root]
        heapq.heapify(lst)
        cost = 1

        while node.state != goal_state:
            node = heapq.heappop(lst)
            state = node.state

            for action in self.get_legal_actions(state):
                new_state = self.apply_action(state, action)
                if new_state in node.predecessors:
                    continue

                new_node = NodeAStar(action, new_state)
                new_node.heuristic = self.heuristic.get_evaluation(new_state)
                new_node.cumulated_cost = node.cumulated_cost - node.heuristic + cost + new_node.heuristic
                new_node.parent = node
                new_node.predecessors = node.predecessors
                new_node.predecessors.add(node.state)
                heapq.heappush(lst, new_node)

        solution_actions = []
        while node != root:
            solution_actions.append(node.action)
            node = node.parent
        solution_actions.reverse()
        return solution_actions
