"""
State is a 42b value:
5 11  17  23  29  35  41
4 10  16  22  28  34  40
3  9  15  21  27  33  39
2  8  14  20  26  32  38
1  7  13  19  25  31  37
0  6  12  18  24  30  36

For instance:
STATE            STATE_binary     RED state        YEL state
_ _ _ _ _ _ _    0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0
_ _ _ _ _ _ _    0 0 0 0 0 0 0    0 0 0 0 0 0 0    0 0 0 0 0 0 0
_ _ _ Y _ _ _    0 0 0 1 0 0 0    0 0 0 0 0 0 0    0 0 0 1 0 0 0
_ _ _ R _ _ _    0 0 0 1 0 0 0    0 0 0 1 0 0 0    0 0 0 0 0 0 0
_ _ _ Y _ R _    0 0 0 1 0 1 0    0 0 0 0 0 1 0    0 0 0 1 0 0 0
R Y _ R _ Y _    1 1 0 1 0 1 0    1 0 0 1 0 0 0    0 1 0 0 0 1 0
"""
import random
import time
from state import State
import math


max_depth = None    # How many moves ahead the AI will think (given as a command line argument).
max_player = None   # The player whose move it is will be the MAX player in the algorithms.
min_player = None   # The other player will be the MIN player.


# Agents are AI algorithms which can be picked as players through the command line arguments.
class Agent:
    ident = 0

    def __init__(self):
        self.id = Agent.ident
        Agent.ident += 1

    def get_chosen_column(self, state, depth):
        pass


# No AI used, player selects a column on his own.
class Human(Agent):
    pass


# An AI which just plays a random move.
class ExampleAgent(Agent):
    def get_chosen_column(self, state, depth):
        time.sleep(random.random())
        columns = state.get_possible_columns()
        return columns[random.randint(0, len(columns) - 1)]


# An AI which uses Minimax algorithm with alpha-beta pruning.
class MinimaxABAgent(Agent):
    def get_chosen_column(self, state, depth):
        global max_player, min_player, max_depth
        max_player = state.get_next_on_move()
        min_player = State.RED if max_player == State.YEL else State.YEL
        max_depth = depth if depth != 0 else math.inf

        root = Node(state, -1)
        minimax_ab(root, max_player, -math.inf, math.inf, 0)

        chosen_col = root.chosen_succ
        return chosen_col


# An AI which uses NegaScout algorithm.
class Negascout(Agent):
    def get_chosen_column(self, state, depth):
        global max_player, min_player, max_depth
        max_player = state.get_next_on_move()
        min_player = State.RED if max_player == State.YEL else State.YEL
        max_depth = depth if depth != 0 else math.inf

        root = Node(state, -1)
        negascout(root, max_player, -math.inf, math.inf, 0)

        chosen_col = root.chosen_succ
        return chosen_col




# A node in the game tree we are creating consists of the following values:
# - game's current state (layout of the played tokens).
# - id of the column which was selected in order to reach the current state from the previous one.
# - id of the next column which will be picked (this attribute will be set later by the node's child).
class Node:
    def __init__(self, state, col):
        self.state = state
        self.col = col
        self.chosen_succ = None


# Evaluates how good the current state is for the MAX Player by counting all possible wins and losses.
# Possible improvement for the future: don't count all possible wins and losses equally, give advantage to
#                                      victories which can be reached sooner and to postponed losses.
def node_evaluation(state):
    status = state.get_state_status()
    # MAX Player won (count_tokens used in order to give higher priority to the wins with fewer tokens used)
    if status == max_player:
        return 1000 - count_tokens(state.get_checkers(max_player))
    # MIN Player won (count_tokens used in order to give higher priority to the wins with fewer tokens used)
    elif status == min_player:
        return -1000 + count_tokens(state.get_checkers(min_player))
    elif status == State.DRAW:
        return 0

    max_player_tokens = state.get_checkers(max_player)
    min_player_tokens = state.get_checkers(min_player)
    max_wins_cnt = 0
    max_losses_cnt = 0

    # State.win_masks is a list of 42b values.
    # Each value consists of only zeroes except for the four ones which represent the four winning tokens.
    # List consists all possible winning positions in the game regardless of the current state.
    for mask in State.win_masks:
        if (mask & min_player_tokens) == 0:
            max_wins_cnt += 1   # It's still possible for the MAX Player to win this way.
        if (mask & max_player_tokens) == 0:
            max_losses_cnt += 1
    return max_wins_cnt - max_losses_cnt


# Columns closer to the middle of the table are given a bigger priority (it's an advantage to control the middle).
def col_priority(column):
    arr = [1, 3, 5, 6, 4, 2, 0]
    return arr[column]


# Counts how many tokens the selected player has already placed.
def count_tokens(checkers):
    cnt = 0
    mask = 1
    for i in range(42):
        if (checkers & mask) != 0:
            cnt += 1
        mask <<= 1
    return cnt


# Returns None if the game hasn't ended yet.
def is_terminal_node(state):
    return state.get_state_status() is not None


# Minimax algorithm with alpha-beta pruning.
def minimax_ab(node, player, alpha, beta, depth):
    if is_terminal_node(node.state) or depth == max_depth:
        return node_evaluation(node.state)

    # Node's direct children will be visited in the ascending order of their evaluations.
    # Thus, the chances of pruning are higher.
    sorted_cols = sorted(node.state.get_possible_columns(),
                         key=lambda column: (node_evaluation(node.state.generate_successor_state(column)),
                                             col_priority(column)), reverse=True)

    if player == max_player:
        score = -math.inf
        for col in sorted_cols:
            child_state = node.state.generate_successor_state(col)
            child_score = minimax_ab(Node(child_state, col), min_player, alpha, beta, depth + 1)
            if child_score > score:
                score = child_score
                alpha = score
                node.chosen_succ = col
            if alpha >= beta:
                break
        return score
    else:
        score = +math.inf
        for col in sorted_cols:
            child_state = node.state.generate_successor_state(col)
            child_score = minimax_ab(Node(child_state, col), max_player, alpha, beta, depth + 1)
            if child_score < score:
                score = child_score
                beta = score
                node.chosen_succ = col
            if alpha >= beta:
                break
        return score


# NegaScout algorithm.
def negascout(node, player, alpha, beta, depth):
    if is_terminal_node(node.state) or depth == max_depth:
        return node_evaluation(node.state) * (-1 if player == min_player else 1)
    other_player = max_player if player == min_player else min_player

    sorted_cols = sorted(node.state.get_possible_columns(),
                         key=lambda column: (node_evaluation(node.state.generate_successor_state(column)),
                                             col_priority(column)), reverse=True)

    score = -math.inf
    for col in sorted_cols:
        # Assume that the first child will lead to the best path.
        if col == sorted_cols[0]:
            child_state = node.state.generate_successor_state(col)
            child_score = -negascout(Node(child_state, col), other_player, -beta, -alpha, depth + 1)
            node.chosen_succ = col
        else:
            # For other children use the null alpha-beta window.
            child_state = node.state.generate_successor_state(col)
            child_score = -negascout(Node(child_state, col), other_player, -alpha - 1, -alpha, depth + 1)
            # If another child leads to a better path, traverse its subtree again but this time using the full window.
            if alpha < child_score < beta:
                child_score = -negascout(Node(child_state, col), other_player, -beta, -alpha, depth + 1)
                node.chosen_succ = col

        score = max(score, child_score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return score
