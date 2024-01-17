import config


class Heuristic:
    def get_evaluation(self, state):
        pass


class ExampleHeuristic(Heuristic):
    def get_evaluation(self, state):
        return 0


# Returns the number of tiles which are not in their goal position.
class HammingHeuristic(Heuristic):
    def get_evaluation(self, state):
        order = 0
        heuristic = 0
        for tile in state:
            if tile != 0 and (tile-1) != order:
                heuristic += 1
            order += 1
        return heuristic


# Measures for each tile (except the empty tile) how far it is from its goal position.
# Returns the sum of all 8 measurements.
class ManhattanHeuristic(Heuristic):
    def get_evaluation(self, state):
        order = 0
        heuristic = 0
        for tile in state:
            if tile != 0 and (tile-1) != order:
                row_mov = abs((tile-1) // config.N - order // config.N)
                col_mov = abs((tile-1) % config.N - order % config.N)
                heuristic += row_mov + col_mov
            order += 1
        return heuristic
