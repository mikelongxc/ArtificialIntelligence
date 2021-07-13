# Name:         Michael Long
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Tile Driver I
# Term:         Summer 2021

import queue
from typing import List, Tuple


class Heuristic:

    @staticmethod
    def get(tiles: Tuple[int, ...]) -> int:
        """
        Return the estimated distance to the goal using Manhattan distance
        and linear conflicts.

        Only this static method should be called during a search; all other
        methods in this class should be considered private.

        >>> Heuristic.get((0, 1, 2, 3))
        0
        >>> Heuristic.get((3, 2, 1, 0))
        6
        """
        width = int(len(tiles) ** 0.5)
        return (Heuristic._get_manhattan_distance(tiles, width)
                + Heuristic._get_linear_conflicts(tiles, width))

    @staticmethod
    def _get_manhattan_distance(tiles: Tuple[int, ...], width: int) -> int:
        """
        Return the Manhattan distance of the given tiles, which represents
        how many moves is tile is away from its goal position.
        """
        distance = 0
        for i in range(len(tiles)):
            if tiles[i] != 0:
                row_dist = abs(i // width - tiles[i] // width)
                col_dist = abs(i % width - tiles[i] % width)
                distance += row_dist + col_dist
        return distance

    @staticmethod
    def _get_linear_conflicts(tiles: Tuple[int, ...], width: int) -> int:
        """
        Return the number of linear conflicts in the tiles, which represents
        the minimum number of tiles in each row and column that must leave and
        re-enter that row or column in order for the puzzle to be solved.
        """
        conflicts = 0
        rows = [[] for i in range(width)]
        cols = [[] for i in range(width)]
        for i in range(len(tiles)):
            if tiles[i] != 0:
                if i // width == tiles[i] // width:
                    rows[i // width].append(tiles[i])
                if i % width == tiles[i] % width:
                    cols[i % width].append(tiles[i])
        for i in range(width):
            conflicts += Heuristic._count_conflicts(rows[i])
            conflicts += Heuristic._count_conflicts(cols[i])
        return conflicts * 2

    @staticmethod
    def _count_conflicts(ints: List[int]) -> int:
        """
        Return the minimum number of tiles that must be removed from the given
        list in order for the list to be sorted.
        """
        if Heuristic._is_sorted(ints):
            return 0
        lowest = None
        for i in range(len(ints)):
            conflicts = Heuristic._count_conflicts(ints[:i] + ints[i + 1:])
            if lowest is None or conflicts < lowest:
                lowest = conflicts
        return 1 + lowest

    @staticmethod
    def _is_sorted(ints: List[int]) -> bool:
        """Return True if the given list is sorted and False otherwise."""
        for i in range(len(ints) - 1):
            if ints[i] > ints[i + 1]:
                return False
        return True


class State:

    def __init__(self, tiles: Tuple[int, ...], path: str, g: int):
        self.tiles = tiles

        self.g = g
        self.h = Heuristic.get(tiles)
        self.f = self.g + self.h

        # represents current path i.e. "HJKL"
        self.path = path

    # for priority queue
    def __lt__(self, other):
        return self.f < other.f

    def __repr__(self):
        return ", ".join([str(x) for x in self.tiles])

    def is_goal_state(self):
        return self.h == 0


def solve_puzzle(tiles: Tuple[int, ...]) -> str:
    """
    Return a string (containing characters "H", "J", "K", "L") representing the
    optimal number of moves to solve the given puzzle.
    """

    q: queue.PriorityQueue = queue.PriorityQueue()

    # get current state and check if is already at goal
    cur_state = State(tiles, "", 0)
    if cur_state.is_goal_state():
        return cur_state.path

    q.put(cur_state)

    # A* algorithm
    while 1:
        cur_state = q.get()

        found_path = configure_frontier(cur_state, q)
        if found_path != "":
            return found_path


def configure_frontier(state: State, q: queue.PriorityQueue) -> str:
    """
    Helper function for solve_puzzle. Returns either an empty string
    or the optimal path if it was found
    """
    empty_index = state.tiles.index(0)

    # deduce which moves are allowed
    allowed_moves = configure_moves(state, empty_index)

    # for each allowed move, add new state to frontier states
    for move in allowed_moves:
        next_state = create_new_state(state, move, empty_index)

        if next_state.is_goal_state():
            return next_state.path
        q.put(next_state)

    return ""


def configure_moves(state: State, empty_index: int) -> str:
    """
    Helper function for find_frontier_states(). Returns a string of which
    moves are allowed given the location of the empty tile
    """
    allowed_moves = ""
    opposite_moves = {"H": "L", "J": "K", "K": "J", "L": "H"}
    width = int(len(state.tiles) ** 0.5)

    # take away "left" move if empty tile on right edge of puzzle
    if empty_index % width != width - 1:
        allowed_moves += "H"
    # take away "down" move if empty tile on the top edge of puzzle
    if empty_index >= width:
        allowed_moves += "J"
    # take away "up" move if empty tile on bottom edge of puzzle
    if empty_index < len(state.tiles) - width:
        allowed_moves += "K"
    # take away "right" move if empty tile on left edge of puzzle
    if empty_index % width != 0:
        allowed_moves += "L"

    # take away opposite move from previous move
    if len(state.path) != 0:
        opposite = opposite_moves.get(state.path[len(state.path) - 1])
        allowed_moves = allowed_moves.replace(opposite, "")

    return allowed_moves


def create_new_state(state: State, move: str, empty_index: int) -> State:
    """
    Helper function for find_frontier_states(). Returns a newly configured
    State object based on a singular move made.
    """
    next_tiles = []
    width = int(len(state.tiles) ** 0.5)

    # copy tiles into mutable array
    for i in range(len(state.tiles)):
        next_tiles.append(state.tiles[i])

    # swap tiles with appropriate move made
    swap_tiles(move, next_tiles, empty_index, width)

    # increment g because config is just 1 singular move away
    g = state.g + 1

    # update path string with new move
    path = state.path + move

    return State(tuple(next_tiles), path, g)


def swap_tiles(move: str, next_tiles: List, empty_index: int, width: int)\
        -> None:
    """
    Helper function for create_new_state(). Swaps any two tiles
    i.e. (3, 2, 0, 1) to (3, 2, 1, 0)
    """
    if move == "H":
        next_tiles[empty_index], next_tiles[empty_index + 1] \
            = next_tiles[empty_index + 1], next_tiles[empty_index]
    elif move == "J":
        next_tiles[empty_index], next_tiles[empty_index - width] \
            = next_tiles[empty_index - width], next_tiles[empty_index]
    elif move == "K":
        next_tiles[empty_index], next_tiles[empty_index + width] \
            = next_tiles[empty_index + width], next_tiles[empty_index]
    elif move == "L":
        next_tiles[empty_index], next_tiles[empty_index - 1] \
            = next_tiles[empty_index - 1], next_tiles[empty_index]


def main() -> None:
    # tiles = (0, 1, 2, 3)
    #tiles = (6, 7, 8, 3, 0, 5, 1, 2, 4)
    # tiles = (7, 0, 8, 6, 3, 5, 1, 2, 4)
    tiles = (0, 3, 6, 5, 4, 7, 2, 1, 8)
    print(solve_puzzle(tiles))


if __name__ == "__main__":
    main()
