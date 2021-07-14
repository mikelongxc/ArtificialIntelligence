# Name:         Michael Long
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Tile Driver II
# Term:         Summer 2021

import random
import queue
from typing import Callable, List, Tuple

import tiledriver


def is_solvable(tiles: Tuple[int, ...]) -> bool:
    """
    Return True if the given tiles represent a solvable puzzle and False
    otherwise.

    >>> is_solvable((3, 2, 1, 0))
    True
    >>> is_solvable((0, 2, 1, 3))
    False
    """
    _, inversions = _count_inversions(list(tiles))
    width = int(len(tiles) ** 0.5)
    if width % 2 == 0:
        row = tiles.index(0) // width
        return (row % 2 == 0 and inversions % 2 == 0 or
                row % 2 == 1 and inversions % 2 == 1)
    else:
        return inversions % 2 == 0


def _count_inversions(ints: List[int]) -> Tuple[List[int], int]:
    """
    Count the number of inversions in the given sequence of integers (ignoring
    zero), and return the sorted sequence along with the inversion count.

    This function is only intended to assist |is_solvable|.

    >>> _count_inversions([3, 7, 1, 4, 0, 2, 6, 8, 5])
    ([1, 2, 3, 4, 5, 6, 7, 8], 10)
    """
    if len(ints) <= 1:
        return ([], 0) if 0 in ints else (ints, 0)
    midpoint = len(ints) // 2
    l_side, l_inv = _count_inversions(ints[:midpoint])
    r_side, r_inv = _count_inversions(ints[midpoint:])
    inversions = l_inv + r_inv
    i = j = 0
    sorted_tiles = []
    while i < len(l_side) and j < len(r_side):
        if l_side[i] <= r_side[j]:
            sorted_tiles.append(l_side[i])
            i += 1
        else:
            sorted_tiles.append(r_side[j])
            inversions += len(l_side[i:])
            j += 1
    sorted_tiles += l_side[i:] + r_side[j:]
    return (sorted_tiles, inversions)

class State:

    def __init__(self, tiles: Tuple[int, ...], prev_move: str):
        self.tiles = tiles
        self.prev_move = prev_move
        self.lc = tiledriver.Heuristic._get_linear_conflicts\
            (tiles, int(len(tiles) ** 0.5))

        self.plateau_count = 0

    def __eq__(self, other):
        for i in range(len(self.tiles)):
            if self.tiles[i] != other.tiles[i]:
                return False
        return True

    # for priority queue
    def __lt__(self, other):
        return self.lc < other.lc

    def increment_plateau_count(self):
        self.plateau_count += 1





def generate_random(width: int) -> Tuple[int, ...]:
    """
    Generates random solvable list (NOT TUPLE)

    """

    random_tiles = []
    for i in range(width ** 2):
        random_tiles.append(i)

    random.shuffle(random_tiles)

    while not is_solvable(tuple(random_tiles)):
        random.shuffle(random_tiles)

    print("generating random")
    return tuple(random_tiles)


def conflict_tiles(width: int, min_lc: int) -> Tuple[int, ...]:
    """
    Create a solvable shuffled puzzle of the given width with a minimum number
    of linear conflicts (ignoring Manhattan distance).

    >>> tiles = conflict_tiles(3, 5)
    >>> tiledriver.Heuristic._get_linear_conflicts(tiles, 3)
    5
    """

    # k = width ** width
    # k = 4 * width ** (width/2)
    k = 4 * min_lc ** min_lc
    found = False
    # k_list = [None] * k
    k_list = []
    # successor_list = [(k + (k * 4))]
    successor_queue: queue.PriorityQueue = queue.PriorityQueue()

    # generate initial random k_states
    for i in range(k):
        # gen random tiles, put tiles in state obj.
        tiles = generate_random(width)
        k_state = State(tiles, "")

        # if random and we get lucky, return
        if k_state.lc >= min_lc:
            return tiles

        # add k_state to k_list
        #k_list[i] = k_state
        k_list.append(k_state)

    while not found:
        # for each k_state, add successors to a 'total successor' list
        for k_state in k_list:
            possible_answ = generate_successors(\
                k_state, successor_queue, width, min_lc)

            if possible_answ != []:
                return tuple(possible_answ)


        # for each successor, choose k-best TODO: randomization k-pick
        for i in range(k):
            k_list[i] = successor_queue.get()

        successor_queue.queue.clear()

    return tuple()


def generate_successors(k_state: State, successor_queue: queue.PriorityQueue, \
                        width: int, min_lc: int) -> List[int]:
    """
    From a k-state (max k), add successor states to successor_list
    from |generate_random()| based on whatever move can be made
    """
    opposite_moves = {"H": "L", "J": "K", "K": "J", "L": "H"}

    # find the allowed moves from k_state
    empty_index = k_state.tiles.index(0)

    allowed_moves = _get_allowed_moves(\
        k_state.tiles, width, empty_index, k_state.prev_move)

    # for each move in allowed moves, create new state
    for move in allowed_moves:
        next_frontier = create_frontier_state(\
            k_state.tiles, move, empty_index, width)

        # check to make sure next_frontier doesnt get added if prev_move
        """if next_frontier.prev_move != "" \
                and next_frontier.prev_move == opposite_moves.get(move):
            continue"""

        print("K: " + str(k_state.lc) + ", F: " + str(next_frontier.lc))
        print("index: " + str(next_frontier.tiles.index(0)))
        # plateau
        if k_state.lc < next_frontier.lc:
            successor_queue.put(next_frontier)
        elif k_state.lc == next_frontier.lc:
            if k_state.plateau_count > width - 1:
                successor_queue.put(State(generate_random(width), ""))
            else:
                successor_queue.put(next_frontier)

            k_state.increment_plateau_count()

        else:
            successor_queue.put(State(generate_random(width), ""))


        # check if we get lucky and generated successor is ideal
        if next_frontier.lc >= min_lc:
            return list(next_frontier.tiles)

    return []


def create_frontier_state(tiles: Tuple[int, ...], move: str,\
                     empty_index: int, width: int) -> State:
    """
    Helper function for find_frontier_states(). Returns a newly configured
    State object based on a singular move made.
    """
    next_tiles = []

    # copy tiles into mutable array
    for i in range(len(tiles)):
        next_tiles.append(tiles[i])

    # swap tiles with appropriate move made
    swap_tiles(move, next_tiles, empty_index, width)

    return State(tuple(next_tiles), move)


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


def is_equal(t1: Tuple[int, ...], t2: Tuple[int, ...]) -> bool:
    for i in range(len(t1)):
        if t1[i] != t2[i]:
            return False
    return True


def _get_allowed_moves(tiles: Tuple[int, ...], \
                       width: int, empty_index: int, prev_move: str) -> str:
    allowed_moves = ""
    opposite_moves = {"H": "L", "J": "K", "K": "J", "L": "H"}

    # take away "left" move if empty tile on right edge of puzzle
    if empty_index % width != width - 1:
        allowed_moves += "H"
    # take away "down" move if empty tile on the top edge of puzzle
    if empty_index >= width:
        allowed_moves += "J"
    # take away "up" move if empty tile on bottom edge of puzzle
    if empty_index < len(tiles) - width:
        allowed_moves += "K"
    # take away "right" move if empty tile on left edge of puzzle
    if empty_index % width != 0:
        allowed_moves += "L"

    if prev_move != "":
        opposite = opposite_moves.get(prev_move)
        allowed_moves.replace(opposite, "")

    return allowed_moves


def _create_new_state(tiles: Tuple[int, ...], move: str, \
                      empty_index: int, width: int) -> Tuple[int, ...]:
    """
    Helper function for find_frontier_states(). Returns a newly configured
    State object based on a singular move made.
    """
    next_tiles = []

    # copy tiles into mutable array
    for i in range(len(tiles)):
        next_tiles.append(tiles[i])

    # swap tiles with appropriate move made
    _swap_tiles(move, next_tiles, empty_index, width)

    return tuple(next_tiles)


def _swap_tiles(move: str, next_tiles: List, empty_index: int, width: int)\
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


def shuffle_tiles(width: int, min_len: int,
                  solve_puzzle: Callable[[Tuple[int, ...]], str]
) -> Tuple[int, ...]:
    """
    Create a solvable shuffled puzzle of the given width with an optimal
    solution length equal to or greater than the given minimum length.

    >>> tiles = shuffle_tiles(3, 6, tiledriver.solve_puzzle)
    >>> len(tiledriver.solve_puzzle(tiles))
    6
    """


def main() -> None:
    x = conflict_tiles(3, 10)
    # x = (0, 7, 3, 2, 4, 5, 8, 1, 6)
    # y = tiledriver.Heuristic._get_linear_conflicts(x, 3)

    #c = (7, 4, 0, 5, 1, 3, 6, 2, 8)
    #print(tiledriver.Heuristic._get_linear_conflicts(c, 3))

    # g =  (7, 3, 0, 2, 5, 1, 6, 8, 4)
    print(x)


if __name__ == "__main__":
    main()
