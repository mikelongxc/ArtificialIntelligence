# Name:         Michael Long
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Mine Shafted
# Term:         Summer 2021

import itertools
from typing import Callable, Generator, List, Tuple, Set


class BoardManager:  # do not modify

    def __init__(self, board: List[List[int]]):
        """
        An instance of BoardManager has two attributes.

            size: A 2-tuple containing the number of rows and columns,
                  respectively, in the game board.
            move: A callable that takes an integer as its only argument to be
                  used as the index to explore on the board. If the value at
                  that index is a clue (non-mine), this clue is returned;
                  otherwise, an error is raised.

        This constructor should only be called once per game.

        >>> board = [[0, 1, 1], [0, 2, -1], [0, 2, -1], [0, 1, 1]]
        >>> bm = BoardManager(board)
        >>> bm.size
        (4, 3)
        >>> bm.move(4)
        2
        >>> bm.move(5)
        Traceback (most recent call last):
        ...
        RuntimeError
        """
        self.size: Tuple[int, int] = (len(board), len(board[0]))
        it: Generator[int, int, None] = BoardManager._move(board, self.size[1])
        next(it)
        self.move: Callable[[int], int] = it.send

    def get_adjacent(self, index: int) -> List[int]:
        """
        Return a list of indices adjacent (including diagonally) to the given
        index. All adjacent indices are returned, regardless of whether or not
        they have been explored.

        >>> board = [[0, 1, 1], [0, 2, -1], [0, 2, -1], [0, 1, 1]]
        >>> bm = BoardManager(board)
        >>> bm.get_adjacent(3)
        [0, 1, 4, 6, 7]
        """
        row, col = index // self.size[1], index % self.size[1]
        return [i * self.size[1] + j
                for i in range(max(row - 1, 0), min(row + 2, self.size[0]))
                for j in range(max(col - 1, 0), min(col + 2, self.size[1]))
                if index != i * self.size[1] + j]

    @staticmethod
    def _move(board: List[List[int]], width: int) -> Generator[int, int, None]:
        """
        A generator that may be sent integers (indices to explore on the board)
        and yields integers (clues for the explored indices).

        Do not call this method directly; instead, call the |move| instance
        attribute, which sends its index argument to this generator.
        """
        index = (yield 0)
        while True:
            clue = board[index // width][index % width]
            if clue == -1:
                raise RuntimeError
            index = (yield clue)

class Cell:

    def __init__(self, index: int, adjacent: List[int], clue_val: int = None):
        self.index = index
        self.clue_val = clue_val
        # domain is the list of tuples. [-x: guess no mine, +x: guess mine]
        self.domain: List[Tuple[int, ...]] = []

        # generates from outside. bm.get_adjacent
        self.adjacent = adjacent
        self.finished = False

    def finish(self):
        self.finished = True


class Mineshafted:

    # keeps list of arcs

    def __init__(self):
        print("cock")
        self.arcs = []

    def main(self):
        print("X")


class State:

    def __init__(self, cells: List[Cell] = None):
        self.cells = cells

    def __repr__(self):
        return "state obj."

    def add_cell(self, cell: Cell):
        self.cells.append(cell)


def explore(bm: BoardManager):

    index = 0
    clue = bm.move(index)


def get_adjacentt(size, index):
    print("Test")
    node = (0, 0)
    dir_vector = [-1, 0, 1]
    for i in dir_vector:
        for j in dir_vector:
            if i == j == 0:
                continue
            if 0 <= node[0] + i < size[0] and 0 <= node[1] + j < size[1]:
                print((node[0]+i, node[1]+j))
                # yield (node[0] + i, node[1] + j)


def get_domain(cell: Cell) -> List[List[int]]:
    adjacent = cell.adjacent
    num_adj = len(adjacent)
    return [[n * m for n, m in zip(adjacent, combo)]
            for combo in itertools.product([1, -1], repeat=num_adj)]


def sweep_mines(bm: BoardManager) -> List[List[int]]:
    """
    Given a BoardManager (bm) instance, return a solved board (represented as a
    2D list) by repeatedly calling bm.move(index) until all safe indices have
    been explored. If at any time a move is attempted on a non-safe index, the
    BoardManager will raise an error; this error signifies the end of the game
    and should not attempt to be caught.

    >>> board = [[0, 1, 1], [0, 2, -1], [0, 2, -1], [0, 1, 1]]
    >>> bm = BoardManager(board)
    >>> sweep_mines(bm)
    [[0, 1, 1], [0, 2, -1], [0, 2, -1], [0, 1, 1]]
    """

    # for speed
    found_a_0 = False

    # 1. init: gen cell 0
    cell0 = Cell(0, bm.get_adjacent(0), 0,)
    # gen state
    state = State([])
    # add cell 0 to state
    state.add_cell(cell0)


    ct = 0
    while 1:

        if cell0.clue_val == 0:
            # 2. discover adj.
            for i in range(len(cell0.adjacent)):
                new_index = cell0.adjacent[i]
                clue = bm.move(new_index)
                new_cell = Cell(new_index, bm.get_adjacent(new_index), clue)
                state.add_cell(new_cell)
                cell0.finish()

        # 3. choose next cell if newly discovered is 0
        for i in range(len(state.cells)):
            if not state.cells[i].finished and state.cells[i].clue_val == 0:
                cell0 = state.cells[i]
                found_a_0 = True
                break

        if found_a_0:
            continue

        # cell: reduce adj list
        # cell: create domain

        # 4. gen. domains
        for i in range(len(state.cells)):
            if not state.cells[i].finished:
                # reduce adj list based on finished cells
                # generate domain based on adj list
                print()



        ct += 1




    return [[0, 1, 1], [0, 2, -1], [0, 2, -1], [0, 1, 1]]



def main() -> None:  # optional driver

    board = [[0, 0, 1, -1, 2, 1, 1, 0, 0],\
             [0, 1, 2, 2, 2, -1, 1, 1, 1],\
             [0, 1, -1, 1, 1, 1, 1, 1, -1],\
             [0, 1, 1, 1, 0, 0, 0, 1, 1],\
             [0, 0, 0, 0, 0, 1, 1, 1, 0],\
             [0, 0, 0, 0, 1, 2, -1, 2, 1],\
             [1, 1, 1, 0, 1, -1, 3, -1, 1],\
             [2, -1, 1, 0, 1, 1, 2, 2, 2],\
             [-1, 2, 1, 0, 0, 0, 0, 1, -1]]

    board = [[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, -1, 3, 2],
             [0, 1, 2, -1, -1], [0, 0, 1, 2, 2]]

    board = [[0, 1, 1], [0, 2, -1], [0, 2, -1], [0, 1, 1]]


    bm = BoardManager(board)
    assert sweep_mines(bm) == board


if __name__ == "__main__":
    main()
