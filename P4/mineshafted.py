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

    def __init__(self, index: int, adjacent: List[int],\
                 discovered: int, clue_val: int = None):

        self.index = index
        self.clue_val = clue_val
        # domain is the list of tuples. [-x: guess no mine, +x: guess mine]
        self.domain: List[Tuple[int, ...]] = []
        self.is_mine = False

        # generates from outside. bm.get_adjacent
        self.adjacent = adjacent
        self.finished = False
        self.discovered = discovered

        self.domain_blueprint: List[int] = []

    def __eq__(self, other):
        if other.index == self.index:
            return False
        return True

    def __repr__(self):
        return str((self.index,  self.finished, self.clue_val))

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
        return str(self.cells)

    def oldadd_cell(self, cell: Cell):
        """
        adds UNIQUE element to list of cells
        """
        duplicate = False
        l = len(self.cells)
        for i in range(l):
            idx = self.cells[i].index
            if idx == cell.index:
                duplicate = True
        if not duplicate:
            self.cells.append(cell)

    def add_cell(self, cell: Cell):
        if not self.cells[cell.index].finished:
            self.cells[cell.index] = cell

    def finish(self, cell):
        self.cells[cell.index].finished = True



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


def get_domain(cell: Cell, adjacent: List[int]) -> List[List[int]]:
    num_adj = len(adjacent)

    unreduced = [[n * m for n, m in zip(adjacent, arc)]
            for arc in itertools.product([1, -1], repeat=num_adj)]

    reduced = []
    clue = cell.clue_val
    for i in range(len(unreduced)):
        num_mines = 0
        for j in range(len(unreduced[i])):
            if unreduced[i][j] > 0:
                num_mines += 1
        if num_mines == clue:
            reduced.append(unreduced[i])

    return reduced


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
    #arcs: Set[Tuple[int, ...]]

    arcs_set = set()
    arcs_ordered: List[Tuple[int, ...]] = []

    # gen empty board for later return TODO
    board = []
    size = bm.size
    for r in range(size[0]):
        board.append([])
        for c in range(size[1]):
            board[r].append(None)

    # gen empty list for State TODO move to state class
    state = State([])
    for r in range(size[0] * size[1]):
        state.cells.append(Cell(r, bm.get_adjacent(r), False))

    len_cells = len(state.cells)



    # 1. init: gen cell 0
    cell0 = Cell(0, bm.get_adjacent(0), True, 0)

    # add cell 0 to state
    state.add_cell(cell0)


    ct = 0
    while 1:

        if cell0.clue_val == 0 and not cell0.finished:
            # 2. discover adj.
            for i in range(len(cell0.adjacent)):
                n_index = cell0.adjacent[i]
                clue = bm.move(n_index)
                n_cell = Cell(n_index, bm.get_adjacent(n_index), True, clue)
                state.add_cell(n_cell)
                state.finish(cell0)
                found_a_0 = False

        # 3. choose next cell if newly discovered is 0
        for i in range(len_cells):
            if not state.cells[i].finished and state.cells[i].clue_val == 0:
                cell0 = state.cells[i]
                found_a_0 = True
                break

        if found_a_0:
            continue

        # cell: reduce adj list
        # cell: create domain

        # 4. gen. domains of each unfinished cell
        for i in range(len_cells):
            if not state.cells[i].finished and state.cells[i].discovered:
                old_adj = state.cells[i].adjacent
                new_adj = [] # TODO
                arc_adj = []
                for j in range(len(old_adj)):
                    # TODO remove or create new adj list???
                    if not state.cells[old_adj[j]].discovered:
                        new_adj.append(old_adj[j])
                    """elif state.cells[old_adj[j]].discovered \
                        and not state.cells[old_adj[j]].finished:
                            arc_adj.append(old_adj[j])"""

                state.cells[i].domain_blueprint = new_adj

                # get domain
                state.cells[i].domain = get_domain(state.cells[i], new_adj)

                """for k in range(len(arc_adj)):
                    new_arc = (i, k)"""

        # determine an arc
        for i in range(len_cells):
            arc_true = False
            if not state.cells[i].finished and state.cells[i].discovered:
                for j in range(len_cells):
                    arc_true = False
                    if i != j and state.cells[j].discovered and not state.cells[j].finished:
                        for k in range(len(state.cells[i].domain_blueprint)):
                            for l in range(len(state.cells[j].domain_blueprint)):
                                if state.cells[i].domain_blueprint[k] == state.cells[j].domain_blueprint[l]:
                                    arc_true = True
                                    break
                            if arc_true:
                                break
                        if arc_true:
                            arcs_set.add((i, j))
                            arcs_set.add((j, i))

                            if len(arcs_ordered) < len(arcs_set):
                                arcs_ordered.append((i, j))
                                arcs_ordered.append((j, i))











        # 5. find arcs between multiple cells

        # x. determine which cell is safe. cell0 = . reset loop to explore

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
