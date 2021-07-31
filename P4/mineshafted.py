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
        self.domain: List[List[int, ...]] = []
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
        return str((self.index, self.finished, self.clue_val))

    def finish(self):
        self.finished = True


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

    def finish(self, cell: Cell):
        self.cells[cell.index].finished = True


def get_domain(cell: Cell, adjacent: List[int], known_mines: List[int]) \
        -> List[List[int]]:
    num_adj = len(adjacent)

    saved_popped_indexes = []

    known_mines_set = set(known_mines)
    known_mines = list(known_mines_set)

    unreduced = [[n * m for n, m in zip(adjacent, cell_combo)]
            for cell_combo in itertools.product([1, -1], repeat=num_adj)]

    reduced = []
    clue = cell.clue_val
    for i in range(len(unreduced)):
        num_mines = 0
        for j in range(len(unreduced[i])):
            if unreduced[i][j] > 0:
                num_mines += 1
        if num_mines == clue:
            reduced.append(unreduced[i])

    for i in range(len(known_mines)):
        for j in range(len(reduced)):
            for k in range(len(reduced[j])):
                # if found +- of known mine
                if reduced[j][k] == known_mines[i] \
                        or -1 * reduced[j][k] == known_mines[i]:
                    if reduced[j][k] != known_mines[i]:
                        saved_popped_indexes.append(j)

    if len(saved_popped_indexes) > 0:
        temp = set(saved_popped_indexes)
        saved_popped_indexes.sort()
        saved_popped_indexes = list(temp)
        saved_popped_indexes.sort()
        saved_popped_indexes.reverse()
        for i in range(len(saved_popped_indexes)):
            reduced.pop(saved_popped_indexes[i])

    return reduced


class Mineshafted:

    # keeps list of arcs

    def __init__(self, bm: BoardManager):
        self.bm = bm
        self.found_a_0 = False

        self.arcs_set = set()
        self.arcs_ordered: List[Tuple[int, ...]] = []

        self.board_size = bm.size

        self.known_mines = []

        # gen empty board for later return TODO
        self.board = []
        for r in range(self.board_size[0]):
            self.board.append([])
            for _ in range(self.board_size[1]):
                self.board[r].append(None)

        # gen empty list for State TODO move to state class
        self.state = State([])
        for r in range(self.board_size[0] * self.board_size[1]):
            self.state.cells.append(Cell(r, bm.get_adjacent(r), False))

        self.len_cells = len(self.state.cells)

    def solve(self) -> List[List[int]]:

        # init cell0, add to state
        cell0 = Cell(0, self.bm.get_adjacent(0), True, 0)
        self.board[0][0] = 0
        self.state.add_cell(cell0)

        new_safe = set()

        while 1:
            # 2. discover adj. to cell0

            self.discover_adjacent(cell0)

            if len(new_safe) != 0:
                index = -1 * new_safe.pop()
                clue = self.bm.move(index)
                next_cell = Cell(index, self.bm.get_adjacent(index), True, clue)

                self.state.add_cell(next_cell)

                self.board[index // self.board_size[1]]\
                    [index % self.board_size[1]] = clue

                if self.board_filled(self.board):
                    return self.board

                continue

            # 3. choose next cell if one of the newly discovered was 0
            for i in range(self.len_cells):
                if not self.state.cells[i].finished \
                        and self.state.cells[i].clue_val == 0:
                    cell0 = self.state.cells[i]
                    self.found_a_0 = True
                    break

            if self.found_a_0:
                continue

            # 4. gen. domains of each unfinished cell
            self.generate_domains()

            # 5. determine arcs
            self.determine_arcs()

            # 6. run constraint propagation (ac-3)
            self.ac_3()

            # 7. choose next cell values and mine locations based on reduced
            new_safe = set()
            """for i in range(self.len_cells):
                for j in range(len(self.state.cells[i].domain)):
                    if len(self.state.cells[i].domain) < 2:
                        for n in range(len(self.state.cells[i].domain[j])):
                            idx = self.state.cells[i].domain[j][n]
                            if self.state.cells[i].domain[j][n] < 0:
                                new_safe.add(idx)
                            else:
                                print()
                                # TODO: store in board. convert to 2d?
                                width = self.board_size[1]
                                self.board[idx // width][idx % width] = -1"""
            self.choose_safe_cells(new_safe)

            if self.board_filled(self.board):
                break

        return self.board

    def choose_safe_cells(self, new_safe: Set[int]):
        for i in range(self.len_cells):
            for j in range(len(self.state.cells[i].domain)):
                if len(self.state.cells[i].domain) < 2:
                    self.state.finish(self.state.cells[i])
                    for n in range(len(self.state.cells[i].domain[j])):
                        idx = self.state.cells[i].domain[j][n]
                        if self.state.cells[i].domain[j][n] < 0:
                            new_safe.add(idx)
                        else:
                            # store resulting mine in board and save cell
                            m = Cell(idx, self.bm.get_adjacent(idx), False, -1)
                            m.is_mine = True
                            self.state.add_cell(m)
                            self.state.finish(m)
                            width = self.board_size[1]
                            self.board[idx // width][idx % width] = -1
                            self.known_mines.append(idx)

    def board_filled(self, board: List[List[int]]) -> bool:
        for i in range(len(board)):
            for j in range(len(board[i])):
                if not board[i][j] and board[i][j] != 0:
                    return False
        return True

    def discover_adjacent(self, cell0: Cell):
        """
        given a cell with known index 0, move to all adj indices and add acc.
        """
        if cell0.clue_val == 0 and not cell0.finished:
            for i in range(len(cell0.adjacent)):
                n_index = cell0.adjacent[i]
                clue = self.bm.move(n_index)
                n_cell = Cell(n_index,\
                              self.bm.get_adjacent(n_index), True, clue)
                self.state.add_cell(n_cell)
                self.state.finish(cell0)
                self.found_a_0 = False

                self.board[n_index // self.board_size[1]] \
                    [n_index % self.board_size[1]] = clue

    def generate_domains(self):
        # 4. gen. domains of each unfinished cell

        for i in range(self.len_cells):
            if not self.state.cells[i].finished\
                    and self.state.cells[i].discovered:
                old_adj = self.state.cells[i].adjacent
                new_adj = []
                for j in range(len(old_adj)):
                    if not self.state.cells[old_adj[j]].discovered:
                        new_adj.append(old_adj[j])
                self.state.cells[i].domain_blueprint = new_adj
                # from reduced adjacency list, get domain
                domain = get_domain\
                    (self.state.cells[i], new_adj, self.known_mines)
                self.state.cells[i].domain = domain

    def determine_arcs(self):
        # 5. determine arcs
        for i in range(self.len_cells):
            arc_true = False
            if not self.state.cells[i].finished \
                    and self.state.cells[i].discovered:
                for j in range(self.len_cells):
                    arc_true = False
                    if i != j and self.state.cells[j].discovered \
                            and not self.state.cells[j].finished:
                        for k in range(len(self.state.cells[i]\
                                                        .domain_blueprint)):
                            for l in range(
                                    len(self.state.cells[j].domain_blueprint)):
                                if self.state.cells[i].domain_blueprint[k] == \
                                        self.state.cells[j].domain_blueprint[l]:
                                    arc_true = True
                                    break
                            if arc_true:
                                break
                        if arc_true:
                            self.arcs_set.add((i, j))
                            self.arcs_set.add((j, i))

                            if len(self.arcs_ordered) < len(self.arcs_set):
                                self.arcs_ordered.append((i, j))
                                self.arcs_ordered.append((j, i))

    def ac_3(self):

        # mine_list = []

        arcs_copy = copy_arcs(self.arcs_ordered)
        while self.arcs_ordered:
            arc = self.arcs_ordered.pop(0)
            a = self.state.cells[arc[0]]
            b = self.state.cells[arc[1]]

            a_domain = a.domain.copy()
            real_a_domain = copy_domain(a_domain)
            b_domain = b.domain.copy()

            common = \
                list(set(a.domain_blueprint).intersection(b.domain_blueprint))

            a_reduced = make_new_domain(a_domain, common)
            b_reduced = make_new_domain(b_domain, common)

            reduced = False
            saved_popped_indexes = []
            for i in range(len(a_reduced)):
                # REDUCE if found
                if a_reduced[i] not in b_reduced:
                    # pop invalid variable off of real location
                    # real_a_domain.pop(i)
                    saved_popped_indexes.append(i)
                    print("REDUCE")
                    # add arcs back (*, x)
                    reduced = True

            if reduced:
                for j in range(len(arcs_copy)):
                    if arcs_copy[j][1] == arc[0] \
                            and arcs_copy[j][0] != arc[1]:
                        self.arcs_ordered.append(tuple(arcs_copy[j]))

            saved_popped_indexes.reverse()
            for i in range(len(saved_popped_indexes)):
                real_a_domain.pop(saved_popped_indexes[i])

            self.state.cells[arc[0]].domain = real_a_domain

            # check for any possible mines (both indices are pos)

            # mine_list = mine_list + check_mines(real_A_domain)
            # if len(real_A_domain) > 1:
            #    mine_list = check_mines(real_A_domain)


def check_mines(domain: List[List[int]]) -> List[int]:
    """consistent_mines = False
    for i in range(len(domain) - 1):
        for j in range(len(domain[i])):
            if domain[i][j] != domain[i + 1][j]:
                consistent_mines = False
            else:
                consistent_mines = True"""

    # for each number
    mine_list = []
    mine = True
    for i in range(len(domain[0])):
        for j in range(len(domain) - 1):
            if domain[j][i] != domain[j + 1][i]:
                mine = False
        if mine == True:
            mine_list.append(domain[0][i])
        mine = True

    return mine_list


def make_new_domain(domain: List[List[int]], common: List[int]) \
        -> List[List[int]]:

    new_domain: List[List[int]] = []

    for i in range(len(domain)):
        new_combo = []
        for j in range(len(domain[i])):
            for n in range(len(common)):
                if domain[i][j] != common[n] and domain[i][j] != -1 * common[n]:
                    continue
                else:
                    new_combo.append(domain[i][j])
        new_domain.append(new_combo)

    return new_domain


def copy_domain(domain: List[List[int]]) -> List[List[int]]:
    new_domain = []
    for i in range(len(domain)):
        combo = []
        for j in range(len(domain[i])):
            combo.append(domain[i][j])
        new_domain.append(combo)
    return new_domain


def copy_arcs(arcs: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    new_arcs = []
    for i in range(len(arcs)):
        new_arcs.append(arcs[i] + tuple())

    return new_arcs


def in_list(val: int, ls: List[int]) -> bool:
    for i in range(len(ls)):
        if ls[i] == val:
            return True
    return False


def sweep_mines(bm: BoardManager) -> List[List[int]]:
    """
    Given a BoardManager (bm) instance, return a solved board (represented as a
    2D list) by repeatedly calling bm.move(index) until all safe indices have
    been explored. If at any time a move is attempted on a non-safe index, the
    BoardManager will raise an error; this error signifies the end of the game
    and will not be caught.
    """
    driver = Mineshafted(bm)
    board = driver.solve()

    return board


def main() -> None:  # optional driver

    """board4 = [[0, 0, 1, -1, 2, 1, 1, 0, 0], \
             [0, 1, 2, 2, 2, -1, 1, 1, 1], \
             [0, 1, -1, 1, 1, 1, 1, 1, -1], \
             [0, 1, 1, 1, 0, 0, 0, 1, 1], \
             [0, 0, 0, 0, 0, 1, 1, 1, 0], \
             [0, 0, 0, 0, 1, 2, -1, 2, 1], \
             [1, 1, 1, 0, 1, -1, 3, -1, 1], \
             [2, -1, 1, 0, 1, 1, 2, 2, 2], \
             [-1, 2, 1, 0, 0, 0, 0, 1, -1]] # solves

    board3 = [[0, 0, 0, 0, 0], \
             [0, 1, 1, 1, 0], \
             [0, 1, -1, 3, 2], \
             [0, 1, 2, -1, -1], \
             [0, 0, 1, 2, 2]] # solves

    board2 = [[0, 0, 0, 0, 0, 0, 2, -1, 2],
              [1, 1, 0, 0, 0, 0, 3, -1, 3],
              [-1, 1, 0, 0, 0, 0, 2, -1, 2],
              [1, 1, 0, 0, 0, 0, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 2, 2, 1, 0, 1, 1, 2, 1],
              [1, -1, -1, 1, 0, 1, -1, 2, -1],
              [1, 2, 2, 1, 0, 1, 2, 4, 3],
              [0, 0, 0, 0, 0, 0, 1, -1, -1]] # solves

    board1 = [[0, 0, 0], [1, 1, 1], [1, -1, 2], [1, 2, -1]]  # solves

    board = [[0, 1, -1, 2, 1, 1, 0, 1, -1],
              [0, 1, 2, 3, -1, 1, 1, 2, 2],
              [0, 0, 1, -1, 2, 1, 1, -1, 1],
              [1, 1, 2, 1, 1, 0, 1, 1, 1],
              [1, -1, 2, 1, 0, 0, 0, 0, 0],
              [1, 2, -1, 1, 0, 0, 0, 0, 0],
              [0, 1, 1, 1, 0, 0, 0, 1, 1],
              [0, 1, 1, 1, 0, 0, 1, 2, -1],
              [0, 1, -1, 1, 0, 0, 1, -1, 2]] # UNSOLVABLE"""



    board = [[0, 1, -1, 3, 2, 1], [0, 1, 2, -1, -1, 1], [0, 0, 1, 2, 2, 1]] #no

    board = [[0, 1, 2, 2], [0, 1, -1, -1], [0, 1, 2, 2]]

    # board = [[0, 1, 1], [1, 2, -1], [-1, 2, 1], [1, 1, 0]] # infinite

    # board = [[0, 1, -1], [2, 3, 1], [-1, -1, 1]] # UNSOLVABLE

    #board = [[0, 1, 1], [0, 2, -1], [0, 2, -1], [0, 1, 1]]


    print("testing board: ")
    test(board)


def test(board: List[List[int]]) -> None:
    bm = BoardManager(board)
    assert sweep_mines(bm) == board


def test_all() -> None:
    board4 = [[0, 0, 1, -1, 2, 1, 1, 0, 0], \
              [0, 1, 2, 2, 2, -1, 1, 1, 1], \
              [0, 1, -1, 1, 1, 1, 1, 1, -1], \
              [0, 1, 1, 1, 0, 0, 0, 1, 1], \
              [0, 0, 0, 0, 0, 1, 1, 1, 0], \
              [0, 0, 0, 0, 1, 2, -1, 2, 1], \
              [1, 1, 1, 0, 1, -1, 3, -1, 1], \
              [2, -1, 1, 0, 1, 1, 2, 2, 2], \
              [-1, 2, 1, 0, 0, 0, 0, 1, -1]]  # solves

    board3 = [[0, 0, 0, 0, 0], \
              [0, 1, 1, 1, 0], \
              [0, 1, -1, 3, 2], \
              [0, 1, 2, -1, -1], \
              [0, 0, 1, 2, 2]]  # solves

    board2 = [[0, 0, 0, 0, 0, 0, 2, -1, 2],
              [1, 1, 0, 0, 0, 0, 3, -1, 3],
              [-1, 1, 0, 0, 0, 0, 2, -1, 2],
              [1, 1, 0, 0, 0, 0, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 2, 2, 1, 0, 1, 1, 2, 1],
              [1, -1, -1, 1, 0, 1, -1, 2, -1],
              [1, 2, 2, 1, 0, 1, 2, 4, 3],
              [0, 0, 0, 0, 0, 0, 1, -1, -1]]  # solves

    board1 = [[0, 0, 0], [1, 1, 1], [1, -1, 2], [1, 2, -1]]  # solves

    print("testing board 1: ")
    test(board1)
    print("testing board 2: ")
    test(board2)
    print("testing board 3: ")
    test(board3)
    print("testing board 4: ")
    test(board4)


if __name__ == "__main__":
    main()
