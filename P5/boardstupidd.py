# Name:         Michael Long
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Board Stupid
# Term:         Summer 2021

import math
import random
from typing import Callable, Generator, Optional, Tuple, List


class GameState:

    def __init__(self, board: Tuple[Tuple[Optional[int], ...], ...],
                 player: int) -> None:
        """
        An instance of GameState has the following attributes.

            player: Set as either 1 (MAX) or -1 (MIN).
            moves: A tuple of integers representing empty indices of the board.
            selected: The index that the current player believes to be their
                      optimal move; defaults to -1.
            util: The utility of the board; either 1 (MAX wins), -1 (MIN wins),
                  0 (tie game), or None (non-terminal game state).
            traverse: A callable that takes an integer as its only argument to
                      be used as the index to apply a move on the board,
                      returning a new GameState object with this move applied.
                      This callable provides a means to traverse the game tree
                      without modifying parent states.
            display: A string representation of the board, which should only be
                     used for debugging and not parsed for strategy.

        In addition, instances of GameState may be stored in hashed
        collections, such as sets or dictionaries.

        >>> board = ((   0,    0,    0,    0,   \
                         0,    0, None, None,   \
                         0, None,    0, None,   \
                         0, None, None,    0),) \
                    + ((None,) * 16,) * 3

        >>> state = GameState(board, 1)
        >>> state.util
        None
        >>> state.player
        1
        >>> state.moves
        (0, 1, 2, 3, 4, 5, 8, 10, 12, 15)
        >>> state = state.traverse(0)
        >>> state.player
        -1
        >>> state.moves
        (1, 2, 3, 4, 5, 8, 10, 12, 15)
        >>> state = state.traverse(5)
        >>> state.player
        1
        >>> state.moves
        (1, 2, 3, 4, 8, 10, 12, 15)
        >>> state = state.traverse(1)
        >>> state.player
        -1
        >>> state.moves
        (2, 3, 4, 8, 10, 12, 15)
        >>> state = state.traverse(10)
        >>> state.player
        1
        >>> state.moves
        (2, 3, 4, 8, 12, 15)
        >>> state = state.traverse(2)
        >>> state.player
        -1
        >>> state.moves
        (3, 4, 8, 12, 15)
        >>> state = state.traverse(15)
        >>> state.player
        1
        >>> state.moves
        (3, 4, 8, 12)
        >>> state = state.traverse(3)
        >>> state.util
        1
        """
        self.player: int = player
        self.moves: Tuple[int] = GameState._get_moves(board, len(board))
        self.selected: int = -1
        self.util: Optional[int] = GameState._get_utility(board, len(board))
        self.traverse: Callable[[int], GameState] = \
            lambda index: GameState._traverse(board, len(board), player, index)
        self.display: str = GameState._to_string(board, len(board))
        self.keys: Tuple[int, ...] = tuple(hash(single) for single in board)

    def __eq__(self, other: "GameState") -> bool:
        return self.keys == other.keys

    def __hash__(self) -> int:
        return hash(self.keys)

    def __repr__(self):
        return str(self.util) + str(self.moves)

    @staticmethod
    def _traverse(board: Tuple[Tuple[Optional[int], ...], ...],
                  width: int, player: int, index: int) -> "GameState":
        """
        Return a GameState instance in which the board is updated at the given
        index by the current player.

        Do not call this method directly; instead, call the |traverse| instance
        attribute, which only requires an index as an argument.
        """
        i, j = index // width ** 2, index % width ** 2
        single = board[i][:j] + (player,) + board[i][j + 1:]
        return GameState(board[:i] + (single,) + board[i + 1:], -player)

    @staticmethod
    def _get_moves(board: Tuple[Tuple[Optional[int], ...], ...],
                   width: int) -> Tuple[int]:
        """
        Return a tuple of the unoccupied indices remaining on the board.
        """
        return tuple(j + i * width ** 2 for i, single in enumerate(board)
                     for j, square in enumerate(single) if square == 0)

    @staticmethod
    def _get_utility(board: Tuple[Tuple[Optional[int], ...], ...],
                     width: int) -> Optional[int]:
        """
        Return the utility of the board; either 1 (MAX wins), -1 (MIN wins),
        0 (tie game), or None (non-terminal game state).
        """
        for line in GameState._iter_lines(board, width):
            if line == (1,) * width:
                return 1
            if line == (-1,) * width:
                return -1
        return 0 if len(GameState._get_moves(board, width)) == 0 else None

    @staticmethod
    def _iter_lines(board: Tuple[Tuple[Optional[int], ...], ...],
                    width: int) -> Generator[Tuple[int], None, None]:
        """
        Iterate over all groups of indices that represent a winning condition.
        X lines are row-wise, Y lines are column-wise, and Z lines go through
        all single boards; combinations of these axes refer to the direction
        of the line in 2D or 3D space.
        """
        for single in board:
            # x lines (2D rows)
            for i in range(0, len(single), width):
                yield single[i:i + width]
            # y lines (2D columns)
            for i in range(width):
                yield single[i::width]
            # xy lines (2D diagonals)
            yield single[::width + 1]
            yield single[width - 1:len(single) - 1:width - 1]
        # z lines
        for i in range(width ** 2):
            yield tuple(single[i] for single in board)
        for j in range(width):
            # xz lines
            yield tuple(board[i][j * width + i] for i in range(len(board)))
            yield tuple(board[i][j * width + width - 1 - i]
                        for i in range(len(board)))
            # yz lines
            yield tuple(board[i][j + i * width] for i in range(len(board)))
            yield tuple(board[i][-j - 1 - i * width]
                        for i in range(len(board)))
        # xyz lines
        yield tuple(board[i][i * width + i] for i in range(len(board)))
        yield tuple(board[i][i * width + width - 1 - i]
                    for i in range(len(board)))
        yield tuple(board[i][width ** 2 - width * (i + 1) + i]
                    for i in range(len(board)))
        yield tuple(board[i][width ** 2 - (i * width) - i - 1]
                    for i in range(len(board)))

    @staticmethod
    def _to_string(board: Tuple[Tuple[Optional[int], ...], ...],
                   width: int) -> str:
        """
        Return a string representation of the game board, in which integers
        represent the indices of empty spaces and the characters "X" and "O"
        represent previous move selections for MAX and MIN, repsectively.
        """
        display = "\n"
        for i in range(width):
            for j in range(width):
                line = board[j][i * width:i * width + width]
                start = j * width ** 2 + i * width
                for k, space in enumerate(line):
                    if space == 0:
                        space = start + k
                    else:
                        space = ("X" if space == 1
                                 else "O" if space == -1
                                 else "-")
                    display += "{0:>4}".format(space)
                display += " " * width
            display += "\n"
        return display


class StateNode:

    def __init__(self, index: int, state: GameState,\
                 parent: 'StateNode', root_child: bool):

        self.parent = parent
        self.index = index
        self.state = state

        self.explored_moves: List[StateNode] = []
        self.explored_moves_idx: List[int] = []

        self.root_child = root_child

        self.w = 0
        self.n = 0
        self.t = 0

        self.c = 2 ** 0.5

    def __repr__(self):
        return str(self.index) + ": " + str(self.w) + "/" + str(self.n)

    def get_ucb(self) -> float:
        if self.n == 0:
            return self.c
        return self.w / self.n + (self.c * (math.log(self.t, 2.87) / self.n))

    def get_win_ratio(self) -> float:
        if self.t != 0:
            return self.w / self.t
        return 0

    def update_wins_and_attempts(self, util: int):
        if util == 1:
            self.w += 1
        elif util == 0:
            self.w += 0.5
        self.n += 1
        self.t += 1


class GameTree: # not a real tree structure, just manages the game

    def __init__(self, state: GameState):
        self.traverse_queue = []
        self.root_children: List[StateNode] = []
        self.explored: List[StateNode] = []
        self.frontier: List[StateNode] = []

        self.state = state
        self.debug = None

    def __repr__(self):
        return "GameTree"

    def find_best_move(self) -> None:

        # 1. add all root child nodes to the frontier, generate them
        self._generate_root_child_states()

        # 2. choose first node (rdm)
        selected = self._find_max_ucb(self.root_children)

        first_pass = 0

        for x in range(5000):

            pass



    def _choose_best_ucb(self) -> StateNode:

        # 1. find the best ucb of the root children
        selected = self._find_max_ucb(self.root_children)
        c = selected.c

        # traverse tree if ucb already exists
        while len(selected.explored_moves) > 0:
            parent = selected

            child_best = self._find_max_ucb(selected.explored_moves)

            # if node worse than c, select new random node # TODO: expand?
            if child_best.get_ucb() < c:
                selected = self._new_random_node(parent)

        return selected

    def _new_random_node(self, parent: StateNode) -> StateNode:
        parent_moves = list(parent.state.moves)
        parent_explored_idx = parent.explored_moves_idx

        # for each move that the parent has already explored:
        for i in range(len(parent_explored_idx)):
            # remove already explored moves from move
            if len(parent_moves) > 0:
                parent_moves.remove(parent_explored_idx[i])

        # pick random move number from parent moves
        random_move = parent_moves[random.randint(0, len(parent_moves) - 1)]

        # generate next node from the parent. this will be expanded
        next_state = parent.state.traverse(random_move)
        new_node = StateNode(random_move, next_state, parent, False)

        return new_node

    def _generate_root_child_states(self) -> List[StateNode]:
        root_children: List[StateNode] = []

        root = StateNode(-1, self.state, None, False)

        for index in self.state.moves:
            move_state = self.state.traverse(index)
            new_move = StateNode(index, move_state, root, True)
            self.frontier.append(new_move)
            root_children.append(new_move)

        self.root_children = root_children

        return root_children

    def _find_max_ucb(self, ls: List[StateNode]) -> StateNode:
        """
        :return: random node or best ucb node
        """
        ls_len = len(ls)
        # get random idx of list
        rdm_idx = random.randint(0, ls_len - 1)

        # choose random node to compare (for root child logic)
        cur_max = ls[rdm_idx].get_ucb()
        cur_node = ls[rdm_idx]

        # compare each node ucb. if better node found, use that
        for i in range(ls_len):
            ucb = ls[i].get_ucb()
            if ucb > cur_max:
                cur_max = ucb
                cur_node = ls[i]

        return cur_node


def find_best_move(state: GameState) -> None:
    """
    Search the game tree for the optimal move for the current player, storing
    the index of the move in the given GameState object's selected attribute.
    The move must be an integer indicating an index in the 3D board - ranging
    from 0 to 63 - with 0 as the index of the top-left space of the top board
    and 63 as the index of the bottom-right space of the bottom board.

    This function must perform a Monte Carlo Tree Search to select a move,
    using additional functions as necessary. During the search, whenever a
    better move is found, the selected attribute should be immediately updated
    for retrieval by the instructor's game driver. Each call to this function
    will be given a set number of seconds to run; when the time limit is
    reached, the index stored in selected will be used for the player's turn.
    """

    g = GameTree(state)
    g.find_best_move()


def main() -> None:

    option = 1

    if option == 1:
        test()
    elif option == 2:
        play_game()


def test() -> None:
    """board = ((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))"""
    board = ((0, 0, 0, 0,
                 0, 0, None, None,
                 0, None, 0, None,
                 0, None, None, 0),) \
            + ((None,) * 16,) * 3

    """board = ((0, 0, 0, 0,
              0, 0, None, None,
              0, None, 0, None,
              0, None, None, 0),
             (0, 0, 0, 0,
              0, 0, None, None,
              0, None, 0, None,
              0, None, None, 0),) \
            + ((None,) * 16,) * 2"""
    state = GameState(board, 1)
    print(state.display)
    find_best_move(state)
    print("SELECTED: " + str(state.selected))
    # assert state.selected == 0


def play_game() -> None:
    """
    Play a game of 3D Tic-Tac-Toe with the computer.

    If you lose, you lost to a machine.
    If you win, your implementation was bad.
    You lose either way.
    """
    board = tuple(tuple(0 for _ in range(i, i + 16))
                  for i in range(0, 64, 16))
    board = ((0, 0, 0, 0,
              0, 0, None, None,
              0, None, 0, None,
              0, None, None, 0),) \
            + ((None,) * 16,) * 3

    """board = ((0, 0, 0, 0,
              0, 0, None, None,
              0, None, 0, None,
              0, None, None, 0),
             (0, 0, 0, 0,
              0, 0, None, None,
              None, None, 0, None,
              None, None, None, 0),) \
            + ((None,) * 16,) * 2"""

    state = GameState(board, 1)
    while state.util is None:
        # human move
        print(state.display)
        state = state.traverse(int(input("Move: ")))
        if state.util is not None:
            break
        # computer move
        find_best_move(state)
        move = (state.selected if state.selected != -1
                else random.choice(state.moves))
        state = state.traverse(move)
    print(state.display)
    if state.util == 0:
        print("Tie Game")
    else:
        print(f"Player {state.util} Wins!")


if __name__ == "__main__":
    main()
