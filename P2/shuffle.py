# Name:         
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Tile Driver II
# Term:         Summer 2021

import random
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




def conflict_tiles(width: int, min_lc: int) -> Tuple[int, ...]:
    """
    Create a solvable shuffled puzzle of the given width with a minimum number
    of linear conflicts (ignoring Manhattan distance).

    >>> tiles = conflict_tiles(3, 5)
    >>> tiledriver.Heuristic._get_linear_conflicts(tiles, 3)
    5
    """




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
    pass  # optional program test driver


if __name__ == "__main__":
    main()
