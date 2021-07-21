# Name:         Michael Long
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Biogimmickry
# Term:         Summer 2021

import random
from typing import Callable, Dict, Tuple, List


class FitnessEvaluator:

    def __init__(self, array: Tuple[int, ...]) -> None:
        """
        An instance of FitnessEvaluator has one attribute, which is a callable.

            evaluate: A callable that takes a program string as its only
                      argument and returns an integer indicating how closely
                      the program populated the target array, with a return
                      value of zero meaning the program was accurate.

        This constructor should only be called once per search.

        >>> fe = FitnessEvaluator((0, 20))
        >>> fe.evaulate(">+")
        19
        >>> fe.evaulate("+++++[>++++<-]")
        0
        """
        self.evaluate: Callable[[str], int] = \
            lambda program: FitnessEvaluator._evaluate(array, program)

    @staticmethod
    def interpret(program: str, size: int) -> Tuple[int, ...]:
        """
        Using a zeroed-out memory array of the given size, run the given
        program to update the integers in the array. If the program is
        ill-formatted or requires too many iterations to interpret, raise a
        RuntimeError.
        """
        p_ptr = 0
        a_ptr = 0
        count = 0
        max_steps = 1000
        i_map = FitnessEvaluator._preprocess(program)
        memory = [0] * size
        while p_ptr < len(program):
            if program[p_ptr] == "[":
                if memory[a_ptr] == 0:
                    p_ptr = i_map[p_ptr]
            elif program[p_ptr] == "]":
                if memory[a_ptr] != 0:
                    p_ptr = i_map[p_ptr]
            elif program[p_ptr] == "<":
                if a_ptr > 0:
                    a_ptr -= 1
            elif program[p_ptr] == ">":
                if a_ptr < len(memory) - 1:
                    a_ptr += 1
            elif program[p_ptr] == "+":
                memory[a_ptr] += 1
            elif program[p_ptr] == "-":
                memory[a_ptr] -= 1
            else:
                raise RuntimeError
            p_ptr += 1
            count += 1
            if count > max_steps:
                raise RuntimeError
        return tuple(memory)

    @staticmethod
    def _preprocess(program: str) -> Dict[int, int]:
        """
        Return a dictionary mapping the index of each [ command with its
        corresponding ] command. If the program is ill-formatted, raise a
        RuntimeError.
        """
        i_map = {}
        stack = []
        for p_ptr in range(len(program)):
            if program[p_ptr] == "[":
                stack.append(p_ptr)
            elif program[p_ptr] == "]":
                if len(stack) == 0:
                    raise RuntimeError
                i = stack.pop()
                i_map[i] = p_ptr
                i_map[p_ptr] = i
        if len(stack) != 0:
            raise RuntimeError
        return i_map

    @staticmethod
    def _evaluate(expect: Tuple[int, ...], program: str) -> int:
        """
        Return the sum of absolute differences between each index in the given
        tuple and the memory array created by interpreting the given program.
        """
        actual = FitnessEvaluator.interpret(program, len(expect))
        return sum(abs(x - y) for x, y in zip(expect, actual))

class Program:

    def __init__(self, program: str):
        self.program = program
        self.score = 0

    def __lt__(self, other):
        return self.score < other.score

    def score_fitness(self, fe: FitnessEvaluator) -> int:
        try:
            evaluated_score = fe.evaluate(self.program)
        except RuntimeError:
            evaluated_score = 1000

        self.score = evaluated_score

        return evaluated_score


def crossover(selected: List[Program]) -> List[Program]:
    new_programs = []

    if len(selected[0].program) < len(selected[1].program):
        min_program_len = len(selected[0].program)
        max_program_len = len(selected[1].program)
    else:
        min_program_len = len(selected[1].program)
        max_program_len = len(selected[0].program)

    switch_index = random.randint(0, min_program_len)

    program1 = Program(selected[0].program[0:switch_index])
    program2 = Program(selected[1].program[switch_index:max_program_len])

    new_programs.append(program1)
    new_programs.append(program2)

    return new_programs


def generate_random_program(max_len: int) -> Program:
    """
    Generates a single random program object no larger than max_len
    """
    # TODO: min length of loop sequence 12
    # TODO uncomment valid cmds and program_str

    if max_len == 0:
        max_len = 20

    program_str = ""
    # valid_commands = "><+-[]"
    valid_commands = "><+-"
    for i in range(random.randint(0, max_len)):
        print(i)
        program_str += valid_commands[random.randint(0, 3)]
        # program_str += valid_commands[random.randint(0, 5)]

    return Program(program_str)


def create_program(fe: FitnessEvaluator, max_len: int) -> str:
    """
    Return a program string no longer than max_len that, when interpreted,
    populates a memory array that exactly matches a target array.

    Use fe.evaluate(program) to get a program's fitness score (zero is best).
    """

    # mut_prob = {"<": 0.8, ">": 0.8, "+": 0.6, "-": 0.6, "[": 0.1, "]": 0.1}

    new_population: List[Program] = []

    k = 100        # k represents the initial population size
    # N = 0.5        # N is top percentile for selection process

    converges = True
    while 1:


        # generate initial random, score initial random, add to population
        if converges:
            converges = False
            # initialize empty population list
            population = []
            for i in range(k):
                print(i)
                # generate random program
                program = generate_random_program(max_len)
                # score newly generated random program and stop if 0
                fitness_score = program.score_fitness(fe)
                if fitness_score == 0:
                    return program.program

                population.append(program)
        # if no converge, just score loop and add to new pop
        else:
            # initialize empty population list
            for i in range(k):
                # generate random program
                # program = new_population[i]
                # score newly generated random program and stop if 0
                fitness_score = new_population[i].score_fitness(fe)
                if fitness_score == 0:
                    return new_population[i].program

                population[i] = new_population[i]

                # new_population.append(program)

        new_population = []

        while len(population) != len(new_population):
            # select 2 programs in top N percentile
            selected = random.choices(population, k=2)
            new_programs = crossover(selected)
            # add the new programs to the next_pop list until full
            new_population.append(new_programs[0])
            new_population.append(new_programs[1])


def main() -> None:  # optional driver
    # array = (-1, 2, -3, 4)
    array = (-7, 7, 5, 2)
    max_len = 0  # no BF loop required

    # only attempt when non-loop programs work
    # array = (20, 0)
    # max_len = 15

    program = create_program(FitnessEvaluator(array), max_len)
    #print(program)
    print(FitnessEvaluator.interpret(program, len(array)))
    """if max_len > 0:
        assert len(program) <= max_len
    assert array == FitnessEvaluator.interpret(program, len(array))"""

    # print(FitnessEvaluator.interpret("><<+++", len(array)))


if __name__ == "__main__":
    main()
