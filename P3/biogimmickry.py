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
        z = sum(abs(x - y) for x, y in zip(expect, actual))
        return z

class Program:

    def __init__(self, sequence: str):
        self.sequence = sequence
        self.score = 0

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        return str(self.score)

    def score_fitness(self, fe: FitnessEvaluator) -> int:
        try:
            evaluated_score = fe.evaluate(self.sequence)
        except RuntimeError:
            evaluated_score = 1000

        self.score = evaluated_score

        return evaluated_score


def crossover(selected: List[Program]) -> List[Program]:
    new_programs = []

    if len(selected[0].sequence) < len(selected[1].sequence):
        min_program_len = len(selected[0].sequence)
        max_program_len = len(selected[1].sequence)
    else:
        min_program_len = len(selected[1].sequence)
        max_program_len = len(selected[0].sequence)

    switch_index = random.randint(0, min_program_len)

    #program1_seq = selected[0].sequence[0:switch_index]
    #program2_seq = selected[1].sequence[switch_index:max_program_len]

    swap1a = selected[0].sequence[0:switch_index]
    swap1b = selected[1].sequence[switch_index:max_program_len]

    swap2a = selected[1].sequence[0:switch_index]
    swap2b = selected[0].sequence[switch_index:max_program_len]

    program1_seq = swap1a + swap1b
    program2_seq = swap2a + swap2b

    program1_seq = mutate(program1_seq)
    program2_seq = mutate(program2_seq)

    program1 = Program(program1_seq)
    program2 = Program(program2_seq)

    # add new programs to return list
    new_programs.append(program1)
    new_programs.append(program2)

    return new_programs


def mutate(sequence: str) -> str:
    seq_len = len(sequence)
    index = random.randint(0, seq_len)
    valid_commands = "><+-"

    rdm_int = random.randint(-3, 3) # TODO: add +2 to 3 for []
    if rdm_int < 0:
        return sequence

    new_cmd = valid_commands[rdm_int]

    # sequence = sequence.replace(sequence,)

    return sequence[:index] + new_cmd + sequence[index + 1:]


def generate_random_program(max_len: int) -> Program:
    """
    Generates a single random program object no larger than max_len
    """
    # TODO: min length of loop sequence 12
    # TODO uncomment valid cmds and program_str

    if max_len == 0:
        max_len = 20

    sequence_str = ""
    # valid_commands = "><+-[]"
    valid_commands = "><+-"
    for i in range(random.randint(0, max_len)):
        # print(i)
        sequence_str += valid_commands[random.randint(0, 3)]

    return Program(sequence_str)


def create_program(fe: FitnessEvaluator, max_len: int) -> str:
    """
    Return a program string no longer than max_len that, when interpreted,
    populates a memory array that exactly matches a target array.

    Use fe.evaluate(program) to get a program's fitness score (zero is best).
    """

    # mut_prob = {"<": 0.8, ">": 0.8, "+": 0.6, "-": 0.6, "[": 0.1, "]": 0.1}

    # new_population: List[Program] = []

    k = 4000        # k represents the initial population size
    # N = 0.5        # N is top percentile for selection process

    converges = True
    gen_no = 0
    while 1:
        gen_no = gen_no + 1
        # print(gen_no)
        if gen_no == 15:
            converges = True
            gen_no = 0

        # generate initial random, score initial random, add to population
        if converges:
            converges = False
            # initialize empty population list
            population = []
            for i in range(k):
                # generate random program
                program = generate_random_program(max_len)
                # score newly generated random program and stop if 0
                fitness_score = program.score_fitness(fe)
                if fitness_score == 0:
                    # print("gen no (regen): " + str(gen_no))
                    return program.sequence
                population.append(program)

        new_population: List[Program] = []

        while len(population) != len(new_population):
            # select 2 programs in top N percentile
            n = k//10

            weights = []
            for i in range(len(population)):
                weights.append(10 - population[i].score)

            # weights = populate_weights(k, population)

            selected = random.choices(population, weights=weights, k=n)

            for i in range(0, n, 2):
                n_p = []
                n_p.append(selected[i])
                n_p.append(selected[i+1])

                new_programs = crossover(n_p)

                score1 = new_programs[0].score_fitness(fe)
                score2 = new_programs[1].score_fitness(fe)

                if score1 == 0:
                    return new_programs[0].sequence
                elif score2 == 0:
                    return new_programs[1].sequence

                # add the new programs to the next_pop list until full
                new_population.append(new_programs[0])
                new_population.append(new_programs[1])

        for i in range(k):
            population[i] = new_population[i]

        # copy_array(population, new_population, k)


def populate_weights(k: int, population: List[Program]) -> List[int]:
    weights = []
    for i in range(k):
        weights.append(20 - population[i].score)
    return weights


def copy_array(into: List[Program], of: List[Program], k: int) -> None:
    for i in range(k):
        into[i] = of[i]


def main() -> None:  # optional driver
    # array = (-1, 2, -3, 4)
    # array = (1, 3, 2, 2, 0, 1, 0) # works
    # array = [-1, 2, -3, -3]
    # array = [0, 0, 0, 0, 0, 0, 0, 0]
    array = [13]
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
