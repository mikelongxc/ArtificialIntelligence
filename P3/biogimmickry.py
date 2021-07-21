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


def crossover(p1: Program, p2: Program) -> List[Program]:
    selected = [p1, p2]

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
        max_len = 35

    sequence_str = ""
    valid_commands = "><+-"
    for _ in range(random.randint(0, max_len)):
        sequence_str += valid_commands[random.randint(0, 3)]

    return Program(sequence_str)


def generate_random(fe: FitnessEvaluator, max_len: int,\
                    k: int, population: List[Program]) -> str:
    for _ in range(k):
        # generate random program
        if max_len == 0:
            program = generate_random_program(max_len)
        else:
            program = Program(generate_random_loop(max_len))
        # score newly generated random program and stop if 0
        fitness_score = program.score_fitness(fe)
        if fitness_score == 0:
            return program.sequence

        population.append(program)

    return ""


def create_program(fe: FitnessEvaluator, max_len: int) -> str:
    """
    Return a program string no longer than max_len that, when interpreted,
    populates a memory array that exactly matches a target array.

    Use fe.evaluate(program) to get a program's fitness score (zero is best).
    """

    # mut_prob = {"<": 0.8, ">": 0.8, "+": 0.6, "-": 0.6, "[": 0.1, "]": 0.1}

    # new_population: List[Program] = []

    # k = 1000
    # N = 0.5        # N is top percentile for selection process

    converges = True
    gen_no = 0

    while 1:
        k = 1000 # k represents the initial population size
        gen_no = gen_no + 1
        print(gen_no)
        if gen_no == 100:
            converges = True
            gen_no = 0

        # generate initial random, score initial random, add to population
        if converges:
            converges = False
            population: List[Program] = []
            res = generate_random(fe, max_len, k, population)
            if res != "":
                # print("from RANDOM")
                return res

        new_population: List[Program] = []
        ct = [0]

        while ct[0] != k:
            weights = populate_weights(k, population)

            population.sort(key=lambda program: program.score)

            selected = random.choices(population, weights=weights, k=k//2)
            selected.sort(key=lambda program: program.score)

            if bad_average(selected):
                k = 0
                converges = True
                gen_no = False
                break

            res = select(new_population, selected, fe, k//2, ct)
            if res != "":
                return res

        for i in range(k):
            population[i] = new_population[i]
            

def select(new_population: List[Program], selected: List[Program],\
           fe: FitnessEvaluator, n: int, ct: List[int]) -> str:
    for i in range(0, n, 2):
        new_programs = crossover(selected[i], selected[i + 1])

        score1 = new_programs[0].score_fitness(fe)
        score2 = new_programs[1].score_fitness(fe)

        if score1 == 0:
            return new_programs[0].sequence
        elif score2 == 0:
            return new_programs[1].sequence

        # add the new programs to the next_pop list until full
        new_population.append(new_programs[0])
        new_population.append(new_programs[1])

        ct[0] = ct[0] + 2

    return ""


def bad_average(selected: List[Program]) -> bool:
    _sum = 0
    for i in range(len(selected)):
        _sum += selected[i].score
    if (_sum / len(selected)) > 15:
        return True
    return False



def populate_weights(k: int, population: List[Program]) -> List[int]:
    weights = []
    for i in range(k):
        weights.append(20 - population[i].score)
    weights.sort(reverse=True)
    return weights


def copy_array(into: List[Program], of: List[Program], k: int) -> None:
    for i in range(k):
        into[i] = of[i]


def generate_random_loop(max_len: int) -> str:
    """
    This function is the rdm generator for functions with loops

    Yes, this function was hacked together and is objectively "bad" code
    but I was able to find out the patterns associated with loops and
    separated them into four parts:
        1 the pointers that change between the counter and the
            loop-incremented-value.
        2 the total number of iterations for the counter (with the counter
            decrement/increment)
        3 the actual -/+ inside the loop
        4 the loop brackets

    these components make it so that randomization is a little less naive and
    each sequence generated is a valid loop
    """
    sequence = ""
    # 1: ptr handling
    _dir = random.randint(0, 1) # 0 if ><> 1 if x><
    _niter = random.randint(0, 1) # 0 if +++ or -
    _loopval = random.randint(0, 1)

    plusorminus = "+-"
    minusorplus = "-+"

    ptr_dirs = "><"
    dirs_ptrs = "<>"

    # 1. optional counter index position
    if _dir == 0:
        sequence = ">"

    # 2. total loop incrementer/decrementer
    num_n_iter = random.randint(0, max_len // 4)
    for _ in range(num_n_iter):
        sequence = sequence + plusorminus[_niter]

    # 4. opening brackets
    sequence = sequence + "["

    # 1. ptr that goes to adjust loop values from the counter index
    sequence = sequence + dirs_ptrs[_dir]

    # 3. what the loop is changing (+/-)
    num_loop_val = random.randint(0, max_len // 2)
    for _ in range(num_loop_val):
        sequence = sequence + plusorminus[_loopval]

    # 1. ptr changer
    sequence = sequence + ptr_dirs[_dir]

    # 2. loop decrement or incrementer (before closing bracket)
    sequence = sequence + minusorplus[_niter]

    # 4. closing bracket
    sequence = sequence + "]"

    return sequence[:max_len]


def main() -> None:  # optional driver

    # x = generate_random_loop(15)
    """for i in range(30):
        x = generate_random_loop(20)
        print(x)"""
    array = (0, 20)
    max_len = 15

    program = create_program(FitnessEvaluator(array), max_len)
    print(program)
    print(FitnessEvaluator.interpret(program, len(array)))


if __name__ == "__main__":
    main()
