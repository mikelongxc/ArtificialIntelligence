# Name:         Michael Long
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Moonlander II
# Term:         Summer 2021

import random
from typing import Callable, Tuple, List


class ModuleState:  # do not modify class

    def __init__(self, fuel: int, altitude: float, force: float,
                 transition: Callable[[float, float], float],
                 velocity: float = 0.0,
                 actions: Tuple[int, ...] = tuple(range(5))) -> None:
        """
        An instance of ModuleState has the following attributes.

            fuel: The amount of fuel (in liters) able to be used.
            altitude: The distance (in meters) of the module from the surface
                      of its target object.
            velocity: The speed of the module, where a positive value indicates
                      movement away from the target object and a negative value
                      indicates movement toward it. Defaults to zero.
            actions: The available fuel rates, where 0 indicates free-fall and
                     the highest-valued action indicates maximum thrust away
                     from the target object. Defaults to (0, 1, 2, 3, 4).
            use_fuel: A callable that takes an integer as its only argument to
                      be used as the fuel rate for moving to the next state.
        """
        self.fuel: int = fuel
        self.altitude: float = altitude
        self.velocity: float = velocity
        self.actions: Tuple[int, ...] = actions
        self.use_fuel: Callable[[int], ModuleState] = \
            lambda rate: self._use_fuel(force, transition, rate)

    def __repr__(self) -> str:
        if not self.altitude:
            return ("-" * 16 + "\n"
                    + f" Remaining Fuel: {self.fuel:4} l\n"
                    + f"Impact Velocity: {self.velocity:7.2f} m/s\n")
        else:
            return (f"    Fuel: {self.fuel:4} l\n"
                    + f"Altitude: {self.altitude:7.2f} m\n"
                    + f"Velocity: {self.velocity:7.2f} m/s\n")

    def set_actions(self, n: int) -> None:
        """
        Set the number of actions available to the module simulator, which must
        be at least two. Calling this method overrides the default number of
        actions set in the constructor.

        >>> module.set_actions(8)
        >>> module.actions
        (0, 1, 2, 3, 4, 5, 6, 7)
        """
        if n < 2:
            raise ValueError
        self.actions = tuple(range(n))

    def _use_fuel(self, force: float, transition: Callable[[float, int], float],
                  rate: int) -> "ModuleState":
        """
        Return a ModuleState instance in which the fuel, altitude, and velocity
        are updated based on the fuel rate chosen.

        Do not call this method directly; instead, call the |use_fuel| instance
        attribute, which only requires a fuel rate as its argument.
        """
        if not self.altitude:
            return self
        fuel = max(0, self.fuel - rate)
        if not fuel:
            rate = 0
        acceleration = transition(force * 9.8, rate / (len(self.actions) - 1))
        altitude = max(0.0, self.altitude + self.velocity + acceleration / 2)
        velocity = self.velocity + acceleration
        return ModuleState(fuel, altitude, force, transition, velocity=velocity,
                           actions=self.actions)


class QState:

    def __init__(self, state: ModuleState):
        """
        serves as a wrapper class for ModuleState.

        for hash, eq in QTable

        """
        self.state = state

        self.altitude = state.altitude
        self.velocity = state.velocity
        self.fuel = state.fuel

        self.actions = state.actions

        self.hash_value = 0

        # TODO: place bin args here??? based on given state nah,.

    def __eq__(self, other) -> bool:
        # TODO: binning
        # TODO: repr by fuel???
        is_equal = 0

        if self._float_eq(self.altitude, other.altitude):
            is_equal += 1
        if self._float_eq(self.velocity, other.velocity):
            is_equal += 1
        if self._float_eq(self.fuel, other.fuel):
            is_equal += 1

        return True if is_equal == 3 else False

    def __hash__(self):
        if self.hash_value != 0:
            return self.hash_value
        else: # TODO change
            print("ERROR ERROR ERROR ERROR ERROR")
            exit(1)

    def update_hash(self, hashed: int) -> None:
        self.hash_value = hashed

    def _float_eq(self, o1: float, o2: float) -> bool:
        epsilon = 0.001

        if abs(o1 - o2) < epsilon:
            return True

        return False


class QTable:

    def __init__(self, actions: Tuple[int, ...]):
        self.table = [[]]
        self.state_dict = {}
        self.actions = actions
        self.num_states = 1000

        self._init_table()

    def get(self, sa_pair: Tuple[QState, int]) -> float:
        # TODO: retrieval of bin
        # TODO: NOTE NOTE NOTE (s, a) IS NOT THE SAME AS (alt, vel)
        s = sa_pair[0]
        a = sa_pair[1]
        altitude = s.altitude
        velocity = s.velocity
        hashed = get_table_hash(velocity, altitude)

        s.update_hash(hashed)

        u = self.state_dict.get(s)

        if not u and u != 0:
            return 0

        return u

    def update(self, sa_pair: Tuple[QState, int], u: float):
        """

        :param s: state to find (row)
        :param a: action to find (col)

        :param u: utility given to update (s,a) in table

        :return:
        """

        # use dict to register index-row of state

        # bin velocities by 5
        # bin altitudes by 5

        # just get a hash based on the velocity and altitude

        s = sa_pair[0]
        a = sa_pair[1]

        velocity = s.velocity
        altitude = s.altitude

        # get hash value, update hash
        hashed = get_table_hash(velocity, altitude)
        s.update_hash(hashed)

        # add to dict with table index

        # TODO: how to determine place in table itself?
        # TODO: at what point is a state added and indexed?

        # if not in dict, store in dict
        if s not in self.state_dict:
            self.state_dict.update({s: 0})

        lookup_idx = self.state_dict.get(s)
        self.table[lookup_idx][a] = u


        pass

    def register_new_state(self, s: QState):

        table_index = 0
        if s not in self.state_dict:
            self.state_dict.update({s: table_index})


    def _init_table(self):
        for i in range(self.num_states):
            self.table.append([])
            for j in range(len(self.actions)):
                self.table[i].append(0)

    def lookup_index(self, s: QState) -> int:
        # TODO mmm
        # if newly discovered, add to table
        if s not in self.state_dict:
            self.update()
        else:
            idx = self.state_dict.get(s)

        return idx

    def splitter(self):
        alt_ps = [0.05, 0]


def get_table_hash(velocity: float, altitude: float) -> int:

    # before putting in tuple, bin velocity based on values

    vel_alt_tup = (int(velocity), int(altitude))

    return hash(vel_alt_tup)


class Moonlander:

    def __init__(self, state: ModuleState):
        self.state = state
        self.alpha = 0.5
        self.epsilon = 0.001
        self.gamma = 0.5        # learning rate
        self.default_reward_value = -0.04

        self.max_altitude = state.altitude

        self.q_table = QTable(state.actions)

    def learn_q(self) -> Callable[[ModuleState, int], float]:
        """
        use state.set_actions first

        :return: a q function

        self.state.
            fuel, altitude, velocity, actions, use_fuel
        """

        # init q table. init all values with 0
        # q_table = QTable() # ALREADY INITIALIZED

        # state wrapper for binning by alt and vel
        s = QState(self.state)

        # learning iteration:
        # for each update of a qstate, update neighboring states based on cur
        for _ in range(10000):

            # for each action in state: # TODO ???? how to choose action?
            for i in range(len(s.actions)):
                sa_pair = (s, s.actions[i])

                # get q (u) val
                self.update_q_value(sa_pair)

                #


                # TODO: decay epsilon exponentially. start at 1

            # TODO: need some way to change the state

            # iterate over util list

            saved_idx = [0]
            u = self.max_util_of_state(s, saved_idx)

            s = QState(s.state.use_fuel(saved_idx[0]))


        #

        #

        # q func to return. basically just returning a table
        return lambda s, a: self.q_table.get((s, a))

    def update_q_value(self, sa_pair: Tuple[QState, int]):
        """

        :return: Q(s, a) <-
        """
        alpha = self.alpha
        epsilon = self.epsilon
        gamma = self.gamma
        s = sa_pair[0]
        a = sa_pair[1]

        next_s = QState(s.state.use_fuel(a))
        actions = next_s.state.actions

        # first part of eqn
        p1 = (1 - alpha) * self.q_table.get(sa_pair)

        # second part of eqn
        _reward_s = self.reward(s)
        _max_util_of_successor = self.max_util_of_state(next_s, [0])

        p2 = alpha * (_reward_s + (gamma * _max_util_of_successor))

        # calc both together
        u = p1 + p2

        # update in table
        self.q_table.update(sa_pair, u)

        return u

    def reward(self, s: QState) -> float:
        if not s.state.altitude and s.state.velocity > -1:
            return 1
        elif not s.state.altitude and s.state.velocity <= -1:
            return -1

        # .001% (epsilon) chance
        if random.random() < self.epsilon:
            return -1 * self.default_reward_value

        return self.default_reward_value

    def max_util_of_state(self, next_s: QState, saved_idx: List[int]) -> float:

        # find where next_s row is in q_table (lookup)
        row = self.q_table.lookup_index(next_s)

        # check each
        cur_max = self.q_table.table[row][0]
        saved_idx[0] = 0
        for i in range(len(self.q_table.table[row])):
            if self.q_table.table[row][i] < cur_max:
                cur_max = self.q_table.table[row][i]
                saved_idx[0] = i

        return cur_max


def learn_q(state: ModuleState) -> Callable[[ModuleState, int], float]:
    """
    Return a Q-function that maps a state-action pair to a utility value. This
    function must be a callable with the signature (ModuleState, int) -> float.

    Optional: Use |state.set_actions| to set the size of the action set. Higher
    values offer more control (sensitivity to differences in rate changes), but
    require larger Q-tables and thus more training time.

    :return: a Q function callable

    """

    q = Moonlander(state)
    return q.learn_q()


def main() -> None:

    fuel: int = 100
    altitude: float = 50.0

    g_forces = {"Pluto": 0.063, "Moon": 0.1657, "Mars": 0.378, "Venus": 0.905,
               "Earth": 1.0, "Jupiter": 2.528}

    transition = lambda g, r: g * (2 * r - 1)  # example transition function

    #   #   #

    state = ModuleState(fuel, altitude, g_forces["Moon"], transition)
    q = learn_q(state)
    policy = lambda s: max(state.actions, key=lambda a: q(s, a))

    print(state)
    while state.altitude > 0:
        state = state.use_fuel(policy(state))
        print(state)


if __name__ == "__main__":
    main()
