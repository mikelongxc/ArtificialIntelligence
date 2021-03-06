# Name:         Michael Long
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Moonlander II
# Term:         Summer 2021

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

    def __eq__(self, other) -> bool:
        return self.hash_value == other.hash_value

    def __repr__(self):
        rp = (bin_velocity(self.velocity), \
              bin_altitude(self.altitude), self.fuel)
        return str(rp)

    def __hash__(self):
        return self.hash_value

    def update_hash(self, hashed: int) -> None:
        self.hash_value = hashed

    def _float_eq(self, o1: float, o2: float) -> bool:
        epsilon = 0.001

        if abs(o1 - o2) < epsilon:
            return True

        return False


class QTable:

    def __init__(self, actions: Tuple[int, ...]):
        self.table = []
        self.state_dict = {}
        self.actions = actions
        self.num_states = 1000

    def get(self, sa_pair: Tuple[ModuleState, int]) -> float:

        s = sa_pair[0]
        a = sa_pair[1]

        qs = QState(s)

        idx = self.try_add_state_to_table(qs)

        return self.table[idx][a]

    def update(self, sa_pair: Tuple[QState, int], u: float):

        s = sa_pair[0]
        a = sa_pair[1]

        # either adds state or just return indices
        lookup_idx = self.try_add_state_to_table(s)

        self.table[lookup_idx][a] = u

    def s_in_dict(self, s: QState) -> bool:
        hashed = get_table_hash(s.velocity, s.altitude)
        s.update_hash(hashed)

        if s in self.state_dict:
            return True
        return False

    def try_add_state_to_table(self, s: QState) -> int:
        """

        :param s:
        :return:  INDEX GIVEN BY DICT
        """
        s_already_exists = False
        idx = None

        if self.s_in_dict(s):
            s_already_exists = True
            idx = self.state_dict.get(s)

        if not s_already_exists:
            actions = s.actions
            empty_util = []

            # INIT Q
            for _ in range(len(actions)):
                empty_util.append(0)

            self.table.append(empty_util)

            last_idx = len(self.table) - 1

            self.state_dict.update({s: last_idx})

            idx = last_idx

        return idx

    def register_new_state(self, s: QState):

        table_index = 0
        if s not in self.state_dict:
            self.state_dict.update({s: table_index})

    def lookup_index(self, s: QState) -> int:
        # TODO mmm
        # if newly discovered, add to table

        idx = self.try_add_state_to_table(s)

        # TODO: add known util if terminal to q-table

        return idx

    def print_table(self):
        for x in self.table:
            print(*x, sep=' ')


def terminal_state_util(velocity: float, altitude: float) -> int:
    if altitude <= 0.0001 and velocity > -1:
        return 1
    elif altitude <= 0.0001 and velocity < -1:
        return -1

    return 0


def successful_bin_altitude(base_altitude: float) -> float:
    altitude = base_altitude
    # round to int if bigger than 5
    if base_altitude >= 100:
        altitude = 100
    elif base_altitude > 25:
        altitude = int(5 * round(base_altitude / 5))
    elif base_altitude > 10:
        altitude = int(2 * round(base_altitude / 2))
    elif base_altitude > 5:
        altitude = int(base_altitude)
    elif base_altitude > 0:
        altitude = round(base_altitude, 1)

    return altitude


def bin_altitude(base_altitude: float) -> float:
    altitude = base_altitude

    if base_altitude > 2.0:
        altitude = int(altitude)
    elif base_altitude < 2.0:
        altitude = round(base_altitude * 2) / 2

    return altitude


def bin_velocity(base_velocity: float) -> float:
    velocity = base_velocity

    if velocity > -4:
        velocity = round(base_velocity * 2) / 2
    else:
        velocity = int(base_velocity)

    return velocity


def get_table_hash(base_velocity: float, base_altitude: float) -> int:

    altitude = bin_altitude(base_altitude)
    velocity = bin_velocity(base_velocity)

    vel_alt_tup = (velocity, altitude)

    return hash(vel_alt_tup)


class Moonlander:

    def __init__(self, state: ModuleState):
        self.state = state
        self.alpha = 0.85           # learning rate
        self.alpha_decay = 0.999
        self.epsilon = 0.001
        self.gamma = 0.9        # discounting factor
        self.default_reward_value = -0.01

        self.max_altitude = state.altitude

        self.q_table = QTable(state.actions)

    def learn_q(self) -> Callable[[ModuleState, int], float]:
        """
        use state.set_actions first

        :return: a q function

        self.state.
            fuel, altitude, velocity, actions, use_fuel
        """

        original = QState(self.state)
        s = QState(self.state)

        # learning iteration:
        for _ in range(10000):

            # for each action in state:
            for i in range(len(s.actions)):
                sa_pair = (s, s.actions[i])
                # get q (u) val
                self.update_q_value(sa_pair)

            saved_idx = [0]

            self.max_util_of_state(s, saved_idx)

            s = QState(s.state.use_fuel(saved_idx[0]))

            if s.altitude == 0:
                s = original

            self.epsilon -= 0.01

        # q func to return. basically just returning a table
        return lambda st, ac: self.q_table.get((st, ac))

    def update_q_value(self, sa_pair: Tuple[QState, int]):
        """

        :return: Q(s, a) <-
        """
        alpha = self.alpha
        gamma = self.gamma
        s = sa_pair[0]
        a = sa_pair[1]

        next_s = QState(s.state.use_fuel(a))

        # # # EQUATION
        p1 = (1 - alpha) * self.q_table.get(sa_pair)
        _reward_s = self.reward(s)

        if terminal_state_util(next_s.velocity, next_s.altitude) == 1:
            _reward_s = 1
        elif terminal_state_util(next_s.velocity, next_s.altitude) == -1:
            _reward_s = -1

        _max_util_of_successor = self.max_util_of_state(next_s, [-1])
        p2 = alpha * (_reward_s + (gamma * _max_util_of_successor))
        u = p1 + p2

        # update in table
        self.q_table.update(sa_pair, u)

        return u

    def reward(self, s: QState) -> float:

        if not s.state.altitude and s.state.velocity > -1:
            return 1
        elif not s.state.altitude and s.state.velocity <= -1:
            return -1

        # PENALTY: if height is above max
        if s.state.altitude > self.max_altitude:
            return -0.05

        return self.default_reward_value

    def max_util_of_state(self, next_s: QState, saved_idx: List[int]) -> float:

        if saved_idx[0] == -1:
            util = terminal_state_util(next_s.velocity, next_s.altitude)
            if util != 0:
                return util

        # find where next_s row is in q_table (lookup)
        row = self.q_table.lookup_index(next_s)

        rdm_start = 0

        # check each
        cur_max_real = self.q_table.table[row][rdm_start]
        cur_max = round(cur_max_real, 10)

        saved_idx[0] = rdm_start
        for i in range(len(self.q_table.table[row])):
            next_val_real = self.q_table.table[row][i]
            if round(next_val_real, 10) > cur_max:
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
    # print("simulating LM")
    state.set_actions(5)

    q = Moonlander(state)
    return q.learn_q()


def main() -> None:

    x = 0
    g = "Earth"
    trials = 1

    if x == 0:
        tests(print_all=False, print_fail=True, trials=trials)
    else:
        test(1000, 50.0, True, False, g)


def policy(state: ModuleState, \
           q: Callable[[ModuleState, int], float]) -> int:

    fz = lambda a: q(state, a)
    val = max(state.actions, key=fz)
    return val


def tests(print_all: bool, print_fail: bool, trials: int) -> None:
    # (altitude)
    # fa: List[int] = [10, 25, 50, 75, 100]
    fa: List[int] = [50, 75, 100]
    # fa = [10, 25]
    # fa = [75]
    fuel: int = 1000
    g_forces = {"Pluto": 0.063, "Moon": 0.1657, "Mars": 0.378, "Venus": 0.905,
                "Earth": 1.0, "Jupiter": 2.528}

    for altitude in fa:
        ct = 0
        print()
        print("# T E S T I N G      A L T I T U D E : " + str(altitude) + "m")

        for gs in g_forces:
            ct = 0
            for _ in range(trials):
                ct += test(fuel, altitude, print_all, print_fail, gs)
            print("         [" + gs + "]    success count:   "\
                  + str(ct) + " / " + str(trials))


def test(fuel: int, altitude: float,\
         print_all: bool, print_fail: bool, g: str) -> int:

    g_forces = {"Pluto": 0.063, "Moon": 0.1657, "Mars": 0.378, "Venus": 0.905,
                "Earth": 1.0, "Jupiter": 2.528}
    transition = lambda g, r: g * (2 * r - 1)  # example transition function

    state = ModuleState(fuel, altitude, g_forces[g], transition)
    q = learn_q(state)
    policy = lambda s: max(state.actions, key=lambda a: q(s, a))

    hist = ""

    # print(state)
    # print("beginning individual test")
    while state.altitude > 0:
        # state = state.use_fuel(policy(state, q))
        state = state.use_fuel(policy(state))
        # TODO DEBUG COMMENT # BELOW
        # hist += str(state) + "\n"
        if print_all:
            print(state)

    hist += str(state) + "\n"
    if state.velocity > -1:
        return 1
    else:
        if print_fail:
            print(hist)
        return 0


if __name__ == "__main__":
    main()
