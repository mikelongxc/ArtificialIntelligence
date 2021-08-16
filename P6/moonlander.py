# Name:         Michael Long
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Moonlander II
# Term:         Summer 2021

import random
from typing import Callable, Tuple


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


class QTable:

    def __init__(self):
        self.tab = [[1, 3, 4], [9, 8, 1]]

        pass

    def get(self, x: Tuple[ModuleState, int]) -> float:

        return self.tab[0][0]


class Moonlander:

    def __init__(self, state: ModuleState):
        self.state = state
        self.alpha = 0.5
        self.epsilon = 0.001
        self.gamma = 0.5        # learning rate

    def learn_q(self) -> None: #Callable[[ModuleState, int], float]:
        """
        use state.set_actions first

        :return: a q function

        self.state.
            fuel, altitude, velocity, actions, use_fuel
        """

        for _ in range(1000):

            pass


def learn_q(state: ModuleState) -> Callable[[ModuleState, int], float]:
    """
    Return a Q-function that maps a state-action pair to a utility value. This
    function must be a callable with the signature (ModuleState, int) -> float.

    Optional: Use |state.set_actions| to set the size of the action set. Higher
    values offer more control (sensitivity to differences in rate changes), but
    require larger Q-tables and thus more training time.

    :return: a Q function callable

    """

    state.set_actions(5)
    a1 = state.use_fuel(0)
    b = state.use_fuel(1)
    c = state.use_fuel(4)
    d = state.use_fuel(8)

    #return None

    # q = Moonlander(state)
    # return q.learn_q()

    table = QTable()

    return lambda s, a: table.get((s, a))


def main() -> None:

    fuel: int = 100
    altitude: float = 50.0

    gforces = {"Pluto": 0.063, "Moon": 0.1657, "Mars": 0.378, "Venus": 0.905,
               "Earth": 1.0, "Jupiter": 2.528}

    transition = lambda g, r: g * (2 * r - 1)  # example transition function

    #   #   #

    state = ModuleState(fuel, altitude, gforces["Moon"], transition)
    q = learn_q(state)
    policy = lambda s: max(state.actions, key=lambda a: q(s, a))

    print(state)
    while state.altitude > 0:
        state = state.use_fuel(policy(state))
        print(state)


if __name__ == "__main__":
    main()
