"""Run the simulation of chains in single lane swimming."""

from abc import ABC, abstractmethod
from math import isclose

import numpy as np
from utils import timeit

class Simulation(ABC):
    """Simulation of chains in single lane swimming."""
    def __init__(self,
                 nb_bacteria: int=2000,
                 nb_collisions: int=500) -> None:
        self.bacteria_nb= nb_bacteria
        self.collisions_nb = nb_collisions

        self.drag = np.zeros((self.bacteria_nb, self.collisions_nb + 1))
        self.force = np.zeros((self.bacteria_nb, self.collisions_nb + 1))
        self.velocity = np.zeros((self.bacteria_nb, self.collisions_nb + 1))
        self.position = np.zeros((self.bacteria_nb, self.collisions_nb + 1))

        self.relative_vel = np.zeros((self.bacteria_nb, self.bacteria_nb))
        self.relative_pos = np.zeros((self.bacteria_nb, self.bacteria_nb))
        self.time_collision = np.zeros((self.collisions_nb + 1))
        self.step = 0

        self.chains = np.array([np.identity(self.bacteria_nb) for _ in range(self.collisions_nb +1)])
    
    @abstractmethod
    def update_vel(self) -> None:
        """Update the velocities after collisions"""
        raise NotImplementedError
    
    @abstractmethod
    def update_drag(self) -> None:
        """Update the drag after collisions"""
        raise NotImplementedError

    @abstractmethod
    def sampler(self) -> None:
        """Sample the initial position, velocity, force and drag."""
        raise NotImplementedError
    
    @abstractmethod
    def tumble(self) -> None:
        """Tumbling of the bacteria."""
        raise NotImplementedError
    
    def update_force(self) -> None:
        """Update the force."""
        self.force[:, self.step] = self.velocity[:, self.step] / self.drag[:, self.step]

    def relative_velocities(self) -> None:
        """Calculate the relative velocities of the different bacteria."""
        velocities = np.array([self.velocity[:, self.step - 1]])
        u = np.ones((1, self.bacteria_nb))
        self.relative_vel = velocities.T @ u - u.T @ velocities

    def relative_positions(self) -> None:
        """Calculate the relative velocities of the different bacteria."""
        pos = np.array([self.position[:, self.step - 1]])
        u = np.ones((1, self.bacteria_nb))
        self.relative_pos = u.T @ pos - pos.T @ u

    def time_next_collision(self) -> None:
        """Calculate the time of next collision."""
        time_collisions = np.divide(self.relative_pos,
                                    self.relative_vel,
                                    out=np.full(self.relative_vel.shape, np.inf),
                                    where=self.relative_vel!=0)
        time_collisions[time_collisions <= 0] = np.inf
        self.time_collision[self.step] = np.min(time_collisions)

    def update_position(self) -> None:
        """Update the position just before the merger"""
        self.position[:, self.step] = self.position[:, self.step - 1] +\
            self.velocity[:, self.step - 1] * self.time_collision[self.step]

    def get_chains(self) -> None:
        """Determine if a bacteria is in a chain."""
        positions = self.position[:, self.step]
        for i, pos_i in enumerate(positions):
            for j, pos_j in enumerate(positions):
                if isclose(pos_i, pos_j):
                    self.chains[self.step, i, j] = 1

    def step_process(self) -> None:
        """Process one step of simulation simulation"""
        self.step += 1
        self.relative_velocities()
        self.relative_positions()
        self.time_next_collision()
        self.update_position()
        self.get_chains()
        self.update_vel()
        self.update_drag()
        self.update_force()
        self.tumble()

    @timeit
    def process(self) -> None:
        self.sampler()
        self.get_chains()
        for _ in range(self.collisions_nb):
            self.step_process()


class SimulationMax(Simulation):
    """Simulation with an update of the velocity keeping the max."""
    def tumble(self) -> None:
        pass

    def update_drag(self) -> None:
        self.drag[:, self.step] = self.drag[:, self.step -1]

    def update_vel(self) -> None:
        chains = self.chains[self.step, :, :]
        drag = self.drag[:, self.step - 1]
        forces_in_chains = self.velocity[:, self.step - 1] * drag * chains
        abs_vel = np.max(np.abs(self.velocity[:, self.step - 1] * chains), axis=1)
        sign = np.sign(forces_in_chains.sum(axis=1))
        self.velocity[:, self.step] = sign * abs_vel

    def sampler(self) -> None:
        abs_vel = np.random.lognormal(size=self.bacteria_nb)
        sign = np.random.choice([-1, 1], size=abs_vel.shape)
        self.velocity[:, 0] = sign * abs_vel

        self.position[:, 0] = 1000 * np.arange(self.bacteria_nb)
        self.drag[:, 0] = [10 for _ in range(self.bacteria_nb)]
        self.update_force()

class SimuNoDragUpdate(Simulation):
    def __init__(self) -> None:
        super().__init__()
    
    def tumble(self) -> None:
        pass

    @abstractmethod
    def sampler(self) -> None:
        raise NotImplementedError

    def update_drag(self) -> None:
        self.drag[:, self.step] = self.drag[:, self.step -1]

    def update_vel(self) -> None:
        chains = self.chains[self.step, :, :]
        drag = self.drag[:, self.step - 1]
        forces_in_chains = self.velocity[:, self.step - 1] * drag * chains
        abs_vel = np.abs(forces_in_chains).sum(axis=1) / (drag * chains).sum(axis=1)
        sign = np.sign(forces_in_chains.sum(axis=1))
        self.velocity[:, self.step] = sign * abs_vel


class SimuSampleSpeed(SimuNoDragUpdate):
    """Not tumbling. Sampling velocity. No drag update."""
    def __init__(self) -> None:
        super().__init__()

    def sampler(self) -> None:
        abs_vel = np.random.lognormal(size=self.bacteria_nb)
        sign = np.random.choice([-1, 1], size=abs_vel.shape)
        self.velocity[:, 0] = sign * abs_vel
        self.position[:, 0] = 1000 * np.arange(self.bacteria_nb)
        self.drag[:, 0] = [10 for _ in range(self.bacteria_nb)]
        self.update_force()


class SimuSampleDragForce(SimuNoDragUpdate):
    """Not tumbling. Sampling force and drag. No drag update."""

    def sampler(self) -> None:
        self.position[:, 0] = 1000 * np.arange(self.bacteria_nb)
        drag = np.random.normal(loc=10, size=self.bacteria_nb)
        drag[drag <= 0] = 0.2
        self.drag[:, 0] = drag
        self.force[:, 0] = np.random.normal(size=self.bacteria_nb)
        self.velocity[:, 0] = self.force[:, 0] / self.drag[:, 0]

if __name__ == "__main__":
    simu = SimulationMax()
    simu.sampler()
    simu.process()

    print(simu.velocity)
    print(np.abs(simu.velocity[:, 0]).mean())