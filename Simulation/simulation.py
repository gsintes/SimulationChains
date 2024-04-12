"""Run the simulation of chains in single lane swimming."""

from abc import ABC, abstractmethod

import numpy as np

class Simulation(ABC):
    """Simulation of chains in single lane swimming."""
    def __init__(self,
                 saving_file: str,
                 nb_bacteria: int=1000,
                 nb_collisions: int=100) -> None:
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
    
    @abstractmethod
    def update_vel(self) -> None:
        """Uptade the velocities after """
        raise NotImplementedError

    @abstractmethod
    def sampler(self) -> None:
        """Sample the initial position, velocity, force and drag."""
        raise NotImplementedError
    
    @abstractmethod
    def tumble(self) -> None:
        """Tumbling of the bacteria."""
        raise NotImplementedError
    
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

    def update_drag(self) -> None:
        """Update the drag coefficients"""
        pass #TODO implement

    def get_chains(self) -> None:
        """Determine if a bacteria is in a chain."""
        positions = [self.position[:, self.step]]
        pass #TODO implement


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
        self.tumble()

    def process(self) -> None:
        self.sampler()
        for _ in range(self.collisions_nb):
            self.step_process()

        self.save()

    def save(self) -> None:
        """Save the simulation data."""
        pass #TODO implement

class TestSimu(Simulation):
    def __init__(self) -> None:
        super().__init__("", 3, 1)
    
    def tumble(self) -> None:
        pass
    def sampler(self) -> None:
        self.velocity[:, 0] = [1, -2, 3]
        self.position[:, 0] = [0, 1, 2]

    def update_vel(self) -> None:
        pass
if __name__ == "__main__":
    simu = TestSimu()
    simu.sampler()
    simu.process()
    print(simu.position)
    print(simu.time_collision)
    print(simu.velocity)