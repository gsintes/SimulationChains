"""Run the simulation of chains in single lane swimming."""

import os
from typing import List
from abc import ABC, abstractmethod
from math import isclose
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


class Simulation(ABC):
    """Simulation of chains in single lane swimming."""
    def __init__(self,
                saving_folder: str,
                nb_bacteria: int=1000,
                nb_collisions: int=200,
                nb_simu: int = 1,
                position_random: bool = False,
                initial_size: int = 10000)-> None:
        self.saving_folder = saving_folder
        self.bacteria_nb= nb_bacteria
        self.nb_simu = nb_simu
        self.collisions_nb = nb_collisions
        self.initial_size = initial_size
        self.position_random = position_random

    def initialize(self) -> None:
        """Initialize the simulation."""
        self.drag = np.zeros((self.bacteria_nb, self.collisions_nb + 1))
        self.force = np.zeros((self.bacteria_nb, self.collisions_nb + 1))
        self.velocity = np.zeros((self.bacteria_nb, self.collisions_nb + 1))
        self.position = np.zeros((self.bacteria_nb, self.collisions_nb + 1))

        self.relative_vel = np.zeros((self.bacteria_nb, self.bacteria_nb))
        self.relative_pos = np.zeros((self.bacteria_nb, self.bacteria_nb))
        self.time_collision = np.zeros((self.collisions_nb + 1))
        self.step = 0

        self.concentration = self.bacteria_nb / self.initial_size

        self.chains = np.array([np.identity(self.bacteria_nb) for _ in range(self.collisions_nb +1)])
        if self.position_random:
            self.position[:, 0] = self.initial_size * np.random.random(size=self.bacteria_nb)
        else:
            self.position[:, 0] = self.initial_size  * np.arange(self.bacteria_nb) / self.bacteria_nb

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
        self.update_drag()
        self.update_vel()
        self.update_force()
        self.tumble()

    def process(self) -> None:
        self.initialize()
        self.sampler()
        self.get_chains()
        for _ in range(self.collisions_nb):
            self.step_process()

    def run_simu(self) -> None:
        """Run the simulation."""
        self.process()
        chains_simu = self.get_chains_data()
        chains_simu["Simu_nb"] = 0
        self.chains_data = chains_simu
        for j in range(1, self.nb_simu):
            self.process()
            chains_simu = self.get_chains_data()
            chains_simu["Simu_nb"] = j
            self.chains_data = pd.concat((self.chains_data, chains_simu), ignore_index=True)
        self.chains_data.to_csv(os.path.join(self.saving_folder, "data.csv"), index=False)

    def get_chains_data(self) -> pd.DataFrame:
        """Get the chains at each time step into a dataFrame."""
        count: int = 0
        id: List[int] = []
        bact_nb: List[int] = []
        steps: List[int] = []
        time: List[float] = []
        length: List[int] = []
        vel: List[float] = []
        position: List[float] = []
        step_appear: List[int] = []
        for step in range(self.collisions_nb):
            chains = self.chains[step, :, :]
            for i in range(self.bacteria_nb):
                if sum(chains[i, :i]) == 0:
                    steps.append(step)
                    time.append(sum(self.time_collision[:step + 1]))
                    bact_nb.append(i)
                    l = chains[i, :].sum()
                    length.append(int(l))
                    vel.append(abs(self.velocity[i, step]))
                    pos = self.position[i, step]
                    position.append(pos)
                    try:
                        ind = id.index(i)
                        if l == length[ind]:
                            id.append(id[ind])
                            step_appear.append(step_appear[ind])
                        else:
                            id.append(count)
                            step_appear.append(step)
                            count += 1
                    except ValueError:
                        id.append(count)
                        step_appear.append(step)
                        count += 1

        chains_data = pd.DataFrame({
            "id": id,
            "step": steps,
            "step_appear": step_appear,
            "time": time,
            "chain_length": length,
            "vel": vel,
            "position": position
        })
        return chains_data
    
    def sample_bact(self) -> List[int]:
        """Sample random bacteria for visualisation."""
        initial_vel = np.abs(self.velocity[:, 0])
        val = [0, 0.25, 0.5, 0.75, 1]
        quantiles = list(map(lambda x: np.quantile(initial_vel, x), val))
        ids = []
        for i, q in enumerate(quantiles):
            if i !=0:
                temp_indexes = []
                for j in range(self.bacteria_nb):
                    if quantiles[i - 1] < initial_vel[j] <= q:
                        temp_indexes.append(j)
                ids += random.sample(temp_indexes, 1)
        return ids

    def visualisation_speed_evolution_bact(self) -> None:
        """Visualize the evolution of a bacteria"""
        ids = self.sample_bact()
        vel_tot = np.abs(self.velocity[ids, :])
        steps = list(range(self.collisions_nb + 1))
        chain_length_tot = self.chains[:, :, ids].sum(axis=1)
        fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
        for id in ids:
            vel = np.abs(self.velocity[id, :])
            chain_length = self.chains[:, :, id].sum(axis=1)
            points = np.array([steps, vel]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            cmap = ListedColormap(plt.get_cmap("jet")([i / (chain_length_tot.max() + 1) for i in range(int(chain_length_tot.max()) + 1)]))
            norm = BoundaryNorm([0.5 + i for i in range(int(chain_length_tot.max()) + 1)], cmap.N)
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(chain_length)
            lc.set_linewidth(1)
            line = axs.add_collection(lc)

        fig.colorbar(line, ax=axs)
        axs.set_xlim(0, self.collisions_nb + 1)
        axs.set_ylim(vel_tot.min() - 1, vel_tot.max() + 1)
        plt.ylabel("Velocity")
        plt.xlabel("Number of collisions")        
        plt.show(block=True)


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
        self.drag[:, 0] = [10 for _ in range(self.bacteria_nb)]
        self.update_force()

class SimuNoDragUpdate(Simulation):
    
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

    def sampler(self) -> None:
        abs_vel = np.random.lognormal(size=self.bacteria_nb)
        sign = np.random.choice([-1, 1], size=abs_vel.shape)
        self.velocity[:, 0] = sign * abs_vel
        
        self.drag[:, 0] = [10 for _ in range(self.bacteria_nb)]
        self.update_force()

class SimuSampleData(SimuNoDragUpdate):
    """Not tumbling. No drag update. Sampling velocity from actual distribution."""
    def __init__(self, 
                 saving_folder: str,
                 data: pd.DataFrame,
                 nb_bacteria: int=1000,
                 nb_collisions: int=100,
                 nb_simu: int = 1,
                 position_random: bool=False,
                 initial_size: int= 10000) -> None:
        super().__init__(saving_folder=saving_folder,
                        nb_bacteria=nb_bacteria,
                        nb_collisions=nb_collisions,
                        nb_simu=nb_simu,
                        position_random=position_random,
                        initial_size=initial_size)
        self.data = data

    def sample_vel(self) -> float:
        x = np.random.uniform(0, self.sample_size - 1)
        i = int(np.floor(x))
        p = x - i
        vel = (1 - p) * self.vel[i] + p * self.vel[i + 1]
        return vel
    
    def sampler(self) -> None:
        data1 = self.data[self.data.chain_length==1]
        _vel = data1.velocity.dropna()
        self.vel = np.array(_vel)
        self.vel.sort()
        self.sample_size = len(_vel)

        abs_vel = np.array([self.sample_vel() for _ in range(self.bacteria_nb)])
        sign = np.random.choice([-1, 1], size=abs_vel.shape)
        self.velocity[:, 0] = sign * abs_vel
        self.drag[:, 0] = [10 for _ in range(self.bacteria_nb)]
        self.update_force()        


class SimuSampleDragForce(SimuNoDragUpdate):
    """Not tumbling. Sampling force and drag. No drag update."""

    def sampler(self) -> None:
        drag = np.random.normal(loc=10, size=self.bacteria_nb)
        drag[drag <= 0] = 0.2
        self.drag[:, 0] = drag
        self.force[:, 0] = np.random.normal(size=self.bacteria_nb)
        self.velocity[:, 0] = self.force[:, 0] / self.drag[:, 0]

class SimuDragUptade(Simulation):
    """Update the drag considering that there is only one flagella length in the chain."""    
    
    drag_body = 13e-3
    drag_flagella = 9.1e-3

    def __init__(self, 
                 saving_folder: str,
                 data: pd.DataFrame,
                 nb_bacteria: int=1000,
                 nb_collisions: int=100,
                 nb_simu: int = 1,
                 position_random: bool=False,
                 initial_size: int= 10000) -> None:
        super().__init__(saving_folder=saving_folder,
                        nb_bacteria=nb_bacteria,
                        nb_collisions=nb_collisions,
                        nb_simu=nb_simu,
                        position_random=position_random,
                        initial_size=initial_size)
        self.data = data
    
    def tumble(self) -> None:
        pass

    @abstractmethod
    def sampler(self) -> None:
        raise NotImplementedError

    def update_drag(self) -> None:
        chain_length = self.chains[self.step, :, :].sum(axis=1)
        self.drag[:, self.step] = SimuDragUptade.drag_body + SimuDragUptade.drag_flagella / chain_length

    def update_vel(self) -> None:
        chains = self.chains[self.step, :, :]
        drag = self.drag[:, self.step - 1]
        forces_in_chains = self.velocity[:, self.step - 1] * drag * chains
        abs_vel = np.abs(forces_in_chains).sum(axis=1) / (drag * chains).sum(axis=1)
        sign = np.sign(forces_in_chains.sum(axis=1))
        self.velocity[:, self.step] = sign * abs_vel

    def sample_vel(self) -> float:
        x = np.random.uniform(0, self.sample_size - 1)
        i = int(np.floor(x))
        p = x - i
        vel = (1 - p) * self.vel[i] + p * self.vel[i + 1]
        return vel
    
    def sampler(self) -> None:
        data1 = self.data[self.data.chain_length==1]
        _vel = data1.velocity.dropna()
        self.vel = np.array(_vel)
        self.vel.sort()
        self.sample_size = len(_vel)

        abs_vel = np.array([self.sample_vel() for _ in range(self.bacteria_nb)])
        sign = np.random.choice([-1, 1], size=abs_vel.shape)
        self.velocity[:, 0] = sign * abs_vel
        self.drag[:, 0] = [SimuDragUptade.drag_body + SimuDragUptade.drag_flagella for _ in range(self.bacteria_nb)]
        self.update_force()    


if __name__ == "__main__":
    data = pd.read_csv("/Volumes/Guillaume//Chains/chain_data.csv")
    folder = "/Users/sintes/Desktop/FromDataRandomSpacing"
    simu = SimuSampleData(data=data, saving_folder=folder, nb_bacteria=1000, nb_collisions=500, position_random=True, nb_simu=2)
    simu.run_simu()
    # simu.visualisation_speed_evolution_bact()