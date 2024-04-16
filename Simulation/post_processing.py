"""Post-processing the simulation."""

import warnings
warnings.filterwarnings("ignore")

from typing import List
import os
import shutil

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import simulation as sim

class PostProcessing:
    def __init__(self, simulation: sim.Simulation, fig_folder: str) -> None:
        self.simu = simulation
        self.simu.process()
        self.fig_folder = fig_folder
        self.width_vis_window = 2000

    def get_chains(self) -> None:
        """Get the chains at each time step."""
        count: int = 0
        id: List[int] = []
        bact_nb: List[int] = []
        steps: List[int] = []
        time: List[float] = []
        length: List[int] = []
        vel: List[float] = []
        position: List[float] = []
        one: List[int] = []
        for step in range(self.simu.collisions_nb):
            chains = self.simu.chains[step, :, :]
            for i in range(self.simu.bacteria_nb):
                if sum(chains[i, :i]) == 0:
                    steps.append(step)
                    time.append(sum(self.simu.time_collision[:step + 1]))
                    bact_nb.append(i)
                    l = chains[i, :].sum()
                    length.append(int(l))
                    vel.append(abs(self.simu.velocity[i, step]))
                    pos = self.simu.position[i, step]
                    position.append(pos)
                    one.append(pos // self.width_vis_window)
                    try:
                        ind = id.index(i)
                        if l == length[ind]:
                            id.append(id[ind])
                        else:
                            id.append(count)
                            count += 1
                    except ValueError:
                        id.append(count)
                        count += 1

        self.chains = pd.DataFrame({
            "id": id,
            "step": steps,
            "time": time,
            "chain_length": length,
            "vel": vel,
            "one": one,
            "position": position
        })
        self.chains.to_csv(os.path.join(self.fig_folder, "data.csv"))

    def plot(self) -> None:
        """Plot the velocity with chain length."""
        plt.figure()
        sns.pointplot(data=pp.chains, x="chain_length", y="vel", linestyle="")
        plt.savefig(os.path.join(self.fig_folder, "vel_chainLengthError.png"))
        plt.close()

        plt.figure()
        sns.scatterplot(data=pp.chains, x="chain_length", y="vel", linestyle="")
        plt.savefig(os.path.join(self.fig_folder, "vel_chainLengthScatter.png"))
        plt.close()

    def visualisation(self) -> None:
        """Make a visualisation of the simulation."""
        vis_folder = os.path.join(self.fig_folder,"Visualisation")
        try:
            os.makedirs(vis_folder)
        except FileExistsError:
            shutil.rmtree(vis_folder)
            os.makedirs(vis_folder)
        self.chains["vis_pos"] = self.chains.position % self.width_vis_window
        y_min = self.chains.one.min() + 0.1
        y_max = self.chains.one.max() + 0.1
        for i in range(self.simu.collisions_nb):
            plt.figure(figsize=(10, 8))
            plt.ylim((y_min, y_max))
            sub_data = self.chains[self.chains.step == i]
            sns.scatterplot(data=sub_data, x="vis_pos", y="one", linestyle="", hue="chain_length", marker=".", palette=sns.color_palette("plasma"))
            plt.title(f"Collision number : {i}")
            plt.savefig(os.path.join(vis_folder, f"{i}.png"))
            plt.close()

    def process(self) -> None:
        """Run the post processing."""
        pp.get_chains()
        pp.plot()
        pp.visualisation()


if __name__=="""__main__""":
    
    simu_max = sim.SimulationMax()
    pp = PostProcessing(simu_max, "/Users/sintes/Desktop/NASGuillaume/SimulationChains/Max")
    pp.process()

    simu_df = sim.SimuSampleDragForce()
    pp = PostProcessing(simu_df, "/Users/sintes/Desktop/NASGuillaume/SimulationChains/AverageDragForce")
    pp.process()

    simu_v = sim.SimuSampleSpeed()
    pp = PostProcessing(simu_v, "/Users/sintes/Desktop/NASGuillaume/SimulationChains/AverageSpeed")
    pp.process()

    simu_v = sim.SimuSampleSpeed()
    pp = PostProcessing(simu_v, "/Users/sintes/Desktop/NASGuillaume/SimulationChains/AverageSpeed")
    pp.process()

    data = pd.read_csv("/Users/sintes/Desktop/NASGuillaume/Chains/chain_data.csv")
    simu_d = sim.SimuSampleData(data=data)
    pp = PostProcessing(simu_d, "/Users/sintes/Desktop/NASGuillaume/SimulationChains/FromData")
    pp.process()