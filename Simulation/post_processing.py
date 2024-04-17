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
        self.width_vis_window = 200

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
        step_appear: List[int] = []
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
                            step_appear.append(step_appear[ind])
                        else:
                            id.append(count)
                            step_appear.append(step)
                            count += 1
                    except ValueError:
                        id.append(count)
                        step_appear.append(step)
                        count += 1

        self.chains = pd.DataFrame({
            "id": id,
            "step": steps,
            "step_appear": step_appear,
            "time": time,
            "chain_length": length,
            "vel": vel,
            "one": one,
            "position": position
        })
        self.chains.to_csv(os.path.join(self.fig_folder, "data.csv"))

    def plot_vel(self) -> None:
        """Plot the velocity with chain length."""
        data = pp.chains.drop_duplicates("id")
        plt.figure()
        sns.pointplot(data=pp.chains, x="chain_length", y="vel", hue="step_appear", linestyle="", native_scale=True, errorbar=None)
        plt.savefig(os.path.join(self.fig_folder, "vel_chainLengthError.png"))
        plt.close()

        plt.figure()
        sns.scatterplot(data=data, x="chain_length", y="vel", hue="step_appear", linestyle="")
        plt.savefig(os.path.join(self.fig_folder, "vel_chainLengthScatter.png"))
        plt.close()

    def plot_min(self) -> None:
        """Plot the minimal velocity for each chain length"""
        lengths = self.chains.chain_length.unique()
        mins_vel = np.zeros(len(lengths))
        for i, length in enumerate(lengths):
            mins_vel[i] = self.chains[self.chains.chain_length==length].vel.min()
        plt.figure()
        plt.xlabel("Chain length")
        plt.ylabel("Minimal velocity")
        plt.plot(lengths, mins_vel, linestyle="", marker="o")
        plt.savefig(os.path.join(self.fig_folder, "min.png"))

    def plot_max(self) -> None:
        """Plot the minimal velocity for each chain length"""
        lengths = self.chains.chain_length.unique()
        maxs_vel = np.zeros(len(lengths))
        for i, length in enumerate(lengths):
            maxs_vel[i] = self.chains[self.chains.chain_length==length].vel.max()

        plt.figure()
        plt.xlabel("Chain length")
        plt.ylabel("Maximal velocity")
        plt.plot(lengths, maxs_vel, linestyle="", marker="o")
        plt.savefig(os.path.join(self.fig_folder, "max.png"))

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
            plt.xlim((-5, self.width_vis_window + 5))
            sub_data = self.chains[self.chains.step == i]
            lengthes = sub_data.chain_length.unique()
            lengthes.sort()
            for length in lengthes:
                sub_sub = sub_data[sub_data.chain_length==length]
                plt.plot(sub_sub.vis_pos, sub_sub.one, linestyle="", color="b", markersize=5 * length, marker=".", label=length)

            plt.legend(loc="center left", title="Chain length", bbox_to_anchor=(1.04, 0.5))
            plt.title(f"Collision number : {i}")
            plt.savefig(os.path.join(vis_folder, f"{i}.png"), bbox_inches="tight")
            plt.close()

    def process(self, visualisation: bool=False) -> None:
        """Run the post processing."""
        self.get_chains()
        self.plot_vel()
        self.plot_min()
        self.plot_max()
        if visualisation:
            self.visualisation()


if __name__=="""__main__""":
    
    # simu_max = sim.SimulationMax()
    # pp = PostProcessing(simu_max, "/Users/sintes/Desktop/NASGuillaume/SimulationChains/Max")
    # pp.process()

    # simu_df = sim.SimuSampleDragForce()
    # pp = PostProcessing(simu_df, "/Users/sintes/Desktop/NASGuillaume/SimulationChains/AverageDragForce")
    # pp.process()

    # simu_v = sim.SimuSampleSpeed()
    # pp = PostProcessing(simu_v, "/Users/sintes/Desktop/NASGuillaume/SimulationChains/AverageSpeed")
    # pp.process()


    data = pd.read_csv("/Users/sintes/Desktop/NASGuillaume/Chains/chain_data.csv")
    simu_d = sim.SimuSampleData(data, 1000, 500)
    pp = PostProcessing(simu_d, "/Users/sintes/Desktop/NASGuillaume/SimulationChains/FromData")
    pp.process(visualisation=True)
