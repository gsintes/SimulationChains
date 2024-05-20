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

class PostProcessing:
    def __init__(self, fig_folder: str) -> None:

        self.chains = pd.read_csv(os.path.join(fig_folder, "data.csv"), usecols=["chain_length", "time", "step", "id", "Simu_nb", "vel", "step_appear"])

        self.chains["chain_length"] = pd.to_numeric(self.chains["chain_length"], downcast="unsigned")
        self.chains["step"] = pd.to_numeric(self.chains["step"], downcast="unsigned")
        self.chains["id"] = pd.to_numeric(self.chains["id"], downcast="unsigned")
        self.chains["Simu_nb"] = pd.to_numeric(self.chains["id"], downcast="unsigned")
        self.chains["step_appear"] = pd.to_numeric(self.chains["step_appear"], downcast="unsigned")
        self.chains["vel"] = pd.to_numeric(self.chains["vel"], downcast="float")

        self.chains = self.chains[self.chains["step"]<=500]

        self.lengths = self.chains.chain_length.unique()
        self.fig_folder = fig_folder

    @staticmethod
    def create_folders(folder) -> None:
        """Create and empty folder for visualisation"""
        try:
            os.makedirs(folder)
        except FileExistsError:
            shutil.rmtree(folder)
            os.makedirs(folder)

    def plot_vel(self) -> None:
        """Plot the velocity with chain length."""
        data = self.chains.drop_duplicates(("Simu_nb", "id"))
        plt.figure()
        sns.pointplot(data=data, x="chain_length", y="vel", linestyle="", native_scale=True, errorbar=None, label="Simulation")
        sns.pointplot(data=DATA, x="chain_length", y="velocity", linestyle="", native_scale=True, errorbar=None, label="Experiment")
        plt.savefig(os.path.join(self.fig_folder, "vel_chainLengthError.png"))
        plt.close()

        plt.figure()
        sns.scatterplot(data=data, x="chain_length", y="vel", hue="step_appear", linestyle="")
        plt.savefig(os.path.join(self.fig_folder, "vel_chainLengthScatter.png"))
        plt.close()

    def plot_by_chain_length(self) -> None:
        """Plot the extremal velocity and velocity histograms for each chain length"""
        PostProcessing.create_folders(os.path.join(self.fig_folder,"Velocity_histograms"))
        data = pp.chains.drop_duplicates(("Simu_nb", "id"))
        mins_vel = np.zeros(len(self.lengths))
        maxs_vel = np.zeros(len(self.lengths))

        for i, length in enumerate(self.lengths):
            sub_data = self.data[self.data.chain_length==length]
            maxs_vel[i] = sub_data.vel.max()
            mins_vel[i] = sub_data.vel.min()

            self.histogram_velocity(sub_data, length)

        plt.figure()
        plt.xlabel("Chain length")
        plt.ylabel("Minimal velocity")
        plt.plot(self.lengths, mins_vel, linestyle="", marker="o")
        plt.savefig(os.path.join(self.fig_folder, "min.png"))

        plt.figure()
        plt.xlabel("Chain length")
        plt.ylabel("Maximal velocity")
        plt.plot(self.lengths, maxs_vel, linestyle="", marker="o")
        plt.savefig(os.path.join(self.fig_folder, "max.png"))

    def count_chain_length(self) -> None:
        """Plot the proportion per chain length with time."""
        data = self.chains.groupby(["step", "chain_length"]).count()
        data = data.reset_index()
        data = data.rename(columns={"id": "count"})
        data = data.pivot(index="step", columns="chain_length", values="count")
        data = data.fillna(0)
        data = data.div(data.sum(axis=1), axis=0)

        plt.figure()
        sns.lineplot(data=data, dashes=False, palette="bright")
        plt.savefig(os.path.join(self.fig_folder, "chain_length_evolution.png"))
        plt.show(block=True)

    def plot_histogram_chain_length(self) -> None:
        """Plot the histogram of chain length."""
        data = pp.chains.drop_duplicates(("Simu_nb", "id"))
        plt.figure()
        sns.histplot(data=data, x="chain_length", stat="density", discrete=True)
        plt.savefig(os.path.join(self.fig_folder, "length_hist.png"))
        plt.close()

    def histogram_velocity(self, data: pd.DataFrame, n: int) -> None:
        """Make the histogram of velocity."""
        vis_folder = os.path.join(self.fig_folder,"Velocity_histograms")
        plt.figure()
        sns.histplot(data=data, x="vel")
        plt.savefig(os.path.join(vis_folder, f"hist_vel_length_{n}.png"))
        plt.close()

    def chain_length_distrib_evolution(self) -> None:
        """"Plot the evolution of the chain length distribution."""
        vis_folder = os.path.join(self.fig_folder,"Length_histograms")
        PostProcessing.create_folders(vis_folder)

        for i in range(self.chains.step.max()):
            sub_data = self.chains[self.chains.step==i]
            plt.figure()
            max_chain_length = self.lengths.max()
            plt.xlim((0.9, max_chain_length + 1))
            plt.ylim((0, 1))
            sns.histplot(data=sub_data, x="chain_length", stat="density", discrete=True)
            time = sub_data.time.unique()[0]
            plt.title(f"Collision number : {i}, time: {time:.2f}s")
            plt.savefig(os.path.join(vis_folder, f"{i}.png"))
            plt.close()

    def mean_speed_evolution(self) -> None:
        """Make a visualisation of speed evolution."""
        vis_folder = os.path.join(self.fig_folder,"Speed evolution")
        PostProcessing.create_folders(vis_folder)

        max_vel = self.chains.groupby(by=["step", "chain_length"]).vel.mean().max()
        min_vel = self.chains.vel.min()
        max_chain_length = self.lengths.max()
        for i in range(self.chains.step.max()):
            sub_data = self.chains[self.chains.step==i]
            plt.figure()
            plt.ylim((min_vel - 1, max_vel + 1))
            plt.xlim((0.9, max_chain_length + 1))
            sns.pointplot(data=sub_data, x="chain_length", y="vel", linestyle="", native_scale=True, errorbar=None)
            time = sub_data.time.unique()[0]
            plt.title(f"Collision number : {i}, time: {time:.2f}s")
            plt.savefig(os.path.join(vis_folder, f"{i}.png"))
            plt.close()

    def vel_vs_time(self) -> None:
        """Plot the velocity for the different length with time."""
        plt.figure()
        sns.pointplot(data=self.chains,
                      x="step",
                      y="vel", hue="chain_length", marker=".", linestyle="", native_scale=True, errorbar=None,
                      palette="bright")
        plt.savefig(os.path.join(self.fig_folder, "velocityvsstep.png"))

    def time_vs_step(self) -> None:
        """Plot the time with the number of collision."""
        plt.figure()
        plt.plot(self.chains.step, self.chains.time, ".")
        plt.xlabel("Number of collisions")
        plt.ylabel("Time")
        plt.savefig(os.path.join(self.fig_folder, "time_step.png"))

        times = self.chains.groupby("step").time.mean()
        plt.figure()
        plt.plot(times.index, times.diff(), ".")
        plt.xlabel("Number of collisions")
        plt.ylabel("Time between encounter")
        plt.savefig(os.path.join(self.fig_folder, "time_encounter_step.png"))

    def process(self, visualisation: bool=False) -> None:
        """Run the post processing."""
        # if visualisation:
        #     self.chain_length_distrib_evolution()
            # self.mean_speed_evolution()
        self.count_chain_length()
        # self.vel_vs_time()
        # self.time_vs_step()
        # self.plot_vel()
        # self.plot_by_chain_length()
        # self.plot_histogram_chain_length()



if __name__=="""__main__""":

    DATA = pd.read_csv("/home/guillaume/NAS/Chains/chain_data.csv")
    DATA = DATA[DATA.chain_length <= 8]

    # simu_max = sim.SimulationMax()
    # pp = PostProcessing(simu_max, "/Users/sintes/Desktop/NASGuillaume/SimulationChains/Max")
    # pp.process()

    # pp = PostProcessing("/Volumes/Guillaume/SimulationChains/AverageDragForce")
    # pp.process()

    # simu_v = sim.SimuSampleSpeed()
    # pp = PostProcessing(simu_v, "/Users/sintes/Desktop/NASGuillaume/SimulationChains/AverageSpeed")
    # pp.process()


    # pp = PostProcessing(simu_d, "/Users/sintes/Desktop/NASGuillaume/SimulationChains/FromDataEvenSpacing")
    # pp.process(visualisation=True)

    # pp = PostProcessing("/Volumes/Guillaume/SimulationChains/DragUpdate")
    # pp.process(False)

    pp = PostProcessing( "/home/guillaume/NAS/SimulationChains/FromDataRandomSpacing")
    pp.process(visualisation=False)
