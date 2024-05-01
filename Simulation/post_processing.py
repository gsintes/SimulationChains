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
    def __init__(self, fig_folder: str) -> None:
        self.chains = pd.read_csv(os.path.join(fig_folder, "data.csv"))
        self.fig_folder = fig_folder
        self.width_vis_window = 200

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

    def plot_histogram_chain_length(self) -> None:
        """Plot the histogram of chain length."""
        data = pp.chains.drop_duplicates(("Simu_nb", "id"))
        plt.figure()
        sns.histplot(data=data, x="chain_length", stat="density", discrete=True)
        plt.savefig(os.path.join(self.fig_folder, "length_hist.png"))
        plt.close()

    def histogram_velocity(self) -> None:
        """Make the histogram of velocity by chain length."""
        vis_folder = os.path.join(self.fig_folder,"Velocity_histograms")
        try:
            os.makedirs(vis_folder)
        except FileExistsError:
            shutil.rmtree(vis_folder)
            os.makedirs(vis_folder)

        data = pp.chains.drop_duplicates(("Simu_nb", "id"))
        lengthes = data.chain_length.unique()
        for l in lengthes:
            sub_data = data[data.chain_length==l]
            plt.figure()
            sns.histplot(data=sub_data, x="vel")
            plt.savefig(os.path.join(vis_folder, f"hist_vel_length_{l}.png"))
            plt.close()

    def chain_length_distrib_evolution(self) -> None:
        """"Plot the evolution of the chain length distribution."""
        vis_folder = os.path.join(self.fig_folder,"Length_histograms")
        try:
            os.makedirs(vis_folder)
        except FileExistsError:
            shutil.rmtree(vis_folder)
            os.makedirs(vis_folder)

        for i in range(self.chains.step.max()):
            sub_data = self.chains[self.chains.step==i]
            plt.figure()
            max_chain_length = self.chains.chain_length.max()
            plt.xlim((0.9, max_chain_length + 1))
            plt.ylim((0, 1))
            sns.histplot(data=sub_data, x="chain_length", stat="density", discrete=True)
            time = sub_data.time.unique()[0]
            plt.title(f"Collision number : {i}, time: {time:.2f}s")
            plt.savefig(os.path.join(vis_folder, f"{i}.png"))
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
        y_min = self.chains.one.min() - 1
        y_max = self.chains.one.max() + 1
        for i in range(self.chains.step.max()):
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
            time = sub_data.time.unique()[0]
            plt.title(f"Collision number : {i}, time: {time:.2f}s")
            plt.savefig(os.path.join(vis_folder, f"{i}.png"), bbox_inches="tight")
            plt.close()

    def mean_speed_evolution(self) -> None:
        """Make a visualisation of speed evolution."""
        vis_folder = os.path.join(self.fig_folder,"Speed evolution")
        try:
            os.makedirs(vis_folder)
        except FileExistsError:
            shutil.rmtree(vis_folder)
            os.makedirs(vis_folder)
        max_vel = self.chains.groupby(by=["step", "chain_length"]).vel.mean().max()
        min_vel = self.chains.vel.min()
        max_chain_length = self.chains.chain_length.max()
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
        
    def process(self, visualisation: bool=False) -> None:
        """Run the post processing."""
        if visualisation:
            if len(self.chains.Simu_nb.unique())==1:
                self.visualisation()
            self.chain_length_distrib_evolution()
            self.mean_speed_evolution()
        self.plot_vel()
        self.plot_min()
        self.plot_max()
        self.plot_histogram_chain_length()
        self.histogram_velocity()
        


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


    # data = pd.read_csv("/Users/sintes/Desktop/NASGuillaume/Chains/chain_data.csv")
    # simu_d = sim.SimuSampleData(data, 1000, 500, initial_size=10000)
    # pp = PostProcessing(simu_d, "/Users/sintes/Desktop/NASGuillaume/SimulationChains/FromDataEvenSpacing")
    # pp.process(visualisation=True)

    DATA = pd.read_csv("/Users/sintes/Desktop/NASGuillaume/Chains/chain_data.csv")
    DATA = DATA[DATA.chain_length <= 8]
    pp = PostProcessing( "/Users/sintes/Desktop/NASGuillaume/SimulationChains/FromDataRandomSpacing")
    pp.process(visualisation=True)