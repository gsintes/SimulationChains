"""Make a visualisation of the simulation"""
import os
import shutil

import pandas as pd
import matplotlib.pyplot as plt

import time

def get_nb_rows_first_simu(file: str)-> int:
    """Get the number of rows associated with the first simulation."""
    count = 0
    with open(file, "r") as f:
        sim_nb = 1
        for line in f:
            try:
                sim_nb = int(line.split(",")[-1])
                if sim_nb == 1:
                    break
                count +=1
            except ValueError:
                pass
    return count


class Visualisation:
    """Visualisation of a simulation"""

    def __init__(self, folder: str) -> None:
        self.fig_folder = folder
        self.width_vis_window = 200
        csv_path = os.path.join(self.fig_folder, "data.csv")
        line_nb = get_nb_rows_first_simu(csv_path)
        self.chains = pd.read_csv(csv_path, nrows=line_nb)
        self.chains["one"] = self.chains.position // self.width_vis_window

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

if __name__=="__main__":
    folder = "/Volumes/Guillaume/SimulationChains/DragUpdate"
    vis = Visualisation(folder)
    vis.visualisation()