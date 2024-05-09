"""Theoretical analysis of the aggreagation."""

### Case with 3 bacteria, one at the left l, one at the right r, this two have purely random velocity. One at the center with fixed velocity norm v_c and random orientation s_c.


D0 = 10

import os
from typing import Callable, Iterable, Dict

import numpy as np
from scipy.stats import lognorm
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib as mp

def distrib_norm_vel(v: float) -> float:
    """Give the probability density of the norm of v."""
    if v < 0:
        return 0
    return lognorm.pdf(v, 1)

class Probability_knowing_vc:
    """Calculate the probabilities at v_c fixed."""
    def __init__(self, distrib_norm_vel: Callable[[float], float], v_c:float) -> None:
        assert v_c >= 0
        self.v_c = v_c
        self.distrib_norm_vel = distrib_norm_vel
        self.infinity_encounter_probability()

    def cumulative_norm_vel(self, v: float) -> float:
        """Give the probability that the velocity is smaller than v."""
        return quad(self.distrib_norm_vel, 0, v)[0]

    def distrib_vel_rel_knowingSc(self, v_hat: float, s_c: int) -> float:
        """Calculate the distribution of the relative velocity knowing Sc."""
        assert s_c in (-1, 1)
        if s_c == 1:
            return (distrib_norm_vel(v_hat + self.v_c) + distrib_norm_vel(-(v_hat + self.v_c))) / 2
        elif s_c == -1:
            return self.distrib_vel_rel_knowingSc(-v_hat, 1)
        
    def distrib_vel_rel(self, v_hat: float) -> float:
        """Calculate the distribution of the relative velocity."""
        return (self.distrib_vel_rel_knowingSc(v_hat, 1)  + self.distrib_vel_rel_knowingSc(v_hat, -1)) / 2

    def proba_interval_vel_rel_knowingSc(self, a: float, b: float, s_c: int) -> float:
        """Calculate the cumulative distribution of the relative velocity knowing Sc."""
        assert s_c in (-1, 1)
        return quad(lambda x: self.distrib_vel_rel_knowingSc(x, s_c), a, b)[0]

    def infinity_encounter_probability(self) -> None:
        """Probability to have an encounter at infinity."""
        p_v_inf_v_c = self.cumulative_norm_vel(self.v_c)
        self.pinf = (1 - p_v_inf_v_c ** 2) / 4
    
    def cumulative_time_encounter(self, t: float) -> float:
        """Calculate the culmulative distribution of the time of encounter."""

        bigger_than = ((self.proba_interval_vel_rel_knowingSc(-1000, D0 / t, 1)) * (self.proba_interval_vel_rel_knowingSc(- D0 / t, 1000, 1)) + 
                        self.proba_interval_vel_rel_knowingSc(-1000, D0 / t, -1) * self.proba_interval_vel_rel_knowingSc(- D0 / t, 1000, -1)) / 2 
        return  1 - bigger_than
    
    def get_cdf_dict_time_encounter(self, values: Iterable[float]) -> None:
        """Make a dict with the cdf of the time encounter."""
        self.cdf: Dict[float, float] = {}
        for val in values:
            self.cdf[val] = self.cumulative_time_encounter(val)

    def distribution_time_encounter(self) -> float:
        """Calculate the distribution of the tine of encounter."""
        values = list(self.cdf.keys())
        values.sort()
        self.pdf: Dict[float, float] = {}
        for i, val in enumerate(values):
            if i < len(values) - 1:
                self.pdf[val] = (self.cdf[values[i + 1]] - self.cdf[val]) / (values[i + 1] - val)
            if i == len(values) - 1:
                self.pdf[val] = (self.cdf[val] - self.cdf[values[i - 1]]) / (val - values[i - 1])

    def get_mode_time_encouter(self) -> float:
        """Get the most probable time of encounter."""
        return max(self.pdf, key=self.pdf.get)

if __name__=="__main__":
    save_folder = "/Users/sintes/Library/CloudStorage/OneDrive-Personal/These/ProbabilityEncounter"
    v_cs = [0.1, 0.5, 1, 3, 5]
    t = np.linspace(0.1, 50, 1000)
    v_rel = np.linspace(-30, 30, 200)
    fig1 = plt.figure()
    ax1 = fig1.gca()
    fig2 = plt.figure()
    ax2 = fig2.gca()
    fig3 = plt.figure()
    ax3 = fig3.gca()
    cmap = mp.colormaps['viridis']
    modes = []
    for i, v_c in enumerate(v_cs):
        proba = Probability_knowing_vc(distrib_norm_vel, v_c)       
        proba.get_cdf_dict_time_encounter(t)
        proba.distribution_time_encounter()
        proba_vel_rel = list(map(lambda x: proba.distrib_vel_rel_knowingSc(x, 1), v_rel))
        
        ax1.plot(t, proba.cdf.values(), "-", color=cmap(v_c / max(v_cs)), label=v_c)
        ax2.plot(t, proba.pdf.values(), "-", color=cmap(v_c / max(v_cs)), label=v_c)
        if i % 2 ==0:
            ax3.plot(v_rel, proba_vel_rel, "-", color=cmap(v_c / max(v_cs)), label=v_c)
        modes.append(proba.get_mode_time_encouter())

    ax1.set_xlabel("$t$")
    ax1.set_ylabel("$p(T_e<t)$")

    ax2.set_xlabel("$t$")
    ax2.set_ylabel("$p(T_e=t)$")
    fig1.savefig(os.path.join(save_folder, "cumulTe.png"))
    ax1.legend(title="$v_c$")
    ax2.legend(title="$v_c$")
    fig2.savefig(os.path.join(save_folder, "pdfTe.png"))

    ax3.set_xlabel("$\hat{v}$")
    ax3.set_ylabel("$p(\hat{V}=\hat{v}|S_c=1)$")
    ax3.legend(title="$v_c$")
    fig3.savefig(os.path.join(save_folder, "pdfVrel.png"))

    plt.figure()
    plt.plot(v_cs, modes, "ob")
    plt.xlabel("V_c")
    plt.ylabel("$mode(T_e)$")
    plt.savefig(os.path.join(save_folder, "modeTe.png"))
    
    
    plt.figure()
    pdf =list(map(distrib_norm_vel, t))
    plt.plot(t, pdf)
    plt.xlabel("$v$")
    plt.ylabel("$p(V_{l,r}=v)$")
    plt.savefig(os.path.join(save_folder, "pdfVlr.png"))