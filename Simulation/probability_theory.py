"""Theoretical analysis of the aggreagation."""

### Case with 3 bacteria, one at the left l, one at the right r, this two have purely random velocity. One at the center with fixed velocity norm v_c and random orientation s_c.


D0 = 10

from typing import Callable, Iterable

import numpy as np
from scipy.stats import lognorm
from scipy.integrate import quad
from scipy.misc import derivative
import matplotlib.pyplot as plt
from matplotlib import cm

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
            return (distrib_norm_vel(self.v_c - v_hat) + distrib_norm_vel(v_hat - self.v_c)) / 2
        
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
        cdf = {}
        for val in values:
            cdf[val] = self.cumulative_time_encounter(val)

    def distribution_time_encounter(self) -> float:
        """Calculate the distribution of the tine of encounter."""
        return derivative(lambda x: self.cumulative_time_encounter(x), t, dx=0.1)
    

if __name__=="__main__":
    v_cs = [0.1, 1, 3, 5]
    t = np.linspace(0.1, 50, 100)
    fig1 = plt.figure()
    ax1 = fig1.gca()
    fig2 = plt.figure()
    ax2 = fig2.gca()
    cmap = cm.get_cmap("viridis", len(v_cs))
    for i, v_c in enumerate(v_cs):
        proba = Probability_knowing_vc(distrib_norm_vel, v_c)       
        cdf = list(map(proba.cumulative_time_encounter, t))
        # pdf = list(map(proba.distribution_time_encounter, t))
        # pdf2 = list(map(lambda x: proba.distrib_vel_rel_knowingSc(x, -1), t))
        
        ax1.plot(t, cdf, "-", color=cmap[i], label=v_c)
        # ax2.plot(t, pdf, "-", color=cmap[i], label=v_c)

    # ax1.xlabel("$T_e=t$")
    # plt.ylabel("$p(T_e<t)$")
    # plt.legend(title="$v_c$")
    plt.show(block=True)
    