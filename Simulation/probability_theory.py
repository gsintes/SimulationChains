"""Theoretical analysis of the aggreagation."""

### Case with 3 bacteria, one at the left l, one at the right r, this two have purely random velocity. One at the center with fixed velocity norm v_c and random orientation s_c.


D0 = 10

from typing import Callable

import numpy as np
from scipy.stats import lognorm
from scipy.integrate import quad
from scipy.misc import derivative
import matplotlib.pyplot as plt

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
            return distrib_norm_vel(v_hat + self.v_c) / 2 + distrib_norm_vel(-(v_hat + self.v_c)) / 2
        else:
            return distrib_norm_vel(self.v_c - v_hat) / 2 + distrib_norm_vel(v_hat - self.v_c) / 2
        
    def distrib_vel_rel(self, v_hat: float) -> float:
        """Calculate the distribution of the relative velocity."""
        return (self.distrib_vel_rel_knowingSc(v_hat, 1)  + self.distrib_vel_rel_knowingSc(v_hat, -1)) / 2

    def vel_rel_cumulative_knowingSc(self, v_hat: float, s_c: int) -> float:
        """Calculate the cumulative distribution of the relative velocity knowing Sc."""
        assert s_c in (-1, 1)
        return quad(lambda x: self.distrib_vel_rel(x), 0, v_hat)[0]

    def infinity_encounter_probability(self) -> None:
        """Probability to have an encounter at infinity."""
        p_v_inf_v_c = self.cumulative_norm_vel(self.v_c)
        self.pinf = (1 - p_v_inf_v_c ** 2) / 4
    
    def cumulative_time_encounter(self, t: float) -> float:
        """Calculate the culmulative distribution of the time of encounter."""

        bigger_than = ((self.vel_rel_cumulative_knowingSc(D0 / t, 1)) ** 2  + 
                        (self.vel_rel_cumulative_knowingSc(D0 / t, -1)) **  2) / 2 
        return  1 - self.pinf - bigger_than

    def distribution_time_encounter(self, t: float) -> float:
        """Calculate the distribution of the tine of encounter."""
        return derivative(lambda x: self.cumulative_time_encounter(x), t, dx=1)
    
    @staticmethod
    def esperance(distribution: Callable[[float], float], min_val: float=0, max_val: float = 100) -> float:
        """Calculate the esperance of the distribution."""
        return quad(lambda x: x * distribution(x), min_val, max_val)


if __name__=="__main__":
    v_cs = [0.1, 1, 2, 10]
    t = np.linspace(0.1, 20, 200)
    plt.figure()
    for v_c in v_cs:
        proba = Probability_knowing_vc(distrib_norm_vel, v_c)       
        pdf = list(map(proba.cumulative_time_encounter, t))
        
        plt.plot(t, pdf, label=v_c)
    plt.xlabel("$T_e$")
    plt.ylabel("$p(T_e)$")
    plt.legend()
    plt.show(block=True)
    # print(pdf[-1] + infinity_encounter_probability(v_c=1))