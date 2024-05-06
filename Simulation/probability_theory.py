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

def cumulative_norm_vel(v: float) -> float:
    """Give the probability that the velocity is smaller than v."""
    return quad(distrib_norm_vel, 0, v)[0]

    
def distrib_vel_rel_knowingSc(v_hat: float, v_c: float, s_c: int) -> float:
    """Calculate the distribution of the relative velocity knowing Sc."""
    assert s_c in (-1, 1)
    if s_c == 1:
        return distrib_norm_vel(v_hat + v_c) / 2 + distrib_norm_vel(-(v_hat + v_c)) / 2
    else:
        return distrib_norm_vel(v_c - v_hat) / 2 + distrib_norm_vel(v_hat - v_c) / 2
    
def distrib_vel_rel(v_hat: float, v_c: float) -> float:
    """Calculate the distribution of the relative velocity."""
    return (distrib_vel_rel_knowingSc(v_hat, v_c, 1)  + distrib_vel_rel_knowingSc(v_hat, v_c, -1)) / 2


def vel_rel_cumulative_knowingSc(v_hat: float, v_c: float, s_c: int) -> float:
    """Calculate the cumulative distribution of the relative velocity knowing Sc."""
    assert s_c in (-1, 1)
    return quad(lambda x: distrib_vel_rel(x, v_c), 0, v_hat)[0]

def infinity_encounter_probability(v_c: float) -> float:
    """Probability to have an encounter at infinity."""
    p_v_inf_v_c = cumulative_norm_vel(v_c)
    return (1 - p_v_inf_v_c ** 2) / 4
   

def culmulative_time_encounter(t: float, v_c: float=1) -> float:
    """Calculate the culmulative distribution of the time of encounter."""

    bigger_than = ((vel_rel_cumulative_knowingSc(D0 / t, v_c, 1)) ** 2  + 
                    (vel_rel_cumulative_knowingSc(D0 / t, v_c, -1)) **  2) / 2 
    pinf = infinity_encounter_probability(v_c)
    return  1 - pinf - bigger_than

def distribution_time_encouter(t: float, v_c: float) -> float:
    """Calculate the distribution of the tine of encounter."""
    return derivative(lambda x: culmulative_time_encounter(x, v_c), t, dx=1)

def esperance(distribution: Callable[[float], float], min_val: float=0, max_val: float = 100) -> float:
    """Calculate the esperance of the distribution."""
    return quad(lambda x: x * distribution(x), min_val, max_val)

if __name__=="__main__":
    t = np.linspace(0., 20, 100)
    pdf = list(map(infinity_encounter_probability, t))
    plt.figure()
    plt.plot(t, pdf)
    plt.xlabel("$v_c$")
    plt.ylabel("$p(T_e=+\\infty)$")
    plt.show(block=True)
    # print(pdf[-1] + infinity_encounter_probability(v_c=1))