"""Post-processing the simulation."""

import pandas as pd
import simulation as sim

class PostProcessing:
    def __init__(self, simulation: sim.Simulation) -> None:
        self.simu = simulation
        self.simu.process()

    def get_chains(self) -> None:
        """Get the chains at each time step."""
        count = 0
        id = []
        bact_nb = []
        steps = []
        time = []
        length = []
        vel = []

        for step in range(self.simu.collisions_nb):
            chains = self.simu.chains[step, :, :]
            for i in range(self.simu.bacteria_nb):
                if sum(chains[i, :i]) == 0:
                    steps.append(step)
                    time.append(sum(self.simu.time_collision[:step + 1]))
                    bact_nb.append(i)
                    l = chains[i, :].sum()
                    length.append(l)
                    vel.append(self.simu.velocity[i, step])
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
            "vel": vel
        })

if __name__=="""__main__""":
    simu = sim.Simu1()
    pp = PostProcessing(simu)
    pp.get_chains()
    print(pp.chains)