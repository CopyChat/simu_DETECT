"""
to find the physical link between the SSR classification and the large-scale variability
over la reunion island and it's extended area
"""

__version__ = f'Version 2.0  \nTime-stamp: <2021-05-15>'
__author__ = "ChaoTANG@univ-reunion.fr"

import os
import sys
import hydra
from omegaconf import DictConfig
import GEO_PLOT


@hydra.main(config_path="configs", config_name="detect")
def simulation(cfg: DictConfig) -> None:
    """
    to produce WRF simulation @Reunion island for DETECT project
    """

    if cfg.job.wps.loading_domain:

        os.system("./src/get_domain_ccub.sh")
        print(f'good')

    if cfg.job.wps.plot_domain:

        fig = GEO_PLOT.plot_wrf_domain(num_dom=3, domain_dir=cfg.dir.domain)
        fig.savefig(f'./plot/domain_wrf_d{cfg.simulation.num_dom:g}.png', dpi=300)

    if cfg.job.loading_namelist:
        os.system("./src/get_namelist.sh")


if __name__ == "__main__":
    sys.exit(simulation())
