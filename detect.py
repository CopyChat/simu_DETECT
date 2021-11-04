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
import matplotlib.pyplot as plt
import numpy as np


@hydra.main(config_path="configs", config_name="detect")
def simulation(cfg: DictConfig) -> None:
    """
    to produce WRF simulation @Reunion island for DETECT project
    """

    if cfg.job.loading_model_setup:
        os.system("./src/get_model_setup.sh")

    if cfg.job.wps.loading_domain:
        os.system("./src/get_domain_ccub.sh")
        print(f'good')

    if cfg.job.wps.plot_domain:
        fig = GEO_PLOT.plot_wrf_domain(num_dom=3, domain_dir=cfg.dir.domain)
        fig.savefig(f'./plot/domain_wrf_d{cfg.simulation.num_dom:g}.png', dpi=300)

    if cfg.job.loading_output_for_detect:
        os.system("./src/get_output.sh")
        print(f'good')

    if cfg.job.check_var:

        name_wrf = ['SWDOWN', 'SWDDNI', 'SWDDIF', 'SWDDIR']
        # note: SWDDNI is normal SWDDIR is direct horizontal radiation, so SWDDNI * cos(zenith angle) = SWDDIR

        name_detect = ['GHI', 'DNI', 'DIF', 'direct_horizontal']     # 'DHI' (diffuse horizontal irradiance), same as DIF

        fig = plt.figure(figsize=(15, 6), dpi=220)

        for i in range(len(name_detect)):
            var = GEO_PLOT.read_to_standard_da(
                f'{cfg.dir.working:s}/local_data/addout.hour_d03_2021-07-30_00.nc', name_wrf[i])

            # st denis
            # var = var.set_index(lat='lat', lon='lon')   # has to set index first, the dims have same names with coords
            # var_local = var.sel(lat=-20.89, lon=55.45, method='nearest')

            # not work if the coords has dif names as dims.
            index_lon = int(np.abs(var.lon - 55.45).argmin())
            index_lat = int(np.abs(var.lat - (-20.89)).argmin())

            var_denis = var[:, index_lat, index_lon]

            plot_day = 3

            data = var_denis[:plot_day * 24]

            plt.plot(range(plot_day * 24), data.values, label=name_wrf[i])
            plt.ylabel(f'{name_wrf[i]:s} ({data.units:s})')

        plt.legend()

        first_time = str(var.time[0].dt.strftime("%Y-%m-%d %H:%M").values)
        plt.xlabel(f'hours since {first_time:s}')

        plt.title(f'SWDDNI * cos(zenith angle) = SWDDIR, see solar height plot')

        plt.savefig('./plot/check_variables_wrf.png', dpi=300)
        plt.show()

        print('good')

    print('good')


if __name__ == "__main__":
    sys.exit(simulation())
