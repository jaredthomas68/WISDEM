import os

import numpy as np
import matplotlib.pyplot as plt

from wisdem.glue_code.runWISDEM import run_wisdem

## File management
run_dir = os.path.dirname(os.path.realpath(__file__))
fname_wt_input = run_dir + os.sep + "IEA-15-240-RWT_VolturnUS-S.yaml"
fname_modeling_options = run_dir + os.sep + "modeling_options.yaml"
fname_analysis_options = run_dir + os.sep + "analysis_options.yaml"

wt_opt, modeling_options, opt_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

wt_opt.model.list_outputs(includes=["*h2*"], units=True, print_arrays=True)


wind = wt_opt.get_val("h2.simple_wind.wind", units="m/s")
power = wt_opt.get_val("h2.compute_power.p_wind", units="MW")
h2_prod_rate = wt_opt.get_val("h2.simple_electrolyzer.h2_prod_rate", units="kg/h")
time = np.linspace(0.0, 100.0, 100)

fig, axarr = plt.subplots(4, 1)
axarr[0].plot(time, wind)
axarr[0].set_ylabel("Wind speed, m/s")

axarr[1].plot(time, power)
axarr[1].set_ylabel("Power, MW")

axarr[2].plot(time, h2_prod_rate)
axarr[2].set_ylabel("H2 production rate, kg/h")

axarr[3].plot(time, h2_prod_rate / power)
axarr[3].set_ylabel("H2 production rate by power")
axarr[3].set_xlabel("Nondimensional time")

plt.show()
