import numpy as np
import openmdao.api as om
from scipy import interpolate


class ComputePower(om.ExplicitComponent):
    """
    Simple component to receive wind speeds and powercurve data for the WISDEM
    wind turbine. Based on these values, we produce a timeseries of power data
    that can later be inputted into an electrolyzer model.
    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]
        n_pc_spline = modeling_options["WISDEM"]["RotorSE"]["n_pc_spline"]
        self.n_timesteps = modeling_options["WISDEM"]["HydrogenProduction"]["n_timesteps"]

        self.add_input("wind", shape=self.n_timesteps, units="m/s")
        self.add_input("V_spline", val=np.zeros(n_pc_spline), units="m/s", desc="wind vector")
        self.add_input("P_spline", val=np.zeros(n_pc_spline), units="W", desc="rotor electrical power")
        self.add_output("p_wind", shape=self.n_timesteps, units="W")

    def compute(self, inputs, outputs):
        power_interp = interpolate.interp1d(inputs["V_spline"], inputs["P_spline"])
        outputs["p_wind"] = power_interp(inputs["wind"])
