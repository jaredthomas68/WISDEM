import numpy as np
import openmdao.api as om


class ReadInWind(om.ExplicitComponent):
    """
    Quick wrapper to read in wind timeseries history from TurbSim.
    This requires WEIS to be installed.

    The wind timeseries will then be passed into the powercurve-approximated
    conversion to obtain a power timeseries that can then be passed to an
    electrolyzer model. This means that turbine design changes on the WISDEM
    level will propagate through this component and affect the power and H2 produced.
    """

    def initialize(self):
        self.options.declare("filename")

        try:
            from weis.aeroelasticse.turbsim_file import TurbSimFile
        except:
            raise Exception(
                "Trying to read in a TurbSim file but WEIS is not installed. Please install WEIS to use its file processor."
            )

        out = TurbSimFile(self.options["filename"])
        iy, iz = out._iMid()
        self.wind = out["u"][0, :, iy, iz]
        self.time = out["t"]
        self.n_timesteps = len(self.time)

    def setup(self):
        self.add_output("time", shape=self.n_timesteps, units="s")
        self.add_output("wind", shape=self.n_timesteps, units="m/s")

    def compute(self, inputs, outputs):
        outputs["time"] = self.time
        outputs["wind"] = self.wind
