import numpy as np
import openmdao.api as om


class SimpleWindModel(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("n_timesteps", default=100)

    def setup(self):
        n_timesteps = self.options["n_timesteps"]
        self.add_output("time", shape=n_timesteps, units="h")
        self.add_output("wind", shape=n_timesteps, units="m/s")

    def compute(self, inputs, outputs):
        np.random.seed(314)
        n_timesteps = self.options["n_timesteps"]
        outputs["time"] = np.arange(n_timesteps)
        outputs["wind"] = np.random.random(n_timesteps) * 5.0 + 5.0
