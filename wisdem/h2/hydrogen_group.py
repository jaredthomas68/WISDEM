import openmdao.api as om

from wisdem.h2.simple_wind import SimpleWindModel
from wisdem.h2.electrolyzer import SimpleElectrolyzerModel
from wisdem.h2.compute_power import ComputePower


class HydrogenProduction(om.Group):
    """"""

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]
        n_timesteps = modeling_options["WISDEM"]["HydrogenProduction"]["n_timesteps"]

        self.add_subsystem("simple_wind", SimpleWindModel(n_timesteps=n_timesteps), promotes=["*"])

        self.add_subsystem("compute_power", ComputePower(modeling_options=modeling_options), promotes=["*"])

        self.add_subsystem("simple_electrolyzer", SimpleElectrolyzerModel(n_timesteps=n_timesteps), promotes=["*"])
