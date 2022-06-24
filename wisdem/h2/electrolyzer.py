# Copyright 2022 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt


class electrolyzer:
    """
    This model comes from Kaz's simple electrolyzer model uploaded to
    https://github.nrel.gov/cbay/h2atscale
    """

    def __init__(self):
        pass

    def converter_efficiency(self, Pdc):
        a = -7.798034801
        b = -20.201370915
        c = 8.704400555
        d = -17.680617589

        efficiency = (a * Pdc) / (b + Pdc) + (c * Pdc) / (d + Pdc)
        return efficiency

    def convert_ac_to_dc(self, Pac):
        """
        @Kaz, can create a linear fit for the AC on X-axis and DC on Y-axis
        """
        a = 0.000056676
        b = 1.046711474
        c = 18.856165577
        Pdc_plus = (-b + np.sqrt(b ** 2 - 4 * a * (c - Pac))) / (2 * a)
        Pdc_minus = (-b - np.sqrt(b ** 2 - 4 * a * (c - Pac))) / (2 * a)

        return Pdc_minus, Pdc_plus

    def stack_current_from_voltage(self, V_in, T):
        a1 = -0.000146297
        b1 = 0.025443613
        c1 = -0.178823095
        d1 = 168.652142633

        I = (V_in - c1 * T - d1) / (a1 * T + b1)
        return I

    def stack_voltage_from_current(self, I_in, T):
        a1 = -0.000146297
        b1 = 0.025443613
        c1 = -0.178823095
        d1 = 168.652142633

        V = a1 * I_in * T + b1 * I_in + c1 * T + d1
        return V

    def stack_current_from_power(self, P_in, T):
        a = -0.001385949
        b = 0.085812325
        c = 0.007882033
        d = 5.101696939
        e = -6.423467186
        f = 148.372002335

        I = a * P_in ** 2 + b * T ** 2 + c * P_in * T + d * P_in + e * T + f
        return I

    def getH2GenRate(self, kw, eff=0.7):
        hhv = 39.4  # kWh/kg
        energy_req = hhv / eff
        mflow = kw / energy_req

        return mflow

    def calcMFRfromCurrent(self, amps, num_cells=100):
        """
        calculate a mass flow rate (kg/h) from the current input
        amps - current [A]
        num_cells - number of cells [-]; for a 750-kW stack, we had 100 cells
        """
        mfr = 3.63 * 10 ** (-5) * amps * num_cells
        return mfr


class SimpleElectrolyzerModel(om.ExplicitComponent):
    """
    This is an OpenMDAO wrapper to the simple electrolyzer model above.

    It makes some assumptions about the number of electrolyzers, stack size, and
    how to distribute electricity across the different electrolyzers. These
    could be later made into WISDEM modeling options to allow for more user configuration.
    """

    def setup(self):
        self.add_input("p_wind", shape_by_conn=True, units="kW")
        self.add_input("time", shape_by_conn=True, units="h")
        self.add_output("h2_prod_rate", shape_by_conn=True, copy_shape="time", units="kg/h")
        self.add_output("h2_produced", units="kg")

        self.n_electrolyzers = 5  # hardcoded for now, should actually depend on rated turbine power
        self.elec_stack_size = 750  # kW
        self.distribution_type = "full"  # even or full; (divided wind power
        # evenly across electrolyzers or run as many
        # at capacity as possible and the rest empty)

        self.elec = electrolyzer()

    def compute(self, inputs, outputs):
        if self.distribution_type == "even":
            # Split power evenly across all electrolyzer stacks
            Pdc_minus, Pdc_plus = self.elec.convert_ac_to_dc(inputs["p_wind"] / self.n_electrolyzers)

            # create a current profile
            # this h2_current will be sent to the physical stack
            h2_current = self.elec.stack_current_from_power(Pdc_plus, 70)

            # calculate a mass flow rate
            h2_prod_rate_from_current = self.elec.calcMFRfromCurrent(h2_current)

            outputs["h2_prod_rate"] = h2_prod_rate_from_current

        elif self.distribution_type == "full":
            p_wind = inputs["p_wind"]
            for idx, power in enumerate(p_wind):
                h2_prod_rate = 0.0

                num_stacks_at_full = power // self.elec_stack_size
                Pdc_minus, Pdc_plus = self.elec.convert_ac_to_dc(self.elec_stack_size)

                # create a current profile
                # this h2_current will be sent to the physical stack
                h2_current = self.elec.stack_current_from_power(Pdc_plus, 70)

                # calculate a mass flow rate
                h2_prod_rate_from_current = self.elec.calcMFRfromCurrent(h2_current) * num_stacks_at_full

                leftover_power = power % self.elec_stack_size
                Pdc_minus, Pdc_plus = self.elec.convert_ac_to_dc(leftover_power)

                # create a current profile
                # this h2_current will be sent to the physical stack
                h2_current = self.elec.stack_current_from_power(Pdc_plus, 70)

                # calculate a mass flow rate
                h2_prod_rate_from_current += self.elec.calcMFRfromCurrent(h2_current)

                outputs["h2_prod_rate"][idx] = h2_prod_rate_from_current

        outputs["h2_produced"] = np.trapz(outputs["h2_prod_rate"], inputs["time"])


if __name__ == "__main__":
    prob = om.Problem(model=om.Group())
    prob.model.add_subsystem("electrolyzer", SimpleElectrolyzerModel(), promotes=["*"])

    prob.setup()
    prob["p_wind"] = np.linspace(2000.0, 4000.0, 100)
    prob["time"] = np.linspace(0.0, 100.0, 100)

    prob.run_model()

    prob.model.list_outputs(print_arrays=True)
