# Copyright 2022 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import math
import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt


class electrolyzer:
    """
    First draft of a generic / generalized electrolyzer model for H2@scale
    """

    def __init__(self):
        
        # Standard state -> P = 1 atm, T = 298.15 K

        # Constants:
        #self.moles_h2_per_g = 0.4960613 # (mols)
        self.F = 96485.34 # Faraday's constant (C/mol)
        self.R = 8.314 # Ideal gas constant J/(mol*K)
        self.n = 2 # number of electrons transfered in reaction
        self.E_th_0 = 1.481 # Thermoneutral voltage at standard state
        


        # Assumptions: TODO: update with corrected assumptions for H2@Scale project, possible allow as arguments specified during call?
        self.n_cells = 100 # Number of cells
        self.cell_area = 1580 # Cell active area (cm^2)
        self.stack_rating_kW = 750 # Stack rating (kW)
        self.stack_input_voltage = 250 # VDC 
        # self.h2_pres_out = 31 # H2 outlet pressure (bar)

    # def converter_efficiency(self, Pdc): # from empirical ESIF model, unsure if used elsewhere in WEIS/WISDEM
    #     a = -7.798034801
    #     b = -20.201370915
    #     c = 8.704400555
    #     d = -17.680617589

    #     efficiency = (a * Pdc) / (b + Pdc) + (c * Pdc) / (d + Pdc)
    #     return efficiency

    def convert_ac_to_dc(self, Pac): # from empirical ESIF model, can we leave this or should we get a generalized converter model?
        """
        @Kaz, can create a linear fit for the AC on X-axis and DC on Y-axis
        """
        a = 0.000056676
        b = 1.046711474
        c = 18.856165577
        Pdc_plus = (-b + np.sqrt(b ** 2 - 4 * a * (c - Pac))) / (2 * a)
        Pdc_minus = (-b - np.sqrt(b ** 2 - 4 * a * (c - Pac))) / (2 * a)
        return Pdc_minus, Pdc_plus

    # def stack_current_from_voltage(self, V_in, T): # from empirical ESIF model, unsure if used elsewhere in WEIS/WISDEM
    #     a1 = -0.000146297
    #     b1 = 0.025443613
    #     c1 = -0.178823095
    #     d1 = 168.652142633

    #     I = (V_in - c1 * T - d1) / (a1 * T + b1)
    #     return I

    # def stack_voltage_from_current(self, I_in, T): # from empirical ESIF model, unsure if used elsewhere in WEIS/WISDEM
    #     a1 = -0.000146297
    #     b1 = 0.025443613
    #     c1 = -0.178823095
    #     d1 = 168.652142633

    #     V = a1 * I_in * T + b1 * I_in + c1 * T + d1
    #     return V

    def stack_current_from_power(self, P_in, T): 
        ### Old empirical model can be removed
        # a = -0.001385949
        # b = 0.085812325
        # c = 0.007882033
        # d = 5.101696939
        # e = -6.423467186
        # f = 148.372002335

        # I = a * P_in ** 2 + b * T ** 2 + c * P_in * T + d * P_in + e * T + f
        # return I

        # Curtail Wind Power if over electrolyzer rating:
        # self.P_in = P_in
        # self.P_in = np.where(self.P_in > self.stack_rating_kW,
        #                      self.stack_rating_kW,
        #                      self.P_in)
        
        T_K = T + 273.15 
        # Cell reversible voltage:
        E_rev_0 = 1.229  # Reversible cell voltage at standard state
        p_atmo = 101325 # (Pa) atmospheric pressure / pressure of water 
        p_anode = 101325 # (Pa) pressure at anode, assumed atmo 
        p_cathode = 3e+6 #(Pa) pressure at cathode, assumed atmo 
        p_h2O_sat = 19946 # (Pa) @60C TODO: update to be f(T)? https://www.engineeringtoolbox.com/water-vapor-saturation-pressure-d_599.html?vA=60&units=C#
        E_rev = E_rev_0 + ((self.R*T_K)/(self.n*self.F)) * (np.log(((p_anode-p_h2O_sat)/p_atmo) * math.sqrt((p_cathode-p_h2O_sat)/p_atmo))) # General Nernst equation

        # Activation overpotential:
        # constants below assumed from https://www.sciencedirect.com/science/article/pii/S0360319916318341?via%3Dihub 
        T_anode = T_K # TODO: updated with realistic anode temperature? 70-80 C nominal operating temperature 58C
        T_cathode = T_K # TODO: updated with realistic anode temperature?
        alpha_a = 2 # anode charge transfer coefficient TODO: is this a realistic value?
        alpha_c = 0.5 # cathode charge transfer coefficient TODO: is this a realistic value?
        i_0_a = (2 * 10**(-7)) # anode exchange current density TODO: update to be f(T)?
        i_0_c = (2 * 10**(-3)) # cathode exchange current density TODO: update to be f(T)?
        I_in = (P_in*1000)/(self.stack_input_voltage*self.n_cells) # TODO: best way to calculate?
        i = I_in/self.cell_area 

        V_act_a = ((self.R*T_anode)/(alpha_a*self.F)) * np.arcsinh(i/(2*i_0_a)) # derived from Butler-Volmer eqs
        V_act_c = ((self.R*T_cathode)/(alpha_c*self.F)) * np.arcsinh(i/(2*i_0_c)) # derived from Butler-Volmer eqs

        # Ohmic overpotential:
        # pulled from https://www.sciencedirect.com/science/article/pii/S0360319917309278?via%3Dihub 
        lambda_nafion = (((-2.89556 + (0.016 * T_K)) + 1.625) / 0.1875) # TODO: pulled from empirical data, is there a better eq?
        t_nafion = 0.03 # (cm) TODO: confirm actual thickness?
        sigma_nafion = (((0.005139 * lambda_nafion)-0.00326) * math.exp(1268*((1/303)-(1/T_K)))) # TODO: confirm with Nel, is there a better eq?
        R_ohmic = (t_nafion/sigma_nafion) #TODO: update to include electron resistance?

        # Cell / Stack voltage:
        self.V_cell = (E_rev + V_act_a + V_act_c + (i*R_ohmic))
        V_stack = self.V_cell * self.n_cells

        # Stack Current
        I = ((P_in*1000)/V_stack)

        return I 


    # def getH2GenRate(self, kw, eff=0.7):
    #     hhv = 39.4  # kWh/kg
    #     energy_req = hhv / eff
    #     mflow = kw / energy_req

    #     return mflow

    def faradaic_efficiency(self, I):

        # https://res.mdpi.com/d_attachment/energies/energies-13-04792/article_deploy/energies-13-04792-v2.pdf
        
        f_1 = 250 #(mA2/cm4)
        f_2 = 0.996
        i_cell = I * 1000

        n_F = ((((i_cell/self.cell_area)**2) / (f_1 + ((i_cell/self.cell_area)**2))) * f_2)
    
        return n_F

    def electrolysis_efficiency(self):
        #https://www.sciencedirect.com/science/article/pii/S2589299119300035#b0500
        # Based on 1st law of thermo
        n_Th = self.E_th_0/self.V_cell # TODO: Is a valid efficiency loss? confirm should we include i*ASR in denominator?

        return n_Th

    def total_efficiency(self, I):
        n_F = self.faradaic_efficiency(I)
        n_Th = self.electrolysis_efficiency()

        n_Tot = n_F * n_Th

        return n_Tot

    def calcMFRfromCurrent(self, amps, n_Tot):
        """
        calculate a mass flow rate (kg/h) from the current input
        amps - current [A]
        num_cells - number of cells [-]; for a 750-kW stack, we had 100 cells
        """
        mfr = 3.63 * 10 ** (-5) * amps * self.n_cells * n_Tot
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
            h2_current = self.elec.stack_current_from_power(Pdc_plus, 60)

            # Calculate efficiencies:
            n_Tot = self.elec.total_efficiency(h2_current)

            # calculate a mass flow rate
            h2_prod_rate_from_current = self.elec.calcMFRfromCurrent(h2_current, n_Tot)

            outputs["h2_prod_rate"] = h2_prod_rate_from_current

        elif self.distribution_type == "full":
            p_wind = inputs["p_wind"]
            for idx, power in enumerate(p_wind):
                h2_prod_rate = 0.0

                num_stacks_at_full = power // self.elec_stack_size
                Pdc_minus, Pdc_plus = self.elec.convert_ac_to_dc(self.elec_stack_size)

                # create a current profile
                # this h2_current will be sent to the physical stack
                h2_current = self.elec.stack_current_from_power(Pdc_plus, 60)

                # Calculate efficiencies:
                n_Tot = self.elec.total_efficiency(h2_current)

                # calculate a mass flow rate
                h2_prod_rate_from_current = self.elec.calcMFRfromCurrent(h2_current, n_Tot) * num_stacks_at_full

                leftover_power = power % self.elec_stack_size
                Pdc_minus, Pdc_plus = self.elec.convert_ac_to_dc(leftover_power)

                # create a current profile
                # this h2_current will be sent to the physical stack
                h2_current = self.elec.stack_current_from_power(Pdc_plus, 60)

                # calculate a mass flow rate
                h2_prod_rate_from_current += self.elec.calcMFRfromCurrent(h2_current, n_Tot)

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
