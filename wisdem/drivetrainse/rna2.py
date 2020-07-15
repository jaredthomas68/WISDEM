from __future__ import print_function
import numpy as np
from openmdao.api import ExplicitComponent, Group, IndepVarComp

from wisdem.commonse.utilities import hstack, vstack
from wisdem.commonse.csystem import DirectionVector
from wisdem.commonse import gravity

# This is an extremely simple RNA mass calculator that should be used when DriveSE otherwise seems too complicated

#--------------------------------------------
class RNAMass(om.ExplicitComponent):
    def setup(self):

        # variables
        self.add_input('rotor_mass', 0.0, units='kg', desc='mass of the rotor')
        self.add_input('nacelle_mass', 0.0, units='kg', desc='mass of nacelle')

        self.add_input('hub_system_cm', np.zeros(3), units='m', desc='location of hub center of mass relative to tower top in yaw-aligned c.s.')
        self.add_input('nacelle_cm', np.zeros(3), units='m', desc='location of nacelle center of mass relative to tower top in yaw-aligned c.s.')

        # order for all moments of inertia is (xx, yy, zz, xy, xz, yz) in the yaw-aligned coorinate system
        self.add_input('blades_I', np.zeros(6), units='kg*m**2', desc='mass moments of inertia of all blades about hub center')
        self.add_input('hub_system_I', np.zeros(6), units='kg*m**2', desc='mass moments of inertia of hub about its center of mass')
        self.add_input('nacelle_I', np.zeros(6), units='kg*m**2', desc='mass moments of inertia of nacelle about its center of mass')

        # outputs
        self.add_output('rna_mass', 0.0, units='kg', desc='total mass of RNA')
        self.add_output('rna_cm', np.zeros(3), units='m', desc='location of RNA center of mass relative to tower top in yaw-aligned c.s.')
        self.add_output('rna_I_TT', np.zeros(6), units='kg*m**2', desc='mass moments of inertia of RNA about tower top in yaw-aligned coordinate system')

        self.declare_partials('*','*')

    def _assembleI(self, I):
        Ixx, Iyy, Izz, Ixy, Ixz, Iyz = I[0], I[1], I[2], I[3], I[4], I[5] 
        return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])


    def _unassembleI(self, I):
        return np.array([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]])


    def compute(self, inputs, outputs):

        nac_mass = inputs['nacelle_mass']
        rotor_mass = inputs['rotor_mass']
        outputs['rna_mass'] = rotor_mass + nac_mass
        # rna cm
        outputs['rna_cm'] = (rotor_mass*inputs['hub_system_cm'] + nac_mass*inputs['nacelle_cm'])/outputs['rna_mass']

        # rna I
        blades_I = self._assembleI(inputs['blades_I'])
        hub_I = self._assembleI(inputs['hub_system_I'])
        nac_I = self._assembleI(inputs['nacelle_I'])
        rotor_I = blades_I + hub_I

        R = inputs['hub_system_cm']
        rotor_I_TT = rotor_I + rotor_mass*(np.dot(R, R)*np.eye(3) - np.outer(R, R))

        R = inputs['nacelle_cm']
        nac_I_TT = nac_I + nac_mass*(np.dot(R, R)*np.eye(3) - np.outer(R, R))

        outputs['rna_I_TT'] = self._unassembleI(rotor_I_TT + nac_I_TT)


    def compute_partials(self, inputs, J):

        nac_mass = inputs['nacelle_mass']
        hub_cm = inputs['hub_system_cm']
        nac_cm = inputs['nacelle_cm']
        hub_I = inputs['hub_system_I']
        nac_I = inputs['nacelle_I']
        rotor_mass = inputs['rotor_mass']
        rna_mass = rotor_mass + nac_mass
        

        # mass
        J['rna_mass', 'rotor_mass'] = 1.0
        J['rna_mass', 'nacelle_mass'] = 1.0
        J['rna_mass', 'hub_system_cm'] = np.zeros(3)
        J['rna_mass', 'nacelle_cm'] = np.zeros(3)
        J['rna_mass', 'blades_I'] = np.zeros(6)
        J['rna_mass', 'hub_system_I'] = np.zeros(6)
        J['rna_mass', 'nacelle_I'] = np.zeros(6)
        

        # cm
        numerator = rotor_mass*hub_cm+nac_mass*nac_cm

        J['rna_cm', 'rotor_mass'] = (rna_mass*hub_cm-numerator)/rna_mass**2
        J['rna_cm', 'nacelle_mass'] = (rna_mass*nac_cm-numerator)/rna_mass**2
        J['rna_cm', 'hub_system_cm'] = rotor_mass/rna_mass*np.eye(3)
        J['rna_cm', 'nacelle_cm'] = nac_mass/rna_mass*np.eye(3)
        J['rna_cm', 'blades_I'] = np.zeros((3, 6))
        J['rna_cm', 'hub_system_I'] = np.zeros((3, 6))
        J['rna_cm', 'nacelle_I'] = np.zeros((3, 6))


        # I
        R = hub_cm
        const = self._unassembleI(np.dot(R, R)*np.eye(3) - np.outer(R, R))

        J['rna_I_TT', 'rotor_mass'] = const
        dI_drx = rotor_mass*self._unassembleI(2*R[0]*np.eye(3) - np.array([[2*R[0], R[1], R[2]], [R[1], 0.0, 0.0], [R[2], 0.0, 0.0]]))
        dI_dry = rotor_mass*self._unassembleI(2*R[1]*np.eye(3) - np.array([[0.0, R[0], 0.0], [R[0], 2*R[1], R[2]], [0.0, R[2], 0.0]]))
        dI_drz = rotor_mass*self._unassembleI(2*R[2]*np.eye(3) - np.array([[0.0, 0.0, R[0]], [0.0, 0.0, R[1]], [R[0], R[1], 2*R[2]]]))
        J['rna_I_TT', 'hub_system_cm'] = np.vstack([dI_drx, dI_dry, dI_drz]).T

        R = nac_cm
        const = self._unassembleI(np.dot(R, R)*np.eye(3) - np.outer(R, R))
        J['rna_I_TT', 'nacelle_mass'] = const
        dI_drx = nac_mass*self._unassembleI(2*R[0]*np.eye(3) - np.array([[2*R[0], R[1], R[2]], [R[1], 0.0, 0.0], [R[2], 0.0, 0.0]]))
        dI_dry = nac_mass*self._unassembleI(2*R[1]*np.eye(3) - np.array([[0.0, R[0], 0.0], [R[0], 2*R[1], R[2]], [0.0, R[2], 0.0]]))
        dI_drz = nac_mass*self._unassembleI(2*R[2]*np.eye(3) - np.array([[0.0, 0.0, R[0]], [0.0, 0.0, R[1]], [R[0], R[1], 2*R[2]]]))
        J['rna_I_TT', 'nacelle_cm'] = np.vstack([dI_drx, dI_dry, dI_drz]).T

        J['rna_I_TT', 'blades_I'] = np.eye(6)
        J['rna_I_TT', 'hub_system_I'] = np.eye(6)
        J['rna_I_TT', 'nacelle_I'] = np.eye(6)

        

#--------------------------------------------
class RotorLoads(om.ExplicitComponent):
    def setup(self):

        # variables
        self.add_input('Fxyz', np.zeros(3), units='N', desc='forces in hub-aligned coordinate system')
        self.add_input('Mxyz', np.zeros(3), units='N*m', desc='moments in hub-aligned coordinate system')
        self.add_input('hub_system_cm', np.zeros(3), units='m', desc='position of rotor hub relative to tower top in yaw-aligned c.s.')
        self.add_input('rna_mass', 0.0, units='kg', desc='mass of rotor nacelle assembly')
        self.add_input('rna_cm', np.zeros(3), units='m', desc='location of RNA center of mass relative to tower top in yaw-aligned c.s.')

        self.add_discrete_input('rna_weightM', True, desc='Flag to indicate whether or not the RNA weight should be considered.\
                          An upwind overhang may lead to unconservative estimates due to the P-Delta effect(suggest not using). For downwind turbines set to True. ')

        # parameters
        self.add_discrete_input('downwind', False)
        self.add_input('tilt', 0.0, units='deg')

        # out
        self.add_output('top_F', np.zeros(3), units='N')  # in yaw-aligned
        self.add_output('top_M', np.zeros(3), units='N*m')

        self.declare_partials('top_F', ['Fxyz','Mxyz','hub_system_cm','rna_mass','rna_cm'])
        self.declare_partials('top_M', ['Fxyz','Mxyz','hub_system_cm','rna_mass','rna_cm'])


    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        F = inputs['Fxyz']
        M = inputs['Mxyz']
        tilt = float(inputs['tilt'])
        
        F = DirectionVector.fromArray(F).hubToYaw(tilt)
        M = DirectionVector.fromArray(M).hubToYaw(tilt)

        # change x-direction if downwind
        hub_cm = np.copy(inputs['hub_system_cm'])
        rna_cm = np.copy(inputs['rna_cm'])
        if discrete_inputs['downwind']:
            hub_cm[0] *= -1
            rna_cm[0] *= -1
        hub_cm = DirectionVector.fromArray(hub_cm)
        rna_cm = DirectionVector.fromArray(rna_cm)
        self.save_rhub = hub_cm
        self.save_rcm = rna_cm

        # aerodynamic moments
        M = M + hub_cm.cross(F)
        self.saveF = F


        # add weight loads
        F_w = DirectionVector(0.0, 0.0, -float(inputs['rna_mass'])*gravity)
        M_w = rna_cm.cross(F_w)
        self.saveF_w = F_w

        Fout = F + F_w

        if discrete_inputs['rna_weightM']:
            Mout = M + M_w
        else:
            Mout = M
            #REMOVE WEIGHT EFFECT TO ACCOUNT FOR P-Delta Effect
            print("!!!! No weight effect on rotor moments -TowerSE  !!!!")

        # put back in array
        outputs['top_F'] = np.array([Fout.x, Fout.y, Fout.z])
        outputs['top_M'] = np.array([Mout.x, Mout.y, Mout.z])

    def compute_partials(self, inputs, J, discrete_inputs):

        dF = DirectionVector.fromArray(inputs['Fxyz']).hubToYaw(inputs['tilt'])
        dFx, dFy, dFz = dF.dx, dF.dy, dF.dz

        dtopF_dFx = np.array([dFx['dx'], dFy['dx'], dFz['dx']])
        dtopF_dFy = np.array([dFx['dy'], dFy['dy'], dFz['dy']])
        dtopF_dFz = np.array([dFx['dz'], dFy['dz'], dFz['dz']])
        dtopF_dF = hstack([dtopF_dFx, dtopF_dFy, dtopF_dFz])
        dtopF_w_dm = np.array([0.0, 0.0, -gravity])

        #dtopF = hstack([dtopF_dF, np.zeros((3, 6)), dtopF_w_dm, np.zeros((3, 3))])


        dM = DirectionVector.fromArray(inputs['Mxyz']).hubToYaw(inputs['tilt'])
        dMx, dMy, dMz = dM.dx, dM.dy, dM.dz
        dMxcross, dMycross, dMzcross = self.save_rhub.cross_deriv(self.saveF, 'dr', 'dF')

        dtopM_dMx = np.array([dMx['dx'], dMy['dx'], dMz['dx']])
        dtopM_dMy = np.array([dMx['dy'], dMy['dy'], dMz['dy']])
        dtopM_dMz = np.array([dMx['dz'], dMy['dz'], dMz['dz']])
        dtopM_dM = hstack([dtopM_dMx, dtopM_dMy, dtopM_dMz])
        dM_dF = np.array([dMxcross['dF'], dMycross['dF'], dMzcross['dF']])

        dtopM_dFx = np.dot(dM_dF, dtopF_dFx)
        dtopM_dFy = np.dot(dM_dF, dtopF_dFy)
        dtopM_dFz = np.dot(dM_dF, dtopF_dFz)
        dtopM_dF = hstack([dtopM_dFx, dtopM_dFy, dtopM_dFz])
        dtopM_dr = np.array([dMxcross['dr'], dMycross['dr'], dMzcross['dr']])

        dMx_w_cross, dMy_w_cross, dMz_w_cross = self.save_rcm.cross_deriv(self.saveF_w, 'dr', 'dF')

        if discrete_inputs['rna_weightM']:
            dtopM_drnacm = np.array([dMx_w_cross['dr'], dMy_w_cross['dr'], dMz_w_cross['dr']])
            dtopM_dF_w = np.array([dMx_w_cross['dF'], dMy_w_cross['dF'], dMz_w_cross['dF']])
        else:
            dtopM_drnacm = np.zeros((3, 3))
            dtopM_dF_w = np.zeros((3, 3))
        dtopM_dm = np.dot(dtopM_dF_w, dtopF_w_dm)

        if discrete_inputs['downwind']:
            dtopM_dr[:, 0] *= -1
            dtopM_drnacm[:, 0] *= -1

        #dtopM = hstack([dtopM_dF, dtopM_dM, dtopM_dr, dtopM_dm, dtopM_drnacm])

        
        J['top_F', 'Fxyz'] = dtopF_dF
        J['top_F', 'Mxyz'] = np.zeros((3, 3))
        J['top_F', 'hub_system_cm'] = np.zeros((3, 3))
        J['top_F', 'rna_mass'] = dtopF_w_dm
        J['top_F', 'rna_cm'] = np.zeros((3, 3))

        J['top_M', 'Fxyz'] = dtopM_dF
        J['top_M', 'Mxyz'] = dtopM_dM
        J['top_M', 'hub_system_cm'] = dtopM_dr
        J['top_M', 'rna_mass'] = dtopM_dm
        J['top_M', 'rna_cm'] = dtopM_drnacm
