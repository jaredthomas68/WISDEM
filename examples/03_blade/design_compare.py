import numpy as np
import matplotlib.pyplot as plt
from wisdem.glue_code.runWISDEM import run_wisdem
import os

show_plots = True

## File management
run_dir                = os.path.dirname( os.path.realpath(__file__) ) + os.sep
fname_wt_input1        = run_dir + 'blade.yaml'
fname_wt_input2        = run_dir + os.sep + 'outputs_aero' + os.sep + 'blade_out.yaml'
# fname_wt_input2        = run_dir + os.sep + 'outputs_struct' + os.sep + 'blade_out.yaml'
# fname_wt_input2        = run_dir + os.sep + 'outputs_aerostruct' + os.sep + 'blade_out.yaml'
fname_modeling_options = run_dir + 'modeling_options.yaml'
fname_analysis_options = run_dir + 'analysis_options_no_opt.yaml'

wt_opt1, modeling_options1, analysis_options1 = run_wisdem(fname_wt_input1, fname_modeling_options, fname_analysis_options)
# wt_opt2, modeling_options2, analysis_options2 = run_wisdem(fname_wt_input2, fname_modeling_options, fname_analysis_options)

label1    = 'Initial'
label2    = 'Optimized'
font_size        = 12
extension = '.png' # '.pdf'

folder_output = analysis_options1['general']['folder_output']

list_of_yamls = [wt_opt1, wt_opt1, wt_opt1]
list_of_yaml_labels = [label1, label2, "bamboo"]
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


### Do not change code below here

"""
Simple plots:
edge
flap
induction
lift_coeff
mass
skin_opt
strains_opt
torsion

More complex:
chord
sc_opt
twist_opt

Special:
AOA
af_efficiency

"""

# # Twist
# ftw, axtw = plt.subplots(1,1,figsize=(5.3, 4))
# axtw.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['blade.outer_shape_bem.twist'] * 180. / np.pi,'--',color = colors[0], label=label1)
# axtw.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['blade.outer_shape_bem.twist'] * 180. / np.pi,'-',color = colors[1], label=label2)
# s_opt_twist = np.linspace(0., 1., 8)
# twist_opt  = np.interp(s_opt_twist, wt_opt2['blade.outer_shape_bem.s'], wt_opt2['ccblade.theta'])
# axtw.plot(s_opt_twist, twist_opt * 180. / np.pi,'o',color = colors[1], markersize=3, label='Optimized - Control Points')
# axtw.plot(s_opt_twist, np.array(analysis_options2['optimization_variables']['blade']['aero_shape']['twist']['lower_bound']) * 180. / np.pi, ':o', color=colors[2],markersize=3, label = 'Bounds')
# axtw.plot(s_opt_twist, np.array(analysis_options2['optimization_variables']['blade']['aero_shape']['twist']['upper_bound']) * 180. / np.pi, ':o', color=colors[2], markersize=3)
# axtw.legend(fontsize=font_size)
# axtw.set_ylim([-5, 20])
# plt.xlabel('Blade Nondimensional Span [-]', fontsize=font_size+2, fontweight='bold')
# plt.ylabel('Twist [deg]', fontsize=font_size+2, fontweight='bold')
# plt.xticks(fontsize=font_size)
# plt.yticks(fontsize=font_size)
# plt.grid(color=[0.8,0.8,0.8], linestyle='--')
# plt.subplots_adjust(bottom = 0.15, left = 0.15)
# fig_name = 'twist_opt' + extension
# ftw.savefig(os.path.join(folder_output, fig_name))
# 
# # Chord
# fc, axc = plt.subplots(1,1,figsize=(5.3, 4))
# axc.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['blade.outer_shape_bem.chord'],'--', color = colors[0], label=label1)
# axc.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['blade.outer_shape_bem.chord'],'-', color = colors[1], label=label2)
# s_opt_chord = np.linspace(0., 1., 8)
# chord_opt  = np.interp(s_opt_chord, wt_opt2['blade.outer_shape_bem.s'], wt_opt2['ccblade.chord'])
# chord_init = np.interp(s_opt_chord, wt_opt2['blade.outer_shape_bem.s'], wt_opt1['blade.outer_shape_bem.chord'])
# axc.plot(s_opt_chord, chord_opt,'o',color = colors[1], markersize=3, label='Optimized - Control Points')
# axc.plot(s_opt_chord, np.array(analysis_options2['optimization_variables']['blade']['aero_shape']['chord']['min_gain']) * chord_init, ':o', color=colors[2],markersize=3, label = 'Bounds')
# axc.plot(s_opt_chord, np.array(analysis_options2['optimization_variables']['blade']['aero_shape']['chord']['max_gain']) * chord_init, ':o', color=colors[2], markersize=3)
# axc.legend(fontsize=font_size)
# plt.xlabel('Blade Nondimensional Span [-]', fontsize=font_size+2, fontweight='bold')
# plt.ylabel('Chord [m]', fontsize=font_size+2, fontweight='bold')
# plt.xticks(fontsize=font_size)
# plt.yticks(fontsize=font_size)
# plt.grid(color=[0.8,0.8,0.8], linestyle='--')
# plt.subplots_adjust(bottom = 0.15, left = 0.15)
# fig_name = 'chord' + extension
# fc.savefig(os.path.join(folder_output, fig_name))
# 
# 
# # Spar caps
# fsc, axsc = plt.subplots(1,1,figsize=(5.3, 4))
# n_layers = len(wt_opt1['blade.internal_structure_2d_fem.layer_thickness'][:,0])
# spar_ss_name = analysis_options1['optimization_variables']['blade']['structure']['spar_cap_ss']['name']
# spar_ps_name = analysis_options2['optimization_variables']['blade']['structure']['spar_cap_ps']['name']
# for i in range(n_layers):
#     if modeling_options1['blade']['layer_name'][i] == spar_ss_name:
#         axsc.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['blade.internal_structure_2d_fem.layer_thickness'][i,:] * 1.e+3,'--', color = colors[0], label=label1)
#         axsc.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['blade.internal_structure_2d_fem.layer_thickness'][i,:] * 1.e+3,'-', color = colors[1], label=label2)
# 
#         s_opt_sc = np.linspace(0., 1., 8)
#         sc_opt  = np.interp(s_opt_sc, wt_opt2['blade.outer_shape_bem.s'], wt_opt2['blade.internal_structure_2d_fem.layer_thickness'][i,:] * 1.e+3)
#         sc_init = np.interp(s_opt_sc, wt_opt2['blade.outer_shape_bem.s'], wt_opt1['blade.internal_structure_2d_fem.layer_thickness'][i,:] * 1.e+3)
#         axsc.plot(s_opt_sc, sc_opt,'o',color = colors[1], markersize=3, label='Optimized - Control Points')
#         axsc.plot(s_opt_sc, np.array(analysis_options2['optimization_variables']['blade']['structure']['spar_cap_ss']['min_gain']) * sc_init, ':o', color=colors[2],markersize=3, label = 'Bounds')
#         axsc.plot(s_opt_sc, np.array(analysis_options2['optimization_variables']['blade']['structure']['spar_cap_ss']['max_gain']) * sc_init, ':o', color=colors[2], markersize=3)
# 
# axsc.legend(fontsize=font_size)
# plt.ylim([0., 200])
# plt.xlabel('Blade Nondimensional Span [-]', fontsize=font_size+2, fontweight='bold')
# plt.ylabel('Spar Caps Thickness [mm]', fontsize=font_size+2, fontweight='bold')
# plt.xticks(fontsize=font_size)
# plt.yticks(fontsize=font_size)
# plt.grid(color=[0.8,0.8,0.8], linestyle='--')
# plt.subplots_adjust(bottom = 0.15, left = 0.15)
# fig_name = 'sc_opt' + extension
# fsc.savefig(os.path.join(folder_output, fig_name))
# 
# # Skins
# f, ax = plt.subplots(1,1,figsize=(5.3, 4))
# ax.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['blade.internal_structure_2d_fem.layer_thickness'][1,:] * 1.e+3,'--', color = colors[0], label=label1)
# ax.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['blade.internal_structure_2d_fem.layer_thickness'][1,:] * 1.e+3,'-', color = colors[1], label=label2)
# ax.legend(fontsize=font_size)
# #plt.ylim([0., 120])
# plt.xlabel('Blade Nondimensional Span [-]', fontsize=font_size+2, fontweight='bold')
# plt.ylabel('Outer Shell Skin Thickness [mm]', fontsize=font_size+2, fontweight='bold')
# plt.xticks(fontsize=font_size)
# plt.yticks(fontsize=font_size)
# plt.grid(color=[0.8,0.8,0.8], linestyle='--')
# plt.subplots_adjust(bottom = 0.15, left = 0.15)
# fig_name = 'skin_opt' + extension
# f.savefig(os.path.join(folder_output, fig_name))
# 
# 
# 
# 
# # Strains spar caps
# feps, axeps = plt.subplots(1,1,figsize=(5.3, 4))
# axeps.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['rlds.frame.strainU_spar'] * 1.e+6,'--',color = colors[0], label=label1)
# axeps.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['rlds.frame.strainU_spar'] * 1.e+6,'-',color = colors[1], label=label2)
# axeps.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['rlds.frame.strainL_spar'] * 1.e+6,'--',color = colors[0])
# axeps.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['rlds.frame.strainL_spar'] * 1.e+6,'-',color = colors[1])
# # axeps.plot(np.array([0.,1.]), np.array([3000.,3000.]), ':',color=colors[2], label='Constraints')
# # axeps.plot(np.array([0.,1.]), np.array([-3000.,-3000.]), ':', color=colors[2],)
# plt.ylim([-5e+3, 5e+3])
# axeps.legend(fontsize=font_size)
# plt.xlabel('Blade Nondimensional Span [-]', fontsize=font_size+2, fontweight='bold')
# plt.ylabel('Spar Caps Strains [mu eps]', fontsize=font_size+2, fontweight='bold')
# plt.xticks(fontsize=font_size)
# plt.yticks(fontsize=font_size)
# plt.grid(color=[0.8,0.8,0.8], linestyle='--')
# plt.subplots_adjust(bottom = 0.15, left = 0.2)
# fig_name = 'strains_opt' + extension
# feps.savefig(os.path.join(folder_output, fig_name))
# 
# 
# # Angle of attack and stall angle
# faoa, axaoa = plt.subplots(1,1,figsize=(5.3, 4))
# axaoa.plot(wt_opt1['stall_check.s'], wt_opt1['stall_check.aoa_along_span'],'--',color = colors[0], label=label1)
# axaoa.plot(wt_opt2['stall_check.s'], wt_opt2['stall_check.aoa_along_span'],'-',color = colors[1], label=label2)
# axaoa.plot(wt_opt1['stall_check.s'], wt_opt1['stall_check.stall_angle_along_span'],':',color=colors[2], label='Stall')
# axaoa.legend(fontsize=font_size)
# axaoa.set_ylim([0, 20])
# plt.xlabel('Blade Nondimensional Span [-]', fontsize=font_size+2, fontweight='bold')
# plt.ylabel('Angle of Attack [deg]', fontsize=font_size+2, fontweight='bold')
# plt.xticks(fontsize=font_size)
# plt.yticks(fontsize=font_size)
# plt.grid(color=[0.8,0.8,0.8], linestyle='--')
# plt.subplots_adjust(bottom = 0.15, left = 0.15)
# fig_name = 'aoa' + extension
# faoa.savefig(os.path.join(folder_output, fig_name))
# 
# # Airfoil efficiency
# feff, axeff = plt.subplots(1,1,figsize=(5.3, 4))
# axeff.plot(wt_opt1['blade.outer_shape_bem.s'], wt_opt1['sse.powercurve.cl_regII'] / wt_opt1['sse.powercurve.cd_regII'],'--',color = colors[0], label=label1)
# axeff.plot(wt_opt2['blade.outer_shape_bem.s'], wt_opt2['sse.powercurve.cl_regII'] / wt_opt2['sse.powercurve.cd_regII'],'-',color = colors[1], label=label2)
# axeff.legend(fontsize=font_size)
# plt.xlabel('Blade Nondimensional Span [-]', fontsize=font_size+2, fontweight='bold')
# plt.ylabel('Airfoil Efficiency [-]', fontsize=font_size+2, fontweight='bold')
# plt.xticks(fontsize=font_size)
# plt.yticks(fontsize=font_size)
# plt.grid(color=[0.8,0.8,0.8], linestyle='--')
# plt.subplots_adjust(bottom = 0.15, left = 0.15)
# fig_name = 'af_efficiency' + extension
# feff.savefig(os.path.join(folder_output, fig_name))
# 
# 
# def simple_plot_results(x_axis_label, y_axis_label, x_axis_data_name, y_axis_data_name, plot_filename):
#     f, ax = plt.subplots(1,1,figsize=(5.3, 4))
#     for i_yaml, yaml_data in enumerate(list_of_yamls):
#         ax.plot(yaml_data[x_axis_data_name], yaml_data[y_axis_data_name],'--', color = colors[i_yaml], label=list_of_yaml_labels[i_yaml])
#     ax.legend(fontsize=font_size)
#     plt.xlabel(x_axis_label, fontsize=font_size+2, fontweight='bold')
#     plt.ylabel(y_axis_label, fontsize=font_size+2, fontweight='bold')
#     plt.xticks(fontsize=font_size)
#     plt.yticks(fontsize=font_size)
#     plt.grid(color=[0.8,0.8,0.8], linestyle='--')
#     plt.subplots_adjust(bottom = 0.15, left = 0.15)
#     fig_name = plot_filename + extension
#     f.savefig(os.path.join(folder_output, fig_name))
# 
# # Edgewise stiffness
# simple_plot_results('Blade Nondimensional Span [-]', 'Edgewise Stiffness [Nm2]', 'blade.outer_shape_bem.s', 'elastic.EIxx', 'edge')
# 
# # Torsional stiffness
# simple_plot_results('Blade Nondimensional Span [-]', 'Torsional Stiffness [Nm2]', 'blade.outer_shape_bem.s', 'elastic.GJ', 'torsion')
# 
# # Flapwise stiffness
# simple_plot_results('Blade Nondimensional Span [-]', 'Flapwise Stiffness [Nm2]', 'blade.outer_shape_bem.s', 'elastic.EIyy', 'flap')
# 
# # Mass
# simple_plot_results('Blade Nondimensional Span [-]', 'Unit Mass [kg/m]', 'blade.outer_shape_bem.s', 'elastic.rhoA', 'mass')
# 
# # Induction
# simple_plot_results('Blade Nondimensional Span [-]', 'Axial Induction [-]', 'blade.outer_shape_bem.s', 'sse.powercurve.ax_induct_regII', 'induction')
# 
# # Lift coefficient
# simple_plot_results('Blade Nondimensional Span [-]', 'Lift Coefficient [-]', 'blade.outer_shape_bem.s', 'sse.powercurve.cl_regII', 'lift_coeff')
# 
# 
# if show_plots:
#     plt.show()




values_to_print = {
    'AEP' : ['sse.AEP', 'GW*h'],
    'blade mass' : ['elastic.precomp.blade_mass', 'kg'],
    'LCOE' : ['financese.lcoe', 'USD/(MW*h)'],
}



list_of_labels = []
max_label_length = 1
for label in list_of_yaml_labels:
    list_of_labels.append(f"{label:15.12}")
    
case_headers = '| Data name       | ' + ' | '.join(list_of_labels) + ' | Units          |'
# Header describing what we are printing:
title_string = "Comparison between WISDEM results from yaml files"
spacing = (len(case_headers) - len(title_string) - 2) // 2
if not spacing % 2:
    extra_space = " "
else:
    extra_space = ""
print("+" + "-" * (len(case_headers) - 2) + "+")
print("|" + " " * spacing + title_string + " " * spacing + " |")
print("+" + "-" * (len(case_headers) - 2) + "+")

print(case_headers)
print("+" + "-" * (len(case_headers) - 2) + "+")

for key in values_to_print:
    name_str = f"| {key}" + (16 - len(key)) * ' ' + "| "
    
    value_name = values_to_print[key][0]
    units = values_to_print[key][1]
    
    list_of_values = []
    for yaml_data in list_of_yamls:
        value = yaml_data.get_val(value_name, units)
        list_of_values.append(f"{float(value):15.5f}")
    values_str = " | ".join(list_of_values)
    
    units_str = f" | {units}" + (15 - len(units)) * ' ' + "|"
    
    print(name_str + values_str + units_str)




print("+" + "-" * (len(case_headers) - 2) + "+")
print()
print()


# Printing and plotting results
print('AEP ' + label1 + ' = ' + str(wt_opt1['sse.AEP']*1.e-6) + ' GWh')
print('AEP ' + label2 + '   = ' + str(wt_opt2['sse.AEP']*1.e-6) + ' GWh')
print('Cp ' + label1 + '  = ' + str(wt_opt1['sse.powercurve.Cp_aero']))
print('Cp ' + label2 + '    = ' + str(wt_opt2['sse.powercurve.Cp_aero']))
print('Design ' + label1 + ': LCOE in $/MWh: '  + str(wt_opt1['financese.lcoe']*1.e3))
print('Design ' + label2 + ': LCOE in $/MWh: '    + str(wt_opt2['financese.lcoe']*1.e3))


print('Design ' + label1 + ': blade mass in kg: ' + str(wt_opt1['elastic.precomp.blade_mass']))
print('Design ' + label1 + ': blade cost in $: '  + str(wt_opt1['elastic.precomp.total_blade_cost']))
print('Design ' + label2 + ': blade mass in kg: '   + str(wt_opt2['elastic.precomp.blade_mass']))
print('Design ' + label2 + ': blade cost in $: '    + str(wt_opt2['elastic.precomp.total_blade_cost']))

print('Design ' + label1 + ': AEP in GWh: ' + str(wt_opt1['sse.AEP']))
print('Design ' + label2 + ': AEP in GWh: ' + str(wt_opt2['sse.AEP']))

print('Design ' + label1 + ': tip deflection ratio: ' + str(wt_opt1['tcons.tip_deflection_ratio']))
print('Design ' + label2 + ': tip deflection ratio: ' + str(wt_opt2['tcons.tip_deflection_ratio']))

print('Design ' + label2 + ': flap and edge blade frequencies in Hz:')
print(wt_opt1['rlds.frame.flap_mode_freqs'])
print(wt_opt1['rlds.frame.edge_mode_freqs'])
print('Design ' + label1 + ': 3P frequency in Hz' + str(3.*wt_opt1['sse.powercurve.rated_Omega']/60.))
print('Design ' + label1 + ': 6P frequency in Hz' + str(6.*wt_opt1['sse.powercurve.rated_Omega']/60.))
print('Design ' + label2 + ': flap and edge blade frequencies in Hz:')
print(wt_opt2['rlds.frame.flap_mode_freqs'])
print(wt_opt2['rlds.frame.edge_mode_freqs'])
print('Design ' + label2 + ': 3P frequency in Hz' + str(3.*wt_opt2['sse.powercurve.rated_Omega']/60.))
print('Design ' + label2 + ': 6P frequency in Hz' + str(6.*wt_opt2['sse.powercurve.rated_Omega']/60.))

print('Design ' + label1 + ': forces at hub: ' + str(wt_opt1['rlds.aero_hub_loads.Fxyz_hub_aero']))
print('Design ' + label1 + ': moments at hub: ' + str(wt_opt1['rlds.aero_hub_loads.Mxyz_hub_aero']))
print('Design ' + label2 + ': forces at hub: ' + str(wt_opt2['rlds.aero_hub_loads.Fxyz_hub_aero']))
print('Design ' + label2 + ': moments at hub: ' + str(wt_opt2['rlds.aero_hub_loads.Mxyz_hub_aero']))


lcoe_data = np.array([[wt_opt1['financese.turbine_number'], wt_opt2['financese.turbine_number']],
[wt_opt1['financese.machine_rating'][0], wt_opt2['financese.machine_rating'][0]],
[wt_opt1['financese.tcc_per_kW'][0], wt_opt2['financese.tcc_per_kW'][0]],
[wt_opt1['financese.bos_per_kW'][0], wt_opt2['financese.bos_per_kW'][0]],
[wt_opt1['financese.opex_per_kW'][0], wt_opt2['financese.opex_per_kW'][0]],
[wt_opt1['financese.turbine_aep'][0]*1.e-6, wt_opt2['financese.turbine_aep'][0]*1.e-6],
[wt_opt1['financese.fixed_charge_rate'][0]*100., wt_opt2['financese.fixed_charge_rate'][0]*100.],
[wt_opt1['financese.lcoe'][0]*1000., wt_opt2['financese.lcoe'][0]*1000.]])

np.savetxt(os.path.join(folder_output, 'lcoe.dat'), lcoe_data)


