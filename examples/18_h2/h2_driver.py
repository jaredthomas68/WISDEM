import os

from wisdem.glue_code.runWISDEM import run_wisdem

## File management
run_dir = os.path.dirname(os.path.realpath(__file__))
fname_wt_input = run_dir + os.sep + "IEA-15-240-RWT_VolturnUS-S.yaml"
fname_modeling_options = run_dir + os.sep + "modeling_options.yaml"
fname_analysis_options = run_dir + os.sep + "analysis_options.yaml"

wt_opt, modeling_options, opt_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

wt_opt.model.list_outputs(includes=["*h2*"], units=True, print_arrays=True)
