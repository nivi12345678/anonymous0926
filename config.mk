export DESIGN_NICKNAME = softmax_pipe_deep2
export DESIGN_NAME = softmax_pipe_deep2
export PLATFORM    = asap7

export VERILOG_FILES = $(DESIGN_HOME)/src/$(DESIGN_NICKNAME)/softmax_pipe_deep2.v

export SDC_FILE      = $(DESIGN_HOME)/$(PLATFORM)/$(DESIGN_NICKNAME)/constraint.sdc

#export DIE_AREA = 0 0 100 100
#export CORE_AREA = 10 10 100 100
export CORE_UTILIZATION = 45
export PLACE_DENSITY = 0.60
export PLACE_DENSITY_LB_ADDON = 0.3
export TNS_END_PERCENT        = 100
export ABC_AREA = 0
export REMOVE_CELLS_FOR_EQY   = TAPCELL* FILLER* SPARE*
