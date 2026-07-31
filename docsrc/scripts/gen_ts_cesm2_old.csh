#!/bin/csh -v
#
# Test case:
# /glade/campaign/cesm/development/cross-wg/diagnostic_framework/CESM_output_for_testing/b.e23_alpha17f.BLT1850.ne30_t232.092

# =====================================================================
# load modules
# =====================================================================
source $LMOD_ROOT/lmod/init/tcsh
module swap ncarenv ncarenv/23.09
module use /glade/work/bdobbins/Software/Modules
module load cesm_postprocessing_derecho

set case = b.e23_alpha17f.BLT1850.ne30_t232.092
set tslices_dir = /glade/campaign/cesm/development/cross-wg/diagnostic_framework/CESM_output_for_testing
set tseries_dir = /glade/derecho/scratch/fengzhu/CESM_postprocess/timeseries

# =====================================================================
# create case and change xml settings 
# =====================================================================
set scripts_dir = /glade/derecho/scratch/fengzhu/CESM_postprocess/scripts
if ( -d $scripts_dir/$case ) then
   rm -rf $scripts_dir/$case
endif

create_postprocess -caseroot=$scripts_dir/$case

cd $scripts_dir/$case
# /bin/cp /glade/u/home/fengzhu/Github/x4c/docsrc/scripts/xmls/env_timeseries_atm_h0.xml env_timeseries.xml

pp_config --set CASE=$case
pp_config --set DOUT_S_ROOT=$tslices_dir/$case
pp_config --set TIMESERIES_OUTPUT_ROOTDIR=$tseries_dir/$case

# =====================================================================
# write the pbs job file
# =====================================================================
set today   = `date '+%Y%m%d-%H%M%S'`
set log_file = $scripts_dir/$case/logs/timeseries.log.$today

cat >! gen_ts_cesm2.pbs << EOF
#!/bin/bash
#PBS -N gts_cesm2
#PBS -q main
#PBS -l select=16:ncpus=64:mpiprocs=16
#PBS -l walltime=12:00:00
#PBS -l job_priority=premium
#PBS -A P93300324

source $LMOD_ROOT/lmod/init/bash
module load ncarenv/23.09
module load intel
module load intel-mpi/2021.10.0
module load apptainer

export I_MPI_HYDRA_BRANCH_COUNT=36

mpiexec singularity run -B /glade,/var \
    /glade/work/bdobbins/Containers/CESM_Postprocessing/image \
    /opt/ncar/conda/envs/cesm-env2/bin/cesm_tseries_generator.py \
    --caseroot $scripts_dir/$case >> ${log_file} 2>&1
EOF

# =====================================================================
# submit the job
# =====================================================================
qsub gen_ts_cesm2.pbs

exit