#!/bin/zsh

export hist_root=/glade/derecho/scratch/cmip7/archive/
export ts_root=/glade/campaign/cesm/development/cross-wg/diagnostic_framework/x4c/timeseries
export ts_staging=/glade/derecho/scratch/fengzhu/x4c/gen_ts
export casename=b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1
export casefolder=b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1
export syr=$1
export eyr=$2
export timestep=10
export timestep_unit=year
export task_name=gts
export overwrite=True
export account=P93300324
export pyenv=x4c-py313

# =====================================================================
# create d case directory
# =====================================================================
mkdir -p ${casefolder}
cd ${casefolder}

# =====================================================================
# define functions
# =====================================================================
gen_py_script() {
  local name=$1
  local comps=$2
  local comps_info=$3
  local ncpus=$4

  cat >! ${task_name}_${name}_${syr}-${eyr}.py << EOF
import os
import x4c
import time

start = time.time()

dirpath = '$hist_root/$casefolder'
case = x4c.History(dirpath, comps=$comps, comps_info=${comps_info}, casename='$casename')

output_dirpath = '$ts_root/$casefolder'
staging_dirpath = '$ts_staging/$casefolder'
case.gen_ts(
    comps=$comps,
    output_dirpath=output_dirpath,
    staging_dirpath=staging_dirpath,
    timespan=($syr, $eyr),
    timestep=$timestep,
    timestep_unit='$timestep_unit',
    nproc=$ncpus,
    overwrite=$overwrite,
)

end = time.time()
print(f'Elapsed wall-clock time: {(end-start)/60:.1f} mins')
EOF
}

gen_pbs_script() {
  local name=$1
  local nnodes=$2
  local ncpus=$3

  cat >! ${task_name}_${name}_${syr}-${eyr}.pbs << EOF
#!/bin/bash
#PBS -N ${task_name}_${name}_${syr}-${eyr}
#PBS -q main
#PBS -l select=$nnodes:ncpus=$ncpus:mpiprocs=1
#PBS -l walltime=12:00:00
#PBS -A ${account}

source \$LMOD_ROOT/lmod/init/zsh

module load ncarenv/23.09
module load nco
module load conda
conda activate ${pyenv}

mpiexec -n $nnodes python ${task_name}_${name}_${syr}-${eyr}.py
EOF
}

# =====================================================================
# call functions
# =====================================================================
# Define task entries: name|components|comps_info|nnodes|ncpus
task_list=(
  "o.sz|['ocn']|{'ocn': ['mom6.h.sfc', 'mom6.h.z']}|1|128"
  "o.nr+r|['ocn', 'rof']|{'ocn': ['mom6.h.native', 'mom6.h.rho2']}|1|128"
  "ail|['atm', 'ice', 'lnd']|{'atm': ['cam.h0a', 'cam.h1a', 'cam.h2a'],'ice': ['cice.h']}|1|128"
  "i.h1|['ice']|{'ice': ['cice.h1']}|13|128"
)

for entry in "${task_list[@]}"; do
  IFS='|' read -r name comps comps_info nnodes ncpus <<< "$entry"

  gen_py_script "$name" "$comps" "${comps_info}" "$ncpus"
  gen_pbs_script "$name" "$nnodes" "$ncpus" 
  qsub "${task_name}_${name}_${syr}-${eyr}.pbs"
done

exit