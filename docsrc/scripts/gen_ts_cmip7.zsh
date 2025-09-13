#!/bin/zsh

export hist_root=/glade/derecho/scratch/cmip7/archive/
export ts_root=/glade/campaign/cesm/development/cross-wg/diagnostic_framework/x4c/timeseries
export ts_staging=/glade/derecho/scratch/fengzhu/x4c/gen_ts
export casename=b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1
export casefolder=b.e30_beta06.B1850C_LTso.ne30_t232_wgx3.192.wrkflw.1_32
export syr=$1   # e.g., 0001: model year 1 
export eyr=$2   # e.g., 0100: model year 100
export timestep=10
export timestep_unit=year
export task_name=gts
export nnodes=1
export ncpus=128
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
    timespan=('$syr', '$eyr'),
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

python ${task_name}_${name}_${syr}-${eyr}.py
EOF
}

# =====================================================================
# call functions
# =====================================================================
# Define task entries: name|components|comps_info
# task_list=(
#   "o.sfc|['ocn']|{'ocn': ['mom6.h.sfc']}"
#   "o.z|['ocn']|{'ocn': ['mom6.h.z']}"
#   "o.rho2|['ocn']|{'ocn': ['mom6.h.rho2']}"
#   "o.native|['ocn']|{'ocn': ['mom6.h.native']}"
#   "a.h0a|['atm']|{'atm': ['cam.h0a']}"
#   "a.h1a|['atm']|{'atm': ['cam.h1a']}"
#   "a.h2a|['atm']|{'atm': ['cam.h2a']}"
#   "l.h0|['lnd']|{'lnd': ['clm2.h0']}"
#   "i.h|['ice']|{'ice': ['cice.h']}"
#   "i.h1|['ice']|{'ice': ['cice.h1']}"
#   "r.h0|['rof']|{'rof': ['mosart.h0']}"
# )
task_list=(
  "i.h1|['ice']|{'ice': ['cice.h1']}"
#   "aolir|['atm', 'ocn', 'lnd', 'ice', 'rof']|{'atm': ['cam.h0a', 'cam.h1a', 'cam.h2a'], 'lnd': ['clm2.h0'], 'ice': ['cice.h'], 'rof': ['mosart.h0']}"
)
# task_list=(
#   "aolir|['atm', 'ocn', 'lnd', 'ice', 'rof']|{'atm': ['cam.h0a', 'cam.h1a', 'cam.h2a']}"
# )

for entry in "${task_list[@]}"; do
  IFS='|' read -r name comps comps_info <<< "$entry"

  gen_py_script "$name" "$comps" "${comps_info}"
  gen_pbs_script "$name"
  qsub "${task_name}_${name}_${syr}-${eyr}.pbs"
done

exit