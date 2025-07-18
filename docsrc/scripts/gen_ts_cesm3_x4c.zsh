#!/bin/zsh
#
# timeseries to climo conversion
#

export hist_root=/glade/campaign/cesm/development/cross-wg/diagnostic_framework/CESM_output_for_testing
# export ts_root=/glade/campaign/cgd/ppc/fengzhu/CESM_output/timeseries
export ts_root=/glade/derecho/scratch/fengzhu/x4c/gen_ts
export ts_staging=/glade/derecho/scratch/fengzhu/x4c/gen_ts
export casename=b.e30_beta02.BLT1850.ne30_t232.104
export syr=$1
export eyr=$2
export step=10
export task_name=gts
export nnodes=1
export ncpus=128
export compression=1
export overwrite=True
export account=P93300324
export pyenv=x4c-py313

# =====================================================================
# create d case directory
# =====================================================================
mkdir -p ${casename}
cd ${casename}

# =====================================================================
# create Python script
# =====================================================================
cat >! ${task_name}_ocn.h.sfc_${syr}-${eyr}.py << EOF
import os
import x4c
import time

start = time.time()

dirpath = '$hist_root/$casename'
case = x4c.History(dirpath, casename='$casename', hstr_dict={'ocn': ('mom6', ['h.sfc'])}, cesm_ver=3)

output_dirpath = '$ts_root/$casename'
staging_dirpath = '$ts_staging/$casename'
case.gen_ts(
    comps=['ocn'],
    output_dirpath=output_dirpath,
    staging_dirpath=staging_dirpath,
    timespan=($syr, $eyr),
    timestep=$step,
    nproc=$nnodes*$ncpus,
    overwrite=$overwrite,
    compression=$compression,
)

end = time.time()
print(f'Elapsed wall-clock time: {(end-start)/60:.1f} mins')
EOF

# ----------------------------------------------------------------------
cat >! ${task_name}_ocn.h.z_${syr}-${eyr}.py << EOF
import os
import x4c
import time

start = time.time()

dirpath = '$hist_root/$casename'
case = x4c.History(dirpath, casename='$casename', hstr_dict={'ocn': ('mom6', ['h.z'])}, cesm_ver=3)

output_dirpath = '$ts_root/$casename'
staging_dirpath = '$ts_staging/$casename'
case.gen_ts(
    comps=['ocn'],
    output_dirpath=output_dirpath,
    staging_dirpath=staging_dirpath,
    timespan=($syr, $eyr),
    timestep=$step,
    nproc=$nnodes*$ncpus,
    overwrite=$overwrite,
    compression=$compression,
)

end = time.time()
print(f'Elapsed wall-clock time: {(end-start)/60:.1f} mins')
EOF

# ----------------------------------------------------------------------
cat >! ${task_name}_ocn.h.rho2_${syr}-${eyr}.py << EOF
import os
import x4c
import time

start = time.time()

dirpath = '$hist_root/$casename'
case = x4c.History(dirpath, casename='$casename', hstr_dict={'ocn': ('mom6', ['h.rho2'])}, cesm_ver=3)

output_dirpath = '$ts_root/$casename'
staging_dirpath = '$ts_staging/$casename'
case.gen_ts(
    comps=['ocn'],
    output_dirpath=output_dirpath,
    staging_dirpath=staging_dirpath,
    timespan=($syr, $eyr),
    timestep=$step,
    nproc=$nnodes*$ncpus,
    overwrite=$overwrite,
    compression=$compression,
)

end = time.time()
print(f'Elapsed wall-clock time: {(end-start)/60:.1f} mins')
EOF

# ----------------------------------------------------------------------
cat >! ${task_name}_atm_${syr}-${eyr}.py << EOF
import os
import x4c
import time

start = time.time()

dirpath = '$hist_root/$casename'
case = x4c.History(dirpath, casename='$casename', cesm_ver=3)

output_dirpath = '$ts_root/$casename'
staging_dirpath = '$ts_staging/$casename'
case.gen_ts(
    comps=['atm', 'ice'],
    output_dirpath=output_dirpath,
    staging_dirpath=staging_dirpath,
    timespan=($syr, $eyr),
    timestep=$step,
    nproc=$nnodes*$ncpus,
    overwrite=$overwrite,
    compression=$compression,
)

end = time.time()
print(f'Elapsed wall-clock time: {(end-start)/60:.1f} mins')
EOF

# ----------------------------------------------------------------------
cat >! ${task_name}_lnd_${syr}-${eyr}.py << EOF
import os
import x4c
import time

start = time.time()

dirpath = '$hist_root/$casename'
case = x4c.History(dirpath, casename='$casename', cesm_ver=3)

output_dirpath = '$ts_root/$casename'
staging_dirpath = '$ts_staging/$casename'
case.gen_ts(
    comps=['lnd', 'rof'],
    output_dirpath=output_dirpath,
    staging_dirpath=staging_dirpath,
    timespan=($syr, $eyr),
    timestep=$step,
    nproc=$nnodes*$ncpus,
    overwrite=$overwrite,
    compression=$compression,
)

end = time.time()
print(f'Elapsed wall-clock time: {(end-start)/60:.1f} mins')
EOF
# =====================================================================
# load modules
# =====================================================================
cat >! ${task_name}_ocn.h.sfc_${syr}-${eyr}.pbs << EOF
#!/bin/bash
#PBS -N ${task_name}_ocn.h.sfc_${syr}-${eyr}
#PBS -q main
#PBS -l select=$nnodes:ncpus=$ncpus:mpiprocs=1
#PBS -l walltime=12:00:00
#PBS -A ${account}

source $LMOD_ROOT/lmod/init/zsh

module load ncarenv/23.09
module load nco
module load conda
conda activate ${pyenv}

python ${task_name}_ocn.h.sfc_${syr}-${eyr}.py
EOF

# ----------------------------------------------------------------------
cat >! ${task_name}_ocn.h.z_${syr}-${eyr}.pbs << EOF
#!/bin/bash
#PBS -N ${task_name}_ocn.h.z_${syr}-${eyr}
#PBS -q main
#PBS -l select=$nnodes:ncpus=$ncpus:mpiprocs=1
#PBS -l walltime=12:00:00
#PBS -A ${account}

source $LMOD_ROOT/lmod/init/zsh

module load ncarenv/23.09
module load nco
module load conda
conda activate ${pyenv}

python ${task_name}_ocn.h.z_${syr}-${eyr}.py
EOF

# ----------------------------------------------------------------------
cat >! ${task_name}_ocn.h.rho2_${syr}-${eyr}.pbs << EOF
#!/bin/bash
#PBS -N ${task_name}_ocn.h.rho2_${syr}-${eyr}
#PBS -q main
#PBS -l select=$nnodes:ncpus=$ncpus:mpiprocs=1
#PBS -l walltime=12:00:00
#PBS -A ${account}

source $LMOD_ROOT/lmod/init/zsh

module load ncarenv/23.09
module load nco
module load conda
conda activate ${pyenv}

python ${task_name}_ocn.h.rho2_${syr}-${eyr}.py
EOF

# ----------------------------------------------------------------------
cat >! ${task_name}_atm_${syr}-${eyr}.pbs << EOF
#!/bin/bash
#PBS -N ${task_name}_atm_${syr}-${eyr}
#PBS -q main
#PBS -l select=$nnodes:ncpus=$ncpus:mpiprocs=1
#PBS -l walltime=12:00:00
#PBS -A ${account}

source $LMOD_ROOT/lmod/init/zsh

module load ncarenv/23.09
module load nco
module load conda
conda activate ${pyenv}

python ${task_name}_atm_${syr}-${eyr}.py
EOF

# ----------------------------------------------------------------------
cat >! ${task_name}_lnd_${syr}-${eyr}.pbs << EOF
#!/bin/bash
#PBS -N ${task_name}_lnd_${syr}-${eyr}
#PBS -q main
#PBS -l select=$nnodes:ncpus=$ncpus:mpiprocs=1
#PBS -l walltime=12:00:00
#PBS -A ${account}

source $LMOD_ROOT/lmod/init/zsh

module load ncarenv/23.09
module load nco
module load conda
conda activate ${pyenv}

python ${task_name}_lnd_${syr}-${eyr}.py
EOF

# =====================================================================
# submit the job
# =====================================================================
qsub ${task_name}_ocn.h.sfc_${syr}-${eyr}.pbs
qsub ${task_name}_ocn.h.z_${syr}-${eyr}.pbs
qsub ${task_name}_ocn.h.rho2_${syr}-${eyr}.pbs
qsub ${task_name}_atm_${syr}-${eyr}.pbs
qsub ${task_name}_lnd_${syr}-${eyr}.pbs

exit