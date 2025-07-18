import os
import x4c
import time

start = time.time()

dirpath = '/glade/campaign/cesm/development/cross-wg/diagnostic_framework/CESM_output_for_testing/b.e30_beta02.BLT1850.ne30_t232.104'
case = x4c.History(
    dirpath,
    casename='b.e30_beta02.BLT1850.ne30_t232.104',
    comps_info={},
    cesm_ver=3
)

output_dirpath = '/glade/derecho/scratch/fengzhu/x4c/gen_ts/b.e30_beta02.BLT1850.ne30_t232.104'
staging_dirpath = '/glade/derecho/scratch/fengzhu/x4c/gen_ts/b.e30_beta02.BLT1850.ne30_t232.104'
case.gen_ts(
    comps=['lnd', 'rof'],
    output_dirpath=output_dirpath,
    staging_dirpath=staging_dirpath,
    timespan=(1, 10),
    timestep=10,
    nproc=128,
    overwrite=True,
    compression=1,
)

end = time.time()
print(f'Elapsed wall-clock time: {(end-start)/60:.1f} mins')
