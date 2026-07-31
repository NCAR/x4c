import os
import glob
from collections import defaultdict
import pandas as pd
import gzip
from tqdm import tqdm
import xarray as xr
import multiprocessing as mp
import pathlib
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import cftime
from . import visual
import subprocess
from copy import deepcopy

from . import core, utils, diags
from .spell import Spell


class _star:
    ''' Adapt a function of many args for `imap_unordered`, which passes one

    A module-level class rather than a lambda or closure because the wrapped callable
    has to survive pickling to the worker processes.
    '''
    def __init__(self, func):
        self.func = func

    def __call__(self, args):
        return self.func(*args)


def _get_mpi_comm():
    ''' Import mpi4py lazily and return (comm, rank, size)

    Deliberately not a module-level import: `from mpi4py import MPI` initializes the
    MPI runtime for *every* user of x4c, including notebook sessions that only ever
    touch `Timeseries`. That can emit warnings, interact badly with forked worker
    processes, and makes an MPI stack a hard requirement for pure analysis work.
    Only `bigbang`/`bigcrunch`/`gen_ts` actually need it.
    '''
    try:
        from mpi4py import MPI
    except ImportError as e:
        raise ImportError(
            'mpi4py is required for the parallel timeseries generation methods '
            '(`bigbang`, `bigcrunch`, `gen_ts`). Install it via '
            '`conda install -c conda-forge mpi4py`.'
        ) from e

    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()

class History:
    '''Handle CESM history files for a single case.

    Provides utilities to discover history file paths, list time-series
    variables, split (isolate) variables into separate files, and
    re-merge them across time ranges. Designed to work with NCO tools
    and MPI for parallel operations.
    '''
    def __init__(self, root_dir, comps=['atm', 'ocn', 'lnd', 'ice', 'rof'], comps_info=None, casename=None,
                 path_pattern='comp/hist/casename.hstr.date.nc', avoid_list=None):
        '''Initialize History with root directory and component settings.

        Args:
            root_dir (str): Path to the case root directory.
            comps (list): Components to consider (atm, ocn, lnd, etc.).
            comps_info (dict): Optional mapping of component->hstr patterns.
            casename (str): Optional case name override.
            path_pattern (str): Glob-like pattern for history files.
            avoid_list (list): List of substrings to avoid when searching.
        '''
        self.path_pattern = path_pattern
        self.root_dir = root_dir
        self.casename = os.path.basename(root_dir) if casename is None else casename

        self.avoid_list = ['once']
        if avoid_list is not None: self.avoid_list.extend(avoid_list)

        utils.p_header(f'>>> case.root_dir: {self.root_dir}')
        utils.p_header(f'>>> case.casename: {self.casename}')

        _comps_info = {
            'atm': '*',
            'ocn': '*',
            'lnd': '*',
            'ice': '*',
            'rof': '*',
        }
        if comps_info is not None: _comps_info.update(comps_info)

        self.comps_info = {}
        self.paths = {}
        for comp in comps:
            # mdl, hstr = _comps_info[comp]
            hstr = _comps_info[comp]
            self.paths[comp] = {}

            if hstr == '*':
                paths = utils.find_paths(
                    self.root_dir, self.path_pattern,
                    comp=comp, hstr=hstr,
                    avoid_list=self.avoid_list,
                )
                hstr = utils.get_hstr(paths, casename=self.casename)
            elif isinstance(hstr, str):
                # a single hstr given as a plain string, e.g. `comps_info={'atm': 'h0a'}`;
                # wrap it so the loops below iterate over hstrs rather than characters
                hstr = [hstr]

            self.comps_info[comp] = hstr

            for hs in hstr:
                self.paths[comp][hs] = utils.find_paths(
                    self.root_dir, self.path_pattern,
                    comp=comp, hstr=hs,
                    avoid_list=self.avoid_list,  # not the bare arg: keeps the 'once' default applied
                )
                utils.p_success(f'>>> case.paths["{comp}"]["{hs}"] created')

        self.vns = {}
        for comp in comps:
            # mdl, hstr = self.comps_info[comp]
            hstr = self.comps_info[comp]
            self.vns[comp] = {}
            for hs in hstr:
                self.vns[comp][hs] = self.get_ts_vns(comp, hs)
                utils.p_success(f'>>> case.vns["{comp}"]["{hs}"] created')

    def get_ts_vns(self, comp, hstr, exclude_vars=[
            'time', 'time_bnds', 'time_bounds', 'time_bound',
            'time_written', 'date', 'datesec', 'date_written',
        ]):
        '''
        Return list of time-varying variable names for a given component
        and hstr by inspecting the first history file.
        '''
        vns_ts = []
        ds0 = core.open_dataset(self.paths[comp][hstr][0])
        vns = list(ds0.variables)

        for v in vns:
            # if len(ds0[v].dims) >= 2 and 'time' in ds0[v].dims and 'time' not in v and v not in exclude_vars:
            if len(ds0[v].dims) >= 2 and 'time' in ds0[v].dims and v not in exclude_vars:
                vns_ts.append(v)
        
        vns_ts = sorted(vns_ts)

        ds0.close()
        return vns_ts


    def get_paths(self, comp, hstr, timespan=None):
        '''Return history file paths for a component/hstr optionally
        filtered by a timespan.

        timespan may be provided in a variety of formats accepted by
        utils.parse_timespan.
        '''
        paths = self.paths[comp][hstr]

        if timespan is None:
            paths_sub = paths
        else:
            start_dt, end_dt, timespan_precision = utils.parse_timespan(timespan)
            start_dt = utils.datetime_truncate(start_dt, timespan_precision)
            end_dt = utils.datetime_truncate(end_dt, timespan_precision)
            paths_sub = []
            for path in paths:
                date = path.split('.')[-2]
                dt = utils.cesm_str2datetime(date)
                dt = utils.datetime_truncate(dt, timespan_precision)
                if start_dt <= dt <= end_dt:
                    paths_sub.append(path)

        return paths_sub

    def isolate_vn(self, vn, comp, hstr, in_path, output_dirpath, overwrite=True):
        '''Create a new netCDF file containing only variable `vn` from
        the input history file `in_path`.

        Uses `ncks` to drop other variables and writes result to
        `output_dirpath` with a standardized filename.
        '''
        bn_elements = os.path.basename(in_path).split('.')
        bn_elements.insert(-2, vn)
        # print(bn_elements)
        date_str = bn_elements[-2]

        if self.casename is not None:
            fname = f'{self.casename}.{hstr}.{vn}.{date_str}.nc'
        else:
            fname = '.'.join(bn_elements)

        out_path = os.path.join(output_dirpath, fname)
        if overwrite or not os.path.exists(out_path):
            if os.path.exists(out_path): os.remove(out_path)
            vns = self.vns[comp][hstr].copy()
            vns.remove(vn)
            cmd = [
                'ncks', '-h', '-C', '-x',
                '-v', ','.join(vns),
                in_path,
                '-o', out_path
            ]
            # print(cmd)
            subprocess.run(cmd, check=True)

    def bigbang(self, comp, hstr, output_dirpath, timespan=None, overwrite=True, nproc=1, vns=None):
        '''Split history files into per-variable files in parallel using MPI.

        Each MPI rank handles a subset of (file,variable) tasks.
        '''
        comm, rank, size = _get_mpi_comm()

        output_dirpath = pathlib.Path(output_dirpath)
        if rank == 0: output_dirpath.mkdir(parents=True, exist_ok=True)
        comm.Barrier()

        paths = self.get_paths(comp, hstr, timespan=timespan)
        if vns is None: vns = self.vns[comp][hstr]
        arg_list = [(vn, comp, hstr, path, output_dirpath, overwrite) for path in paths for vn in vns]
        tasks = arg_list[rank::size]

        desc = f'[Rank {rank}] Spliting {len(paths)} history files for {len(vns)} variables'
        if nproc <= 1:
            for arg in tqdm(tasks, total=len(tasks), desc=desc):
                self.isolate_vn(*arg)
        else:
            with mp.Pool(processes=nproc) as p:
                # imap_unordered, not starmap: starmap consumes the whole iterable up
                # front, so a tqdm wrapped around it reports 100% before any work runs
                for _ in tqdm(p.imap_unordered(_star(self.isolate_vn), tasks),
                              total=len(tasks), desc=desc):
                    pass
    
    def get_hstr_based_on_vn(self, vn):
        '''Return the first hstr that contains variable `vn`.

        This searches across all components and hstrs and returns the
        matching hstr string or None if not found.
        '''
        for comp, hstrs in self.vns.items():
            for hstr, vns in hstrs.items():
                if vn in vns:
                    return hstr

    def merge_vn(self, hstr, vn, input_dirpath, output_dirpath, timespan=None, overwrite=True, compression=1):
        '''Concatenate per-variable files across time into a single file.

        Uses `ncrcat` with optional compression level to produce an
        aggregated timeseries file for `vn` and `hstr`.
        '''
        paths = sorted(glob.glob(os.path.join(input_dirpath, f'*.{hstr}.{vn}.*.nc')))
        if timespan is None:
            paths_sub = paths
        else:
            start_dt, end_dt, timespan_precision = utils.parse_timespan(timespan)
            start_dt = utils.datetime_truncate(start_dt, timespan_precision)
            end_dt = utils.datetime_truncate(end_dt, timespan_precision)
            paths_sub = []
            for path in paths:
                date = path.split('.')[-2]
                dt = utils.cesm_str2datetime(date)
                dt = utils.datetime_truncate(dt, timespan_precision)
                if start_dt <= dt <= end_dt:
                    paths_sub.append(path)

        date_start = ''.join(paths_sub[0].split('.')[-2].split('-'))
        date_end = ''.join(paths_sub[-1].split('.')[-2].split('-'))

        bn_elements = os.path.basename(paths_sub[0]).split('.')
        bn_elements[-2] = f'{date_start}-{date_end}'
        date_str = bn_elements[-2]

        if self.casename is not None:
            fname = f'{self.casename}.{hstr}.{vn}.{date_str}.nc'
        else:
            fname = '.'.join(bn_elements)
        out_path = os.path.join(output_dirpath, fname)

        if overwrite or not os.path.exists(out_path):
            if os.path.exists(out_path): os.remove(out_path)
            cmd = [
                'ncrcat', '-O', '-4', '-h', '--no_cll_mth',
                '-L', str(compression),
                *paths_sub,
                '-o', out_path
            ]
            # print(cmd)
            subprocess.run(cmd, check=True)

    def bigcrunch(self, comp, hstr, input_dirpath, output_dirpath, timespan=None, overwrite=True, nproc=1, compression=1, vns=None):
        '''Merge per-variable files back into timeseries files in parallel.

        Coordinates work across MPI ranks similar to `bigbang`.
        '''
        comm, rank, size = _get_mpi_comm()

        output_dirpath = pathlib.Path(output_dirpath)
        if rank == 0: output_dirpath.mkdir(parents=True, exist_ok=True)
        comm.Barrier()

        if vns is None: vns = self.vns[comp][hstr]
        arg_list = [(hstr, vn, input_dirpath, output_dirpath, timespan, overwrite, compression) for vn in vns]
        tasks = arg_list[rank::size]

        desc = f'[Rank {rank}] Merging variables'
        if nproc <= 1:
            for arg in tqdm(tasks, total=len(tasks), desc=desc):
                self.merge_vn(*arg)
        else:
            with mp.Pool(processes=nproc) as p:
                for _ in tqdm(p.imap_unordered(_star(self.merge_vn), tasks),
                              total=len(tasks), desc=desc):
                    pass

    def gen_ts(self, output_dirpath, staging_dirpath=None, comps=['atm', 'ocn', 'lnd', 'ice', 'rof'],
               timespan=None, timestep=None, timestep_unit='year',
               dir_structure='comp/proc/tseries/hstr' , overwrite=True, nproc=1, compression=1):
        '''Generate timeseries files for selected components and timespans.

        This orchestrates splitting (`bigbang`) and merging
        (`bigcrunch`) stages and moves results from staging to final
        output directories.
        '''
        comm, rank, size = _get_mpi_comm()

        if staging_dirpath is None: staging_dirpath = output_dirpath
        pathlib.Path(staging_dirpath).mkdir(parents=True, exist_ok=True)
        if timespan is None:
            raise ValueError('Please specify `timespan`.')
        if timestep is None:
            # otherwise this surfaces much later as `TypeError: unsupported operand
            # type(s) for *: 'NoneType' and 'int'` inside parse_timestamps
            raise ValueError(
                'Please specify `timestep` (the chunk length of each output file, in '
                f'units of `timestep_unit={timestep_unit!r}`), e.g. `timestep=10`.'
            )
        # element-wise, so a mixed tuple like ('0001-01', 20) is handled correctly
        timespan = utils.normalize_timespan(timespan)

        timespan_list = utils.parse_timestamps(timespan, timestep=timestep, timestep_unit=timestep_unit)
        if not isinstance(comps, dict): comps = {comp: None for comp in comps}

        move_tasks, clean_tasks = [], []
        for comp, vns in comps.items():
            hstr = self.comps_info[comp]
            # generate timeseries files for each component and each sub-timespan
            utils.p_header(f'>>> Processing component: {comp}')
            for hs in hstr:
                if vns is None:
                    vns_in = self.vns[comp][hs]
                else:
                    vns_in = list(set(self.vns[comp][hs]) & set(vns))
                if len(vns_in) == 0: continue

                utils.p_header(f'>>> Processing hstr: {hs}')
                for timespan_tmp in timespan_list:
                    utils.p_header(f'>>> Processing timespan: {timespan_tmp}')
                    bigbang_dir = os.path.join(staging_dirpath, f'.bigbang_{comp}.{hs}.{timespan_tmp[0]}-{timespan_tmp[1]}')
                    if rank == 0 and os.path.exists(bigbang_dir): shutil.rmtree(bigbang_dir)
                    comm.Barrier()

                    self.bigbang(
                        comp=comp, hstr=hs,
                        output_dirpath=bigbang_dir,
                        timespan=timespan_tmp,
                        overwrite=overwrite,
                        nproc=nproc,
                        vns=vns_in,
                    )
                    comm.Barrier()

                    bigcrunch_dir = os.path.join(staging_dirpath, dir_structure.replace('comp', comp).replace('hstr', hs))
                    self.bigcrunch(
                        comp=comp, hstr=hs,
                        input_dirpath=bigbang_dir,
                        output_dirpath=bigcrunch_dir,
                        timespan=timespan_tmp,
                        overwrite=overwrite,
                        nproc=nproc,
                        compression=compression,
                        vns=vns_in,
                    )
                    comm.Barrier()

                    if rank == 0: clean_tasks.append(bigbang_dir)
                    if rank == 0 and staging_dirpath != output_dirpath:
                        dst_dir = os.path.join(output_dirpath, dir_structure.replace('comp', comp).replace('hstr', hs))
                        move_tasks.append((bigcrunch_dir, dst_dir, timespan_tmp))


        move_tasks = comm.bcast(move_tasks if rank == 0 else None, root=0)
        for i, (bigcrunch_dir, dst_dir, timespan_tmp) in enumerate(move_tasks):
            if i % size != rank: continue
            pathlib.Path(dst_dir).mkdir(parents=True, exist_ok=True)
            date_start = ''.join(timespan_tmp[0].split('-'))
            date_end = ''.join(timespan_tmp[1].split('-'))
            date_str = f'{date_start}*-{date_end}*'
            src_paths = glob.glob(os.path.join(bigcrunch_dir, f'*.{date_str}.nc'))
            if src_paths:
                with mp.Pool(processes=nproc) as pool:
                    arg_list = [(src_path, dst_dir) for src_path in src_paths]
                    for _ in tqdm(
                        pool.imap_unordered(_star(utils.move_and_overwrite), arg_list),
                        total=len(arg_list),
                        desc=f'[Rank {rank}] Moving files from {bigcrunch_dir} to {dst_dir}',
                    ):
                        pass
        comm.Barrier()

        clean_tasks = comm.bcast(clean_tasks if rank == 0 else None, root=0)
        for i, bb_dir in enumerate(clean_tasks):
            if i % size != rank: continue
            if os.path.exists(bb_dir):
                try:
                    shutil.rmtree(bb_dir)
                    print(f'[Rank {rank}] Removed {bb_dir}')
                except Exception as e:
                    print(f'[Rank {rank}] Warning: failed to remove {bb_dir}: {e}')

        comm.Barrier()




    def find_timespan_files(self, timespan, comps=['atm', 'ice', 'ocn', 'rof', 'lnd']):
        ''' List the history files within a timespan, without touching them

        Resolves the glob in Python rather than through a shell, so no part of
        `root_dir` is ever interpreted as shell syntax.

        Args:
            timespan (tuple or list): [start_year, end_year], inclusive, integers
            comps (list): components to search

        Returns:
            list of str: matching paths, sorted
        '''
        start_year, end_year = timespan
        paths = []
        for y in range(start_year, end_year + 1):
            for comp in comps:
                pattern = os.path.join(
                    self.root_dir, comp, 'hist', f'*{y:04d}-[01][0-9][-.]*',
                )
                paths.extend(glob.glob(pattern))
        return sorted(set(paths))

    def rm_timespan(self, timespan, comps=['atm', 'ice', 'ocn', 'rof', 'lnd'],
                    nworkers=None, rehearsal=True):
        ''' Delete the archived history files within a timespan

        This is the one destructive operation in x4c, so it is deliberately
        conservative:

        - `rehearsal=True` (the default) only *reports* what would be deleted.
        - The file list is resolved with `glob` and removed with `os.remove`, rather
          than interpolated into a `rm -f ... shell=True` command. A `root_dir`
          containing a space or a shell metacharacter used to change which files were
          deleted.
        - The exact list is printed before anything is removed.

        Args:
            timespan (tuple or list): [start_year, end_year], inclusive, integers
            comps (list): components to clean
            nworkers (int): parallel workers for the deletion (default: 8)
            rehearsal (bool): if True, only list the files; nothing is deleted

        Returns:
            list of str: the paths that were (or would be) removed
        '''
        # the old default was `threading.active_count()`, which is however many
        # threads happen to be alive -- not a meaningful degree of parallelism
        if nworkers is None: nworkers = 8

        paths = self.find_timespan_files(timespan, comps=comps)

        if len(paths) == 0:
            utils.p_warning(f'>>> No history files found for timespan {timespan} in {comps}.')
            return []

        if rehearsal:
            utils.p_header(f'>>> [rehearsal] {len(paths)} files would be removed:')
            for p in paths:
                print(p)
            utils.p_hint('>>> Nothing was deleted. Re-run with `rehearsal=False` to remove them.')
            return paths

        utils.p_warning(f'>>> Removing {len(paths)} files ...')

        def rm_path(path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass  # already gone; nothing to do

        with tqdm(desc='Removing files', total=len(paths)) as pbar:
            with ThreadPoolExecutor(nworkers) as exe:
                futures = [exe.submit(rm_path, p) for p in paths]
                for future in as_completed(futures):
                    future.result()  # surface any unexpected error
                    pbar.update(1)

        utils.p_success(f'>>> {len(paths)} files removed.')
        return paths



class Timeseries:
    '''CESM Timeseries case helper.

    Manages discovery and loading of preprocessed CESM timeseries files
    produced by CESM postprocessing. Provides convenience methods to
    locate paths, load raw or derived diagnostics, compute spells, and
    create plots and seasonal means.
    '''
    def __init__(self, root_dir, grid_dict=None, casename=None, cesm_ver=3):
        '''Initialize a Timeseries instance and index available files.

        Populates `self.paths`, `self.vns`, and basic metadata used by
        other helper methods.

        Args:
            root_dir (str): root directory of the CESM timeseries output.
            grid_dict (dict): optional mapping of component->grid names.
            casename (str): optional case name override.
            cesm_ver (int): CESM version (affects time adjustments).
        '''
        self.path_pattern='comp/proc/tseries/*/casename.hstr.vn.timespan.nc'
        self.root_dir = os.path.abspath(root_dir)
        self.casename = os.path.basename(root_dir) if casename is None else casename
        self.cesm_ver = cesm_ver

        self.grid_dict = {'atm': 'ne30pg3', 'ocn': 'g16'}
        if grid_dict is not None:
            self.grid_dict.update(grid_dict)

        self.grid_dict['lnd'] = self.grid_dict['atm']
        self.grid_dict['rof'] = self.grid_dict['atm']
        self.grid_dict['ice'] = self.grid_dict['ocn']

        utils.p_header(f'>>> case.root_dir: {self.root_dir}')
        utils.p_header(f'>>> case.path_pattern: {self.path_pattern}')
        utils.p_header(f'>>> case.grid_dict: {self.grid_dict}')
        if self.casename is not None: utils.p_header(f'>>> case.casename: {self.casename}')

        self.paths_all = utils.find_paths(self.root_dir, self.path_pattern)
        self.hstr_all = list(set('.'.join(s.split('.')[:-1]) for s in utils.get_hstr(self.paths_all, casename=self.casename)))

        self.ds = {}
        self.diags = {}

        # one pass, not two: the original walked `paths_all` twice, the first time only
        # to create empty containers. `defaultdict` collapses that.
        paths = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        vns = defaultdict(lambda: defaultdict(set))

        for path in self.paths_all:
            parsed = self._parse_ts_path(path)
            if parsed is None:
                continue
            comp, hstr, vn = parsed
            paths[comp][hstr][vn].append(path)
            # a set, so a variable split across N timespan files is listed once rather
            # than N times
            vns[comp][hstr].add(vn)

        # freeze back to plain dicts so `case.paths['atm']['nope']` raises KeyError
        # instead of silently materializing an empty entry
        self.paths = {c: {h: dict(v) for h, v in hs.items()} for c, hs in paths.items()}
        self.vns = {c: {h: sorted(v) for h, v in hs.items()} for c, hs in vns.items()}

        for comp in self.paths:
            for hstr in self.paths[comp]:
                utils.p_success(f'>>> case.paths["{comp}"]["{hstr}"] created')

        for comp in self.vns:
            for hstr in self.vns[comp]:
                utils.p_success(f'>>> case.vns["{comp}"]["{hstr}"] created')

    def _parse_ts_path(self, path):
        ''' Pull (comp, hstr, vn) out of a timeseries path

        The layout is positional -- ``<root>/<comp>/proc/tseries/<freq>/<case>.<hstr>.<vn>.<timespan>.nc``
        -- so validate rather than trust: an unexpected depth or a filename that does
        not contain the casename used to yield plausible-looking garbage (a directory
        name as `comp`, a mangled `hstr`) with no indication anything was wrong.

        Returns:
            tuple or None: (comp, hstr, vn), or None if the path does not match, in
            which case a warning names the file.
        '''
        rel = os.path.relpath(path, self.root_dir)
        parts = rel.split(os.sep)
        fname = parts[-1]

        # comp/proc/tseries/<freq>/<file>
        if len(parts) != 5 or parts[1] != 'proc' or parts[2] != 'tseries':
            utils.p_warning(f'>>> Skipping unexpected timeseries path layout: {rel}')
            return None

        comp = parts[0]
        elements = fname.split('.')
        if len(elements) < 4:
            utils.p_warning(f'>>> Skipping unparseable timeseries filename: {fname}')
            return None

        vn = elements[-3]
        casename_hstr = fname.split(f'.{vn}.')[0]
        if not casename_hstr.startswith(self.casename):
            utils.p_warning(
                f'>>> Skipping {fname}: does not start with casename `{self.casename}`.'
            )
            return None

        hstr = casename_hstr[len(self.casename):].lstrip('.')
        if hstr == '':
            utils.p_warning(f'>>> Skipping {fname}: could not determine the history stream.')
            return None

        return comp, hstr, vn


    def get_paths(self, comp, hstr, vn, timespan=None):
        '''Return list of timeseries file paths for `vn` under `comp/hstr`.

        If `timespan` is provided it filters the returned paths to those
        fully covering the requested interval.
        '''
        if vn in self.paths[comp][hstr]:
            paths = self.paths[comp][hstr][vn]
            if timespan is None:
                paths_sub = paths
            else:
                start_dt, end_dt, timespan_precision = utils.parse_timespan(timespan)
                paths_sub = []
                for path in paths:
                    start_str, end_str = path.split('.')[-2].split('-')
                    start_str = utils.add_dash_to_timestamp(start_str)
                    end_str = utils.add_dash_to_timestamp(end_str)
                    timespan_tmp = (start_str, end_str)
                    start, end, _ = utils.parse_timespan(timespan_tmp)
                    start = utils.datetime_truncate(start, timespan_precision)
                    end = utils.datetime_truncate(end, timespan_precision)
                    if start_dt <= start and end <= end_dt:
                        paths_sub.append(path)

            return paths_sub
        else:
            # empty rather than None, so callers can `len()` it and report
            # "no files found" instead of raising `TypeError`
            return []

    def get_comp_hstr(self, vn):
        '''Find all (component, hstr) pairs where `vn` is present.'''
        found_comp_hstr = []
        for k, v in self.vns.items():
            comp = k
            for hstr, vns in v.items():
                if vn in vns:
                    found_comp_hstr.append((comp, hstr))

        return found_comp_hstr

    
    def load(self, vn, vtype=None, comp=None, hstr=None, timespan=None, load_idx=-1, verbose=True, reload=False, **kws):
        '''Load a variable or derived diagnostic into `self.ds`.

        Automatically detects whether `vn` is a raw timeseries or a
        derived diagnostic and loads or computes it. Results are stored
        in `self.ds[vn]`.
        '''
        shift_time = True if self.cesm_ver == 1 else False

        if vtype is None:
            vtype = 'derived' if vn in diags.Registry.funcs else 'raw'


        if reload: self.clear_ds(vn)

        if vtype == 'raw':
            found_comp_hstr = self.get_comp_hstr(vn)
            if len(found_comp_hstr) == 0:
                raise ValueError(f'The input variable name `{vn}` is unknown.')
            elif len(found_comp_hstr) == 1:
                comp, hstr = found_comp_hstr[0]
            else:
                if comp is None or hstr is None:
                    raise ValueError(f'The input variable name belongs to multiple (comp, hstr) pairs: {found_comp_hstr}. Please specify via the argument `comp` and `hstr`.')

            timespan = utils.normalize_timespan(timespan)

            paths = self.get_paths(comp, hstr, vn, timespan=timespan)
            if len(paths) == 0: raise ValueError(f'No timeseries files found for variable `{vn}` in component `{comp}` with hstr `{hstr}` within the timespan `{timespan}`.')
            if timespan is None: paths = paths[load_idx]

            if vn in self.ds:
                if self.ds[vn].path != paths:
                    if verbose: utils.p_warning(f'>>> case.ds["{vn}"] will be reloaded due to different paths.')
                    self.clear_ds(vn)
                    self.load(vn, comp=comp, hstr=hstr, timespan=timespan, load_idx=load_idx, verbose=verbose, reload=False, **kws)
                else:
                    if verbose: utils.p_warning(f'>>> case.ds["{vn}"] already loaded; to reload, run case.load("{vn}", ..., reload=True).')

            else:
                _kws = {
                   'vn': vn,
                   'shift_time': shift_time, 
                   'comp': comp,
                   'hstr': hstr,
                   'grid': self.grid_dict[comp],
                }
                _kws.update(kws)
                if not isinstance(paths, (list, tuple)):
                    ds =  core.open_dataset(paths, **_kws)
                else:
                    ds =  core.open_mfdataset(paths, **_kws)

                self.ds[vn] = ds
                self.ds[vn].attrs['vn'] = vn
                if verbose: utils.p_success(f'>>> case.ds["{vn}"] created')

        elif vtype == 'derived':
            if verbose: utils.p_warning(f'>>> {vn} is a supported derived variable.')
            self.ds[vn] = diags.Registry.funcs[vn](self, comp=comp, hstr=hstr, timespan=timespan, load_idx=load_idx, verbose=verbose, reload=reload, **kws)
            self.ds[vn].attrs['vn'] = vn
            if verbose: utils.p_success(f'>>> case.ds["{vn}"] created')
        else:
            raise ValueError('The input variable name is unknown.')

    def calc(self, spell:str, comp=None, timespan=None, load_idx=-1, recalculate=False, verbose=True, **kws):
        '''Compute a diagnostic spell and cache the result.

        The `spell` string controls regridding, slicing, spatial/vertical
        averaging and other modifiers parsed by `Spell`. The final
        xarray DataArray is stored in `self.diags[spell]`.
        '''
        if spell in self.diags and not recalculate:
            utils.p_warning(f'>>> Spell `{spell}` is already calculated and the calculation is skipped.')
        else:
            S = Spell(spell)
            # `Spell.vn` is already the bare name with any slicing call removed, so
            # no `.split('.')` is needed -- which also stops a dotted variable name
            # like `NINO3.4` from being truncated to `NINO3`.
            vn = S.vn

            if vn in self.diags:
                # reuse a previously calculated bare-variable spell (e.g. `case.calc('TS')`)
                da = self.diags[vn]
                utils.p_warning(f'>>> Variable `{vn}` is already calculated and the calculation is skipped.')
            else:
                self.load(vn, comp=comp, timespan=timespan, load_idx=load_idx, verbose=verbose, **kws)
                da = self.ds[vn].x.da

            if S.slicing_method is not None:
                da = getattr(da, S.slicing_method)(*S.slicing_args, **S.slicing_kwargs)

            if S.plev is not None:
                self.load('PS')
                PS = self.ds['PS']['PS']
                hyam = self.ds[vn]['hyam']
                hybm = self.ds[vn]['hybm']
                _kws = {'lev_dim': 'lev'}
                if S.plev_levels is not None:
                    _kws['new_levels'] = np.array(S.plev_levels)

                da = da.x.get_plev(ps=PS, hyam=hyam, hybm=hybm, **_kws)

            if S.ann_method is not None:
                utils.p_hint(f'>>> Timespan: [{da.time.values[0]}, {da.time.values[-1]}]')
                da = utils.ann_modifier(da, ann_method=S.ann_method, long_name=da.long_name)

            if S.regrid is not None:
                da = da.x.regrid(*S.regrid_args, **S.regrid_kwargs)

            # zavg must run before the horizontal mean: it folds the vertical into a
            # volume weight, so a following sa_method (e.g. gm) yields a true
            # volume-weighted average.
            if S.zavg is not None:
                da = da.x.zavg(*S.zavg_args, **S.zavg_kwargs)

            if S.sa_method is not None:
                # `Spell` has already validated the name against Spell.SA_METHODS
                if S.sa_method == 'yz':
                    if da.name != 'MOC':
                        da = da.x.zm
                else:
                    da = getattr(da.x, S.sa_method)

            units = da.attrs.get('units')
            if units == 'degC':
                da.attrs['units'] = '°C'
            elif units == 'K':
                da -= 273.15
                da.attrs['units'] = '°C'

            if S.alias is not None:
                spell = S.alias
                da.name = S.alias

            self.diags[spell] = da.squeeze().compute()
            if verbose: utils.p_success(f'>>> case.diags["{spell}"] created')
        return self.diags[spell]

    def plot(self, spell, t_idx=None, regrid=False, gs='T', ssv='SSH', recalculate_ssv=False, timespan=None, **kws):
        '''Plot a computed diagnostic `spell`.

        Detects plot type (map, ts, zm, yz) from the DataArray and
        dispatches to the plotting helpers in `diags`/`visual`.
        '''
        if spell not in self.diags:
            utils.p_warning(f'>>> "{spell}" not calculated yet. Calculating now ...')
            self.calc(spell, timespan=timespan)

        da = self.diags[spell]

        if t_idx is None:
            if len(da.dims) > 1 and 'time' in da.dims:
                da = da.mean('time')
        else:
            da = da.isel(time=t_idx)
            da.attrs['long_name'] += f'\n{da.time.values}'

        if regrid: da = da.x.regrid()

        if da.x.is_map():
            plot_type = 'map'
        elif len(da.dims) == 2:
            plot_type = 'yz'
        elif 'time' in da.dims or 'year' in da.dims or 'month' in da.dims:
            plot_type = 'ts'
        elif 'lat' in da.dims:
            plot_type = 'zm'
        else:
            raise ValueError('Unkown plot type.')

        kws_dict = deepcopy(diags.DiagPlot.__dict__[f'kws_{plot_type}'])
        _kws = kws_dict[da.name] if da.name in kws_dict else {}
        _kws['gs'] = gs
        _kws = utils.update_dict(_kws, kws)

        if plot_type == 'map':
            # if (ssv, 'ocn') in self.vars_info and (recalculate_ssv or ('ssv' not in self.diags)):
            if len(self.get_comp_hstr(ssv))==1 and (recalculate_ssv or ('ssv' not in self.diags)):
                self.load(ssv)
                da_ssv = self.ds[ssv].x.regrid().x.da.mean('time')
                self.diags['ssv'] = da_ssv

        if 'ssv' in self.diags:
            fig_ax =  da.x.plot(ssv=self.diags['ssv'], **_kws)
        else:
            fig_ax =  da.x.plot(**_kws)

        ax = fig_ax[-1] if isinstance(fig_ax, tuple) else fig_ax

        if 'xlabel' not in kws:
            xlabel = ax.xaxis.get_label()
            if 'climo_period' in da.attrs:
                ax.set_xlabel('Month')
                ax.set_xticks(range(1, 13))
                ax.set_xticklabels(range(1, 13))
            elif 'lat' in str(xlabel):
                ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
                ax.set_xticklabels(['90°S', '60°S', '30°S', 'EQ', '30°N', '60°N', '90°N'])
                ax.set_xlim([-90, 90])
                ax.set_xlabel('Latitude')
        else:
            ax.set_xlabel(kws['xlabel'])

        if 'ylabel' not in kws:
            ylabel = ax.yaxis.get_label()
            if 'depth' in str(ylabel):
                ax.invert_yaxis()
                if 'z_t' in self.diags[spell].coords:
                    if self.diags[spell]['z_t'].units == 'centimeters':
                        ax.set_yticks([0, 2e5, 4e5])
                    elif self.diags[spell]['z_t'].units == 'km':
                        ax.set_yticks([0, 2, 4])
                else:
                    ax.set_yticks([0, 2e5, 4e5])

                ax.set_yticklabels([0, 2, 4])
                ax.set_ylabel('Depth [km]')
        else:
            ax.set_ylabel(kws['ylabel'])

        return fig_ax

    def quickview(self, timespan=None, nrow=None, ncol=None, wspace=0.3, hspace=0.5, ax_loc=None, figsize=None,
                  stat_period=-50, roll_int=50, ylim_dict=None, spells=None, recalculate=False):
        '''Create a multi-panel overview figure for a selection of spells.

        Returns `(fig, ax)` where `ax` is a dict of axes keyed by spell
        keys.
        '''
        spells = {
            'GMST': 'TS:ann:gm',
            'GMRESTOM': 'RESTOM:ann:gm',
            'GMLWCF': 'LWCF:ann:gm',
            'GMSWCF': 'SWCF:ann:gm',
            'NHICEFRAC': 'ICEFRAC:ann:nhs',
            'NHICEFRAC_clim': 'ICEFRAC:climo:nhs',
            'SOMOC': 'MOC:ann:somin',
            'MOC': 'MOC:ann:yz',
        } if spells is None else spells

        ax_loc={
            'GMST': (0, 0),
            'GMRESTOM': (0, 1),
            'GMLWCF': (0, 2),
            'GMSWCF': (0, 3),
            'NHICEFRAC': (1, 0),
            'NHICEFRAC_clim': (1, 1),
            'SOMOC': (1, 2),
            'MOC': (1, 3),
        } if ax_loc is None else ax_loc

        # if the caller supplied their own spells but no layout, lay them out in order
        # rather than indexing the default `ax_loc` with unknown keys
        if set(ax_loc) != set(spells):
            ncol_tmp = np.min([len(spells), 4]) if ncol is None else ncol
            ax_loc = {k: (i // ncol_tmp, i % ncol_tmp) for i, k in enumerate(spells)}

        nsubplots = len(spells)
        utils.p_header(f'>>> Plotting {nsubplots} subplots')
        ncol = np.min([nsubplots, 4]) if ncol is None else ncol
        nrow = int(np.ceil(nsubplots/ncol)) if nrow is None else nrow
        utils.p_hint(f'>>> {nrow = }, {ncol = }')
        figsize = (ncol*5, nrow*4) if figsize is None else figsize

        fig, ax = visual.subplots(
            nrow=nrow, ncol=ncol,
            ax_loc=ax_loc,
            figsize=figsize,
            wspace=wspace,
            hspace=hspace,
        )

        clr_dict = {
            'GMST': 'tab:red',
            'GMSST': 'tab:red',
            'GMSSS': 'tab:purple',
            'GMRESTOM': 'tab:blue',
            'GMLWCF': 'tab:green',
            'GMSWCF': 'tab:orange',
            'NHICEFRAC': 'tab:cyan',
            'NHICEFRAC_clim': 'tab:cyan',
            'SOMOC': 'tab:blue',
        }
        title_dict = {
            'GMST': 'Global Mean Surface Temperature',
            'GMSST': 'Global Mean Sea Surface Temperature',
            'GMSSS': 'Global Mean Sea Surface Salinity',
            'GMRESTOM': 'Global Mean Net Radiative Flux',
            'GMLWCF': 'Global Mean Longwave Cloud Forcing',
            'GMSWCF': 'Global Mean Shortwave Cloud Forcing',
            'NHICEFRAC': 'NH Mean Ice Area',
            'NHICEFRAC_clim': 'NH Mean Ice Area Annual Cycle',
            'SOMOC': 'Southern Ocean (90°S-28°S) MOC',
            'MOC': 'Meridional Ocean Circulation',
        }

        for k, v in spells.items():
            if 'zm' in v:
                self.calc(v, timespan=None, recalculate=recalculate)
            else:
                self.calc(v, timespan=timespan, recalculate=recalculate)

        # `.get`, not `[...]`: `spells` is user-overridable, so a custom key is not in
        # these hard-coded dicts and used to raise KeyError
        for k, v in spells.items():
            utils.p_header(f'>>> Plotting {k}')
            title = title_dict.get(k, k)
            color = clr_dict.get(k)
            if len(self.diags[v].dims) == 1 and 'time' in self.diags[v].dims:
                # timeseries
                if '_clim' in k:
                    self.plot(v, ax=ax[k], title=title, color=color, alpha=1)
                else:
                    self.plot(v, ax=ax[k], title=title, color=color, alpha=0.1)
                    # clamp the smoothing window to the record: the default (50) is
                    # tuned for long runs, and xarray raises
                    # "Moving window must be between 1 and N" on anything shorter
                    nt = self.diags[v].sizes['time']
                    win = int(np.clip(roll_int, 1, nt))
                    if win > 1:
                        vals = self.diags[v].rolling(time=win, center=True).mean().data
                        ax[k].plot(self.diags[v].time, vals, color=color)

                if timespan is not None and 'climo_period' not in self.diags[v].attrs:
                    start_date = cftime.DatetimeNoLeap(timespan[0], 1, 1)
                    end_date = cftime.DatetimeNoLeap(timespan[1], 1, 1)
                    ax[k].set_xlim(start_date, end_date)

            elif k in ['TS']:
                self.plot(v, ax=ax[k], title=title, cbar_kwargs={'orientation': 'horizontal', 'aspect': 20, 'pad': 0.05})
            else:
                self.plot(v, ax=ax[k], title=title)

            if ylim_dict is not None and k in ylim_dict:
                ax[k].set_ylim(ylim_dict[k])

        for k, v in spells.items():
            if len(self.diags[v].dims) == 1 and self.diags[v].dims[0] == 'time' and 'climo_period' not in self.diags[v].attrs:
                ax[k].text(
                    0.95, 0.9,
                    f'last {np.abs(stat_period)}-yr mean: {self.diags[v].isel(time=slice(stat_period,)).mean().values:.2f}',
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    transform=ax[k].transAxes,
                    color=clr_dict.get(k),
                    fontsize=15,
                )

        return fig, ax


    def get_ts(self, vn, comp=None, hstr=None, timespan=None, slicing=False, regrid=False, dlat=1, dlon=1):
        '''Open and return a Dataset for `vn`, without caching it in `self.ds`.

        Applies optional slicing and regridding before returning the Dataset.

        Args:
            vn (str): variable name
            comp (str): component; inferred from `vn` when it is unambiguous
            hstr (str): history-stream tag; inferred from `vn` when it is unambiguous
            timespan (tuple): (start, end), as either ints or 'YYYY-MM'-style strings
            slicing (bool): additionally `.sel` the time axis to `timespan`
            regrid (bool): regrid to a regular `dlat` x `dlon` grid
        '''
        shift_time = True if self.cesm_ver == 1 else False

        if comp is None or hstr is None:
            found_comp_hstr = self.get_comp_hstr(vn)
            if len(found_comp_hstr) == 0:
                raise ValueError(f'The input variable name `{vn}` is unknown.')
            elif len(found_comp_hstr) > 1:
                raise ValueError(
                    f'The input variable name belongs to multiple (comp, hstr) pairs: '
                    f'{found_comp_hstr}. Please specify via the argument `comp` and `hstr`.'
                )
            comp, hstr = found_comp_hstr[0]

        timespan = utils.normalize_timespan(timespan)

        paths = self.get_paths(comp, hstr, vn, timespan=timespan)
        if len(paths) == 0:
            raise ValueError(
                f'No timeseries files found for variable `{vn}` in component `{comp}` '
                f'with hstr `{hstr}` within the timespan `{timespan}`.'
            )

        # `comp`/`grid` go through `open_mfdataset` so that `update_ds` also attaches
        # `gw`/`lat`/`lon`; setting them on `.attrs` afterwards would leave the accessors
        # without a weight
        ds = core.open_mfdataset(
            paths, vn=vn, shift_time=shift_time,
            comp=comp, hstr=hstr, grid=self.grid_dict[comp],
        )

        if slicing:
            if timespan is None:
                raise ValueError('`slicing=True` requires a `timespan`.')
            ds = ds.sel(time=slice(timespan[0], timespan[1]))

        if regrid: ds = ds.x.regrid(dlat=dlat, dlon=dlon)
        return ds

    def save_means(self, vn, comp=None, output_dirpath=None, timespan=None, hstr=None,
                   slicing=False, regrid=False, dlat=1, dlon=1, overwrite=False):
        '''Save seasonal and annual mean files for `vn` into `output_dirpath`.

        Writes files for ANN, DJF, MAM, JJA and SON for the given
        `timespan` and optionally regrids results.
        '''
        if output_dirpath is None: raise ValueError('Please specify `output_dirpath`.')
        if timespan is None: raise ValueError('Please specify `timespan`.')

        output_dirpath = pathlib.Path(output_dirpath)

        if not output_dirpath.exists():
            output_dirpath.mkdir(parents=True, exist_ok=True)
            utils.p_success(f'>>> output directory created at: {output_dirpath}')

        ds = self.get_ts(vn, comp=comp, hstr=hstr, timespan=timespan, slicing=slicing, regrid=False)

        sn_dict = {
            'ANN': list(range(1, 13)),
            'DJF': [12, 1, 2],
            'MAM': [3, 4, 5],
            'JJA': [6, 7, 8],
            'SON': [9, 10, 11],
        }

        for sn, months in sn_dict.items():
            output_subdirpath = pathlib.Path(os.path.join(output_dirpath, sn))
            if not output_subdirpath.exists():
                output_subdirpath.mkdir(parents=True, exist_ok=True)

            fname = f'{timespan[0]}_{timespan[1]}_{vn}_{sn}_means.nc'
            if self.casename is not None: fname = f'{self.casename}_{fname}'

            out_path = os.path.join(output_subdirpath, fname)
            if overwrite or not os.path.exists(out_path):
                # ds_ann = self.get_mean(
                #     vn, comp, months=months, timespan=timespan, adjust_month=adjust_month, slicing=slicing,
                #     regrid=regrid, dlat=dlat, dlon=dlon, chunk_nt=chunk_nt,
                # )
                ds_ann = ds.x.annualize(months=months)
                if regrid: ds_ann = ds_ann.x.regrid(dlat=dlat, dlon=dlon)
                # `.x.to_netcdf`, not `.to_netcdf`: `annualize` carries the grid attrs
                # (`gw`/`lat`/`lon`/`dz`) over, and those are not serializable
                ds_ann.x.to_netcdf(out_path)
                ds_ann.close()


    def clear_ds(self, vn=None):
        ''' Clear the existing `.ds` property
        '''
        if vn is not None:
            self.ds.pop(vn, None)
        else:
            self.ds = {}
        
    def copy(self):
        '''Return a deep copy of this Timeseries instance.'''
        return deepcopy(self)


class Logs:
    '''Manage CESM log files for a case and extract time series variables.

    This helper locates compressed component log files (e.g. `ocn.log.*.gz`),
    parses monthly reported diagnostics, and provides plotting helpers for
    inspected variables.
    '''
    def __init__(self, dirpath, comp='ocn', load_num=None):
        '''Initialize Logs with a directory containing component logs.

        Args:
            dirpath (str): directory containing component log files.
            comp (str): component name prefix for log files (default 'ocn').
            load_num (int or None): limit number of files to load; negative
                values slice from the end.
        '''
        self.dirpath = dirpath
        self.paths = sorted(glob.glob(os.path.join(dirpath, f'{comp}.log.*.gz')))
        if load_num is not None:
            if load_num < 0:
                self.paths = self.paths[load_num:]
            else:
                self.paths = self.paths[:load_num]

        utils.p_header(f'>>> Logs.dirpath: {self.dirpath}')
        if len(self.paths) == 0:
            # otherwise this surfaced as a bare IndexError on the next line
            raise FileNotFoundError(
                f'No `{comp}.log.*.gz` files found in {dirpath}'
            )

        utils.p_header(f'>>> {len(self.paths)} Logs.paths:')
        print(f'Start: {os.path.basename(self.paths[0])}')
        print(f'End: {os.path.basename(self.paths[-1])}')

    def get_vars(self, vn=[
                    'UVEL', 'UVEL2', 'VVEL', 'VVEL2', 'TEMP', 'dTEMP_POS_2D', 'dTEMP_NEG_2D', 'SALT', 'RHO', 'RHO_VINT',
                    'RESID_T', 'RESID_S', 'SU', 'SV', 'SSH', 'SSH2', 'SHF', 'SHF_QSW', 'SFWF', 'SFWF_WRST', 'TAUX', 'TAUX2', 'TAUY',
                    'TAUY2', 'FW', 'TFW_T', 'TFW_S', 'EVAP_F', 'PREC_F', 'SNOW_F', 'MELT_F', 'ROFF_F', 'IOFF_F', 'SALT_F', 'SENH_F',
                    'LWUP_F', 'LWDN_F', 'MELTH_F', 'IFRAC', 'PREC_16O_F', 'PREC_18O_F', 'PREC_HDO_F', 'EVAP_16O_F', 'EVAP_18O_F', 'EVAP_HDO_F',
                    'MELT_16O_F', 'MELT_18O_F', 'MELT_HDO_F', 'ROFF_16O_F', 'ROFF_18O_F', 'ROFF_HDO_F', 'IOFF_16O_F', 'IOFF_18O_F', 'IOFF_HDO_F',
                    'R18O', 'FvPER_R18O', 'FvICE_R18O', 'RHDO', 'FvPER_RHDO', 'FvICE_RHDO', 'ND143', 'ND144', 'IAGE', 'QSW_HBL', 'KVMIX', 'KVMIX_M',
                    'TPOWER', 'VDC_T', 'VDC_S', 'VVC', 'KAPPA_ISOP', 'KAPPA_THIC', 'HOR_DIFF', 'DIA_DEPTH', 'TLT', 'INT_DEPTH', 'UISOP', 'VISOP',
                    'WISOP', 'ADVT_ISOP', 'ADVS_ISOP', 'VNT_ISOP', 'VNS_ISOP', 'USUBM', 'VSUBM', 'WSUBM', 'HLS_SUBM', 'ADVT_SUBM', 'ADVS_SUBM',
                    'VNT_SUBM', 'VNS_SUBM', 'HDIFT', 'HDIFS', 'WVEL', 'WVEL2', 'UET', 'VNT', 'WTT', 'UES', 'VNS', 'WTS', 'ADVT', 'ADVS', 'PV',
                    'Q', 'PD', 'QSW_HTP', 'QFLUX', 'HMXL', 'XMXL', 'TMXL', 'HBLT', 'XBLT', 'TBLT', 'BSF',
                    'NINO_1_PLUS_2', 'NINO_3', 'NINO_3_POINT_4', 'NINO_4',
                ]):

        '''Parse log files and extract listed variables into a DataFrame.

        Args:
            vn (list or str): variables to extract from logs. If a string,
                it will be converted to a single-element list.

        Side effects:
            Sets `self.df`, `self.df_ann` and `self.vn` with parsed results.
        '''
        if not isinstance(vn, (list, tuple)):
            vn = [vn]

        nf = len(self.paths)
        df_list = []
        for idx_file in range(nf):
            # initialize up front: folding this into the scan loop with an `elif`
            # meant the first line of every file was never tested for a match
            vars = {v: [] for v in vn}
            with gzip.open(self.paths[idx_file], mode='rt') as fp:
                lines = fp.readlines()

                # find 1st timestamp. `enumerate`, not `lines.index(line)`: that
                # returned the index of the *first* equal line (wrong whenever a line
                # repeats, which log files do constantly) and made the scan O(n^2)
                # over a multi-MB decompressed log.
                start_date = None
                for i, line in enumerate(lines[:-1]):
                    if 'This run        started from' in line and 'date(month-day-year):' in lines[i+1]:
                        start_date = lines[i+1].split(':')[-1].strip()
                        break

                if start_date is None:
                    raise ValueError(
                        f'Could not find the run start date in {self.paths[idx_file]}: '
                        'expected a "This run        started from" line followed by '
                        '"date(month-day-year):". Is this a POP (ocn) log?'
                    )

                mm, dd, yyyy = start_date.split('-')

                # find variable values
                for line in lines:
                    stripped = line.strip()
                    for v in vn:
                        if stripped.startswith(f'{v}:'):
                            vars[v].append(float(stripped.split(':')[-1]))
                            break

            # a variable absent from this log leaves an empty list, and pandas refuses
            # to build a frame from unequal-length columns; drop them with a note
            # rather than failing on the (large) default `vn` list
            missing = [v for v, vals in vars.items() if len(vals) == 0]
            if missing:
                if idx_file == 0:
                    utils.p_warning(
                        f'>>> {len(missing)} requested variable(s) not found in the logs '
                        f'and skipped: {missing[:8]}{" ..." if len(missing) > 8 else ""}'
                    )
                for v in missing:
                    del vars[v]

            df_tmp = pd.DataFrame(vars)
            dates = xr.date_range(start=f'{yyyy}-{mm}-{dd}', freq='MS', periods=len(df_tmp), calendar='noleap')
            years = []
            months = []
            for date in dates:
                years.append(date.year)
                months.append(date.month)

            df_tmp['Year'] = years
            df_tmp['Month'] = months
            df_list.append(df_tmp)
        
        df = pd.concat(df_list, join='inner').drop_duplicates(subset=['Year', 'Month'], keep='last')
        df = df[ ['Year', 'Month'] + [ col for col in df.columns if col not in ['Year', 'Month']]]
        self.df = df
        self.df_ann = self.df.groupby(self.df.Year).mean()
        self.vn = vn

    def plot_vars(self, vn=None, annualize=True, xlim=None, ylim_dict=None, unit_dict=None, clr_dict=None,
                  figsize=[20, 5], ncol=4, nrow=None, wspace=0.5, hspace=0.5, kws=None, title=None):
        '''Plot one or more variables parsed from the logs.

        Args:
            vn (list or str): variables to plot; defaults to the parsed set.
            annualize (bool): plot annual means when True.
            xlim, ylim_dict, unit_dict, clr_dict: plotting customizations.
            figsize, ncol, nrow: layout options.

        Returns:
            (fig, ax) Matplotlib figure and dict of axes keyed by variable name.
        '''
        kws = {} if kws is None else kws
        unit_dict = {} if unit_dict is None else unit_dict
        clr_dict = {} if clr_dict is None else clr_dict

        _unit_dict = {
            'TEMP': 'degC',
            'SALT': 'kg/kg',
            'QFLUX': 'W/m^2',
            'NINO_3_POINT_4': 'degC',
        }
        _unit_dict.update(unit_dict)

        _clr_dict = {
            'TEMP': 'tab:red',
            'SALT': 'tab:green',
            'QFLUX': 'tab:blue',
            'NINO_3_POINT_4': 'tab:orange',
        }
        _clr_dict.update(clr_dict)

        if vn is None:
            vn = self.vn

        if not isinstance(vn, (list, tuple)):
            vn = [vn]

        if nrow is None:
            nrow = int(np.ceil(len(vn)/ncol))

        if annualize:
            df_plot = self.df_ann
        else:
            df_plot = self.df
            
        fig = plt.figure(figsize=figsize)
        ax = {}
        gs = gridspec.GridSpec(nrow, ncol)
        gs.update(wspace=wspace, hspace=hspace)

        for i, v in enumerate(vn):
            if v in self.df.columns:
                if v not in kws:
                    kws[v] = {}

                ax[v] = fig.add_subplot(gs[i])

                if v in _clr_dict:
                    kws[v]['color'] = _clr_dict[v]

                if v == 'SALT':
                    ax[v].plot(df_plot.index, df_plot[v].values*1e3, **kws[v])
                else:
                    df_plot[v].plot(ax=ax[v], **kws[v])

                if v in _unit_dict:
                    ax[v].set_ylabel(f'{v} [{_unit_dict[v]}]')
                else:
                    ax[v].set_ylabel(v)

                ax[v].ticklabel_format(useOffset=False)
                if xlim is not None:
                    ax[v].set_xlim(xlim)
                if ylim_dict is not None and v in ylim_dict:
                    ax[v].set_ylim(ylim_dict[v])

        if title is not None:
            fig.suptitle(title)

        return fig, ax

    
    def compare_vars(self, L_ref, vn=None, annualize=True, xlim=None, unit_dict=None, clr_dict=None,
                  figsize=[20, 5], ncol=4, nrow=None, wspace=0.3, hspace=0.5, kws=None, title=None):
        '''Overlay variables from a reference `Logs` instance for comparison.

        Args:
            L_ref (Logs): reference Logs instance whose variables will be
                plotted over the current instance's plots in black.
        '''
        if vn is None:
            vn = self.vn

        fig, ax = self.plot_vars(vn=vn, annualize=annualize, xlim=xlim, unit_dict=unit_dict, clr_dict=clr_dict,
                                 figsize=figsize, ncol=ncol, nrow=nrow, wspace=wspace, hspace=hspace, kws=kws, title=title)

        kws = {} if kws is None else kws
        unit_dict = {} if unit_dict is None else unit_dict
        clr_dict = {} if clr_dict is None else clr_dict

        if annualize:
            df_plot = L_ref.df_ann
        else:
            df_plot = L_ref.df

        for v in vn:
            if v not in kws:
                kws[v] = {}

            if v == 'SALT':
                ax[v].plot(df_plot.index, df_plot[v].values*1e3, color='k', **kws[v])
            else:
                df_plot[v].plot(ax=ax[v], color='k', **kws[v])
        
        return fig, ax