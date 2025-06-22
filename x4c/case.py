import os
import glob
import pandas as pd
import gzip
from tqdm import tqdm
import xarray as xr
import multiprocessing as mp
import pathlib
import textwrap
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
import dask

from . import core, utils, diags
from .spell import Spell

class History:
    def __init__(self, root_dir, comps=['atm', 'ocn', 'lnd', 'ice', 'rof'], hstr_dict=None, casename=None,
                 path_pattern='comp/hist/casename.mdl.hstr.date.nc', avoid_list=['nday1', 'once'], cesm_ver=1):
        self.path_pattern = path_pattern
        self.root_dir = root_dir
        self.casename = casename
        utils.p_header(f'>>> case.root_dir: {self.root_dir}')
        utils.p_header(f'>>> case.casename: {self.casename}')

        if cesm_ver == 1:
            _hstr_dict = {
                'atm': ('cam', 'h0'),
                'ocn': ('pop', 'h'),
                'lnd': ('clm2', 'h0'),
                'ice': ('cice', 'h'),
                'rof': ('rtm', 'h0'),
            }
        else:
            _hstr_dict = {
                'atm': ('cam', ['h0a', 'h0i']),
                'ocn': ('mom6', ['h.sfc', 'h.z']),
                'lnd': ('clm2', 'h0'),
                'ice': ('cice', 'h'),
                'rof': ('mosart', 'h0'),
            }
        if hstr_dict is not None:
            _hstr_dict.update(hstr_dict)

        self.comps_info = _hstr_dict
        utils.p_header(f'>>> case.comps_info: {self.comps_info}')

        self.paths = {}
        for comp in comps:
            mdl, hstr = _hstr_dict[comp]
            self.paths[comp] = utils.find_paths(
                self.root_dir, self.path_pattern,
                comp=comp, mdl=mdl, hstr=hstr,
                avoid_list=avoid_list,
            )
            utils.p_success(f'>>> case.paths["{comp}"] created')

        self.vns = {}
        for comp in comps:
            self.vns[comp] = self.get_ts_vns(comp)
            utils.p_success(f'>>> case.vns["{comp}"] created')

    def get_ts_vns(self, comp):
        vns_ts = []
        ds0 = core.open_dataset(self.paths[comp][0])
        vns = list(ds0.variables)
        exclude_vars = ['time', 'time_bnds', 'time_written', 'date', 'datesec', 'date_written']

        for v in vns:
            if len(ds0[v].dims) >= 2 and 'time' in ds0[v].dims and 'time' not in v and v not in exclude_vars:
                vns_ts.append(v)
        
        vns_ts = sorted(vns_ts)

        ds0.close()
        return vns_ts

    def get_paths(self, comp, timespan=None):
        paths = self.paths[comp]

        if timespan is None:
            paths_sub = paths
        else:
            syr, eyr = timespan
            paths_sub = []
            for path in paths:
                yr_tmp = int(path.split('.')[-2].split('-')[0][:4])
                if (yr_tmp >= syr and yr_tmp <= eyr) or (yr_tmp >= syr and yr_tmp <= eyr):
                    paths_sub.append(path)

        return paths_sub

    def isolate_vn(self, vn, comp, in_path, output_dirpath, overwrite=True):
        bn_elements = os.path.basename(in_path).split('.')
        bn_elements.insert(-2, vn)

        if self.casename is not None:
            fname = '.'.join(bn_elements[-5:])
            fname = f'{self.casename}.{fname}'
        else:
            fname = '.'.join(bn_elements)

        out_path = os.path.join(output_dirpath, fname)
        if overwrite or not os.path.exists(out_path):
            if os.path.exists(out_path): os.remove(out_path)
            vns = self.vns[comp].copy()
            vns.remove(vn)
            cmd = f'ncks -h -C -x -v {",".join(vns)} {in_path} -o {out_path}'
            subprocess.run(cmd, shell=True)

    def bigbang(self, comp, output_dirpath, timespan=None, overwrite=True, nproc=1, vns=None):
        output_dirpath = pathlib.Path(output_dirpath)
        output_dirpath.mkdir(parents=True, exist_ok=True)

        paths = self.get_paths(comp, timespan=timespan)

        if vns is None: vns = self.vns[comp]
        if nproc == 1:
            for path in tqdm(paths, desc=f'Spliting {len(paths)} history files'):
                for vn in vns:
                    self.isolate_vn(vn, comp, in_path=path, output_dirpath=output_dirpath, overwrite=overwrite)
        else:
            utils.p_hint(f'>>> nproc: {nproc}')
            with mp.Pool(processes=nproc) as p:
                arg_list = []
                for path in paths:
                    for vn in vns:
                        arg_list.append((vn, comp, path, output_dirpath, overwrite))
                p.starmap(self.isolate_vn, tqdm(arg_list, total=len(arg_list), desc=f'Spliting {len(paths)} history files for {len(vns)} variables'))

    def merge_vn(self, vn, input_dirpath, output_dirpath, timespan=None, overwrite=True, compression=1):
        paths = sorted(glob.glob(os.path.join(input_dirpath, f'*.{vn}.*.nc')))
        if timespan is None:
            paths_sub = paths
        else:
            syr, eyr = timespan
            paths_sub = []
            for path in paths:
                year = int(path.split('.')[-2].split('-')[0])
                if (year >= syr and year <= eyr) or (year >= syr and year <= eyr):
                    paths_sub.append(path)

        date_start = ''.join(paths[0].split('.')[-2].split('-'))
        date_end = ''.join(paths[-1].split('.')[-2].split('-'))

        bn_elements = os.path.basename(paths[0]).split('.')
        bn_elements[-2] = f'{date_start}-{date_end}'

        if self.casename is not None:
            fname = '.'.join(bn_elements[-5:])
            fname = f'{self.casename}.{fname}'
        else:
            fname = '.'.join(bn_elements)
        out_path = os.path.join(output_dirpath, fname)

        if overwrite or not os.path.exists(out_path):
            if os.path.exists(out_path): os.remove(out_path)
            cmd = f'ncrcat -O -4 -h --no_cll_mth -L {compression} {" ".join(paths_sub)} -o {out_path}'
            subprocess.run(cmd, shell=True)

    def bigcrunch(self, comp, input_dirpath, output_dirpath, timespan=None, overwrite=True, nproc=1, compression=1, vns=None):
        output_dirpath = pathlib.Path(output_dirpath)
        output_dirpath.mkdir(parents=True, exist_ok=True)

        if vns is None: vns = self.vns[comp]
        desc = 'Merging variables'
        if nproc == 1:
            for vn in tqdm(vns, desc=desc):
                self.merge_vn(vn, input_dirpath=input_dirpath, output_dirpath=output_dirpath, timespan=timespan, overwrite=overwrite, compression=compression)
        else:
            utils.p_hint(f'>>> nproc: {nproc}')
            with mp.Pool(processes=nproc) as p:
                arg_list = []
                for vn in vns:
                    arg_list.append((vn, input_dirpath, output_dirpath, timespan, overwrite, compression))
                p.starmap(self.merge_vn, tqdm(arg_list, total=len(arg_list), desc=desc))

    def gen_ts(self, output_dirpath, staging_dirpath=None, comps=['atm', 'ocn', 'lnd', 'ice', 'rof'], years_per_file=None, timestep=None, timespan=None,
               dir_structure='comp/proc/tseries/month_1' , overwrite=True, nproc=1, compression=1):

        if staging_dirpath is None: staging_dirpath = output_dirpath
        if timespan is None: raise ValueError('Please specify `timespan`.')
        if years_per_file is None and timestep is None: raise ValueError('Please specify `years_per_file` or the equivalent `timestep`.')
        if years_per_file is not None and timestep is None: timestep = years_per_file

        syr = timespan[0]
        nt = (timespan[-1] - timespan[0] + 1) // timestep
        timespan_list = []
        for i in range(nt):
            timespan_list.append((syr, syr+timestep-1))
            syr += timestep 

        if type(comps) is not dict:
            comps = {comp: None for comp in comps}

        for comp, vns in comps.items():
            # generate timeseries files for each component and each sub-timespan
            utils.p_header(f'>>> Processing component: {comp}')
            for timespan_tmp in timespan_list:
                utils.p_header(f'>>> Processing timespan: {timespan_tmp}')
                bigbang_dir = os.path.join(staging_dirpath, f'.bigbang_{comp}.{timespan_tmp[0]:04d}-{timespan_tmp[1]:04d}')
                if os.path.exists(bigbang_dir): shutil.rmtree(bigbang_dir)
                self.bigbang(comp=comp, output_dirpath=bigbang_dir, timespan=timespan_tmp, overwrite=overwrite, nproc=nproc, vns=vns)

                bigcrunch_dir = os.path.join(staging_dirpath, dir_structure.replace('comp', comp))
                self.bigcrunch(comp=comp, input_dirpath=bigbang_dir, output_dirpath=bigcrunch_dir, timespan=timespan_tmp, overwrite=overwrite, nproc=nproc, compression=compression, vns=vns)

        for comp, vns in comps.items():
            # delete the temporary files
            utils.p_header(f'>>> Postprocessing component: {comp}')
            for timespan_tmp in timespan_list:
                utils.p_header(f'>>> Postprocessing timespan: {timespan_tmp}')
                bigbang_dir = os.path.join(staging_dirpath, f'.bigbang_{comp}.{timespan_tmp[0]:04d}-{timespan_tmp[1]:04d}')
                bigcrunch_dir = os.path.join(staging_dirpath, dir_structure.replace('comp', comp))
                if os.path.exists(bigbang_dir): shutil.rmtree(bigbang_dir)
                if staging_dirpath != output_dirpath:
                    # move files from staging to destination
                    dst_dir = os.path.join(output_dirpath, dir_structure.replace('comp', comp))
                    dst_dir = pathlib.Path(dst_dir)
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    src_paths = glob.glob(os.path.join(bigcrunch_dir, f'*.{timespan_tmp[0]:04d}01-{timespan_tmp[1]:04d}12.nc'))
                    # [shutil.move(src_path, dst_dir) for src_path in src_paths]
                    # print(f'{src_paths =}')
                    # print(f'{dst_dir =}')
                    # for src_path in src_paths:
                    #     dst_path = os.path.join(dst_dir, os.path.basename(src_path))
                    #     if os.path.exists(dst_path):
                    #         os.remove(dst_path)
                            
                    with mp.Pool(processes=nproc) as p:
                        arg_list = [(src_path, dst_dir) for src_path in src_paths]
                        p.starmap(shutil.move, tqdm(arg_list, total=len(arg_list), desc=f'Moving generated files\nfrom: {staging_dirpath}\nto: {output_dirpath}\n'))

                    # utils.p_header(f'>>> Moving generated files\nfrom: {staging_dirpath}\nto: {output_dirpath}\n ')
                    # # utils.rsync_move(src_paths, dst_dir)
                    # src_paths =  f'{bigcrunch_dir}/*.{timespan_tmp[0]:04d}01-{timespan_tmp[1]:04d}12.nc'
                    # cmd = f'rsync -a --remove-source-files {src_paths} {str(dst_dir)}'
                    # print('>>> {cmd}')
                    # subprocess.run(cmd, check=True, shell=True)


    # def split_ds(self, comp, in_path, output_dirpath, overwrite=False, nco=True):
    #     if not nco: ds = xr.load_dataset(in_path)
    #     for vn in self.vns[comp]:
    #         bn_elements = os.path.basename(in_path).split('.')
    #         bn_elements.insert(-2, vn)

    #         if self.casename is not None:
    #             fname = '.'.join(bn_elements[-5:])
    #             fname = f'{self.casename}.{fname}'
    #         else:
    #             fname = '.'.join(bn_elements)

    #         out_path = os.path.join(output_dirpath, fname)
    #         if overwrite or not os.path.exists(out_path):
    #             if os.path.exists(out_path): os.remove(out_path)
    #             vns = self.vns[comp].copy()
    #             vns.remove(vn)
    #             if nco:
    #                 cmd = f'ncks -C -x -v {",".join(vns)} {in_path} -o {out_path}'
    #                 subprocess.run(cmd, shell=True)
    #             else:
    #                 ds_vn = ds.drop_vars(vns)
    #                 ds_vn.to_netcdf(out_path)
    #                 ds_vn.close()

    #     if not nco: ds.close()

    # def bigbang(self, comp, output_dirpath, timespan=None, overwrite=False, nproc=1, nco=True):
    #     output_dirpath = pathlib.Path(output_dirpath)
    #     if not output_dirpath.exists():
    #         output_dirpath.mkdir(parents=True, exist_ok=True)
    #         # utils.p_success(f'>>> output directory created at: {output_dirpath}')

    #     paths = self.get_paths(comp, timespan=timespan)
    #     if nproc == 1:
    #         for path in tqdm(paths, desc='Spliting history files'):
    #             self.split_ds(comp, in_path=path, output_dirpath=output_dirpath, overwrite=overwrite, nco=nco)
    #     else:
    #         utils.p_hint(f'>>> nproc: {nproc}')
    #         with mp.Pool(processes=nproc) as p:
    #             arg_list = []
    #             for path in paths:
    #                 arg_list.append((comp, path, output_dirpath, overwrite, nco))
    #             p.starmap(self.split_ds, tqdm(arg_list, total=len(arg_list), desc=f'Spliting history files'))

    # def merge_ds(self, vn, comp, input_dirpath, output_dirpath, timespan=None, overwrite=False, nco=True):
    #     paths = sorted(glob.glob(os.path.join(input_dirpath, f'*.{vn}.*.nc')))
    #     if timespan is None:
    #         paths_sub = paths
    #     else:
    #         syr, eyr = timespan
    #         paths_sub = []
    #         for path in paths:
    #             year = int(path.split('.')[-2].split('-')[0])
    #             if (year >= syr and year <= eyr) or (year >= syr and year <= eyr):
    #                 paths_sub.append(path)

    #     date_start = ''.join(paths[0].split('.')[-2].split('-'))
    #     date_end = ''.join(paths[-1].split('.')[-2].split('-'))

    #     bn_elements = os.path.basename(paths[0]).split('.')
    #     bn_elements[-2] = f'{date_start}-{date_end}'

    #     if self.casename is not None:
    #         fname = '.'.join(bn_elements[-5:])
    #         fname = f'{self.casename}.{fname}'
    #     else:
    #         fname = '.'.join(bn_elements)
    #     out_path = os.path.join(output_dirpath, fname)

    #     if overwrite or not os.path.exists(out_path):
    #         if os.path.exists(out_path): os.remove(out_path)
    #         if nco:
    #             cmd = f'ncrcat {" ".join(paths_sub)} -o {out_path}'
    #             subprocess.run(cmd, shell=True)

    #         else:
    #             if comp == 'ocn':
    #                 ds =  xr.open_mfdataset(paths_sub, coords='minimal', data_vars=[vn], compat='override')
    #             else:
    #                 ds_list = [xr.load_dataset(path) for path in paths_sub]
    #                 ds = xr.concat(ds_list, dim='time', data_vars=[vn], coords='minimal')

    #             ds.to_netcdf(out_path)
    #             ds.close()

    # def bigcrunch(self, comp, input_dirpath, output_dirpath, timespan=None, overwrite=False, nproc=1, nco=True):
    #     output_dirpath = pathlib.Path(output_dirpath)
    #     if not output_dirpath.exists():
    #         output_dirpath.mkdir(parents=True, exist_ok=True)
    #         utils.p_success(f'>>> output directory created at: {output_dirpath}')

    #     desc = 'Merging variables'
    #     if nproc == 1:
    #         for vn in tqdm(self.vns[comp], desc=desc):
    #             self.merge_ds(vn, comp, input_dirpath=input_dirpath, output_dirpath=output_dirpath, timespan=timespan, overwrite=overwrite, nco=nco)
    #     else:
    #         utils.p_hint(f'>>> nproc: {nproc}')
    #         with mp.Pool(processes=nproc) as p:
    #             arg_list = []
    #             for vn in self.vns[comp]:
    #                 arg_list.append((vn, comp, input_dirpath, output_dirpath, timespan, overwrite, nco))
    #             p.starmap(self.merge_ds, tqdm(arg_list, total=len(arg_list), desc=desc))

    # def gen_ts(self, output_dirpath, comps=['atm', 'ocn', 'lnd', 'ice', 'rof'], timestep=50, timespan=None,
    #            dir_structure='comp/proc/tseries/month_1' , overwrite=False, nproc=1, nco=True):

    #     syr = timespan[0]
    #     nt = (timespan[-1] - timespan[0] + 1) // timestep
    #     timespan_list = []
    #     for i in range(nt):
    #         timespan_list.append((syr, syr+timestep-1))
    #         syr += timestep 

    #     for comp in comps:
    #         utils.p_header(f'>>> Processing component: {comp}')

    #         bigbang_dir = os.path.join(output_dirpath, f'.bigbang_{comp}')
    #         if os.path.exists(bigbang_dir): shutil.rmtree(bigbang_dir)
    #         self.bigbang(comp=comp, output_dirpath=bigbang_dir, timespan=timespan, overwrite=overwrite, nproc=nproc, nco=nco)

    #         bigcrunch_dir = os.path.join(output_dirpath, dir_structure.replace('comp', comp))
    #         self.bigcrunch(comp=comp, input_dirpath=bigbang_dir, output_dirpath=bigcrunch_dir, timespan=timespan, overwrite=overwrite, nproc=nproc, nco=nco)

    #     for comp in comps:
    #         utils.p_header(f'>>> Removing temporary files at: {bigbang_dir}')
    #         bigbang_dir = os.path.join(output_dirpath, f'.bigbang_{comp}')
    #         if os.path.exists(bigbang_dir): shutil.rmtree(bigbang_dir)




    # def _split_vn(self, path, comp, vn, output_dirpath, rewrite=False):
    #     date = os.path.basename(path).split('.')[-2]
    #     out_path = os.path.join(output_dirpath, f'{comp}.{vn}.{date}.nc')
    #     if rewrite or not os.path.exists(out_path):
    #         da = xr.open_dataset(path)[vn]
    #         da.to_netcdf(out_path)
    #         da.close()
    
    # def split_hist(self, comp, output_dirpath, timespan=None, nproc=1):
    #     output_dirpath = pathlib.Path(output_dirpath)
    #     if not output_dirpath.exists():
    #         output_dirpath.mkdir(parents=True, exist_ok=True)
    #         utils.p_success(f'>>> output directory created at: {output_dirpath}')

    #     paths = self.get_paths(comp, timespan=timespan)

    #     with mp.Pool(processes=nproc) as p:
    #         arg_list = []
    #         for path in paths:
    #             for vn in self.vns[comp]:
    #                 arg_list.append((path, comp, vn, f'{output_dirpath}/.{timespan[0]}-{timespan[1]}'))
    #         p.starmap(self._split_vn, tqdm(arg_list, total=len(arg_list), desc=f'Spliting history files'))

    # def get_ts(self, comp, vn, timespan=None, nproc=1):
    #     paths = self.get_paths(comp, timespan=timespan)

    #     with mp.Pool(processes=nproc) as p:
    #         arg_list = [(path, ) for path in paths]
    #         ds_list = p.starmap(xr.open_dataset, tqdm(arg_list, total=len(paths), desc=f'Loading history files'))

    #     ds = xr.concat(ds_list, dim='time', data_vars=[vn])

    #     da = ds[vn]
    #     ds_out = ds.drop_vars(self.vns[comp])
    #     ds_out[vn] = da
    #     return ds_out

    # def save_ts(self, comp, vn, output_dirpath,timespan=None, overwrite=False, casename=None):
    #     output_dirpath = pathlib.Path(output_dirpath)
    #     if not output_dirpath.exists():
    #         output_dirpath.mkdir(parents=True, exist_ok=True)
    #         utils.p_success(f'>>> output directory created at: {output_dirpath}')

    #     paths = self.get_paths(comp, timespan=timespan)
    #     date_start = ''.join(paths[0].split('.')[-2].split('-'))
    #     date_end = ''.join(paths[-1].split('.')[-2].split('-'))

    #     bn_elements = os.path.basename(self.paths[comp][0]).split('.')
    #     bn_elements[-2] = f'{date_start}-{date_end}'
    #     bn_elements.insert(-2, vn)

    #     if casename is not None:
    #         fname = '.'.join(bn_elements[-5:])
    #         fname = f'{casename}.{fname}'
    #     else:
    #         fname = '.'.join(bn_elements)

    #     out_path = os.path.join(output_dirpath, fname)

    #     if overwrite or not os.path.exists(out_path):
    #         ds = self.get_ts(comp, vn, timespan=timespan)
    #         ds.to_netcdf(out_path)
    #         ds.close()

    # def gen_ts(self, output_dirpath, comp=None, vns=None, timestep=50, timespan=None, dir_structure=None, overwrite=False,
    #            nproc=1, casename=None):
    #     if comp is None: raise ValueError('Please specify component via the argument `comp`.')
    #     if timespan is None: raise ValueError('Please specify timespan via the argument `timespan`.')
    #     if vns is None: vns = self.get_real_vns(comp)
    #     if dir_structure is None: dir_structure = f'{comp}/proc/tseries/month_1' 
    #     output_dirpath = os.path.join(output_dirpath, dir_structure)

    #     syr = timespan[0]
    #     nt = (timespan[-1] - timespan[0] + 1) // timestep
    #     timespan_list = []
    #     for i in range(nt):
    #         timespan_list.append((syr, syr+timestep-1))
    #         syr += timestep 

    #     utils.p_header(f'>>> Generating timeseries for {len(vns)} variables:')
    #     for i in range(len(vns)//10+1):
    #         print(vns[10*i:10*i+10])

    #     if nproc == 1:
    #         for v in tqdm(vns, total=len(vns), desc=f'Generating timeseries files'):
    #             for span in timespan_list:
    #                 self.save_ts(comp, v, output_dirpath=output_dirpath, timespan=span, overwrite=overwrite, casename=casename)
    #     else:
    #         utils.p_hint(f'>>> nproc: {nproc}')
    #         with mp.Pool(processes=nproc) as p:
    #             arg_list = []
    #             for v in vns:
    #                 for span in timespan_list:
    #                     arg_list.append((comp, v, output_dirpath, span, overwrite, casename))

    #             p.starmap(self.save_ts, tqdm(arg_list, total=len(vns)*len(timespan_list), desc=f'Generating timeseries files'))

    #     utils.p_success(f'>>> {len(timespan_list)*len(vns)} climo files created in: {output_dirpath}')

    def rm_timespan(self, timespan, comps=['atm', 'ice', 'ocn', 'rof', 'lnd'], nworkers=None, rehearsal=True):
        ''' Rename the archive files within a timespan

        Args:
            timespan (tuple or list): [start_year, end_year] with elements being integers
        '''
        if nworkers is None:
            nworkers = threading.active_count()
            utils.p_header(f'nworkers = {nworkers}')

        start_year, end_year = timespan
        year_list = []
        for y in range(start_year, end_year+1):
            year_list.append(f'{y:04d}')

        def rm_path(year, comp=None, rehearsal=True):
            if rehearsal:
                if comp is None:
                    cmd = f'ls {self.root_dir}/*/hist/*{year}-[01][0-9][-.]*'
                else:
                    cmd = f'ls {self.root_dir}/{comp}/hist/*{year}-[01][0-9][-.]*'
            else:
                if comp is None:
                    cmd = f'rm -f {self.root_dir}/*/hist/*{year}-[01][0-9][-.]*'
                else:
                    cmd = f'rm -f {self.root_dir}/{comp}/hist/*{year}-[01][0-9][-.]*'

            subprocess.run(cmd, shell=True)
            
        if comps == ['atm', 'ice', 'ocn', 'rof', 'lnd']:
            with tqdm(desc=f'Removing files for year', total=len(year_list)) as pbar:
                with ThreadPoolExecutor(nworkers) as exe:
                    futures = [exe.submit(rm_path, year, comp=None, rehearsal=rehearsal) for year in year_list]
                    [pbar.update(1) for future in as_completed(futures)]
        else:
            for comp in comps:
                utils.p_header(f'Processing {comp} ...')
                with tqdm(desc=f'Removing files for year #', total=len(year_list)) as pbar:
                    with ThreadPoolExecutor(nworkers) as exe:
                        futures = [exe.submit(rm_path, year, comp=comp, rehearsal=rehearsal) for year in year_list]
                        [pbar.update(1) for future in as_completed(futures)]



class Timeseries:
    ''' Initialize a CESM Timeseries case generated by CESM Postprocessing

    Args:
        root_dir (str): the root directory of the CESM Timeseries output
        grid_dict (dict): the grid dictionary for different components
        timestep (int): the number of years stored in a single timeseries file
    '''
    def __init__(self, root_dir, grid_dict=None, casename=None):
        self.path_pattern='comp/proc/tseries/month_1/casename.mdl.h_str.vn.timespan.nc'
        self.root_dir = os.path.abspath(root_dir)
        self.casename = casename

        self.grid_dict = {'atm': 'ne30pg3', 'ocn': 'g16'}
        if grid_dict is not None:
            self.grid_dict.update(grid_dict)

        self.grid_dict['lnd'] = self.grid_dict['atm']
        self.grid_dict['rof'] = self.grid_dict['atm']
        self.grid_dict['ice'] = self.grid_dict['ocn']

        utils.p_header(f'>>> case.root_dir: {self.root_dir}')
        utils.p_header(f'>>> case.path_pattern: {self.path_pattern}')
        utils.p_header(f'>>> case.grid_dict: {self.grid_dict}')
        if self.casename is not None:
            utils.p_header(f'>>> case.casename: {self.casename}')

        self.paths = utils.find_paths(self.root_dir, self.path_pattern)

        self.ds = {}
        self.diags = {}
        self.vars_info = {}
        for path in self.paths:
            comp = path.split('/')[-5]
            mdl = path.split('.')[-5]
            h_str = path.split('.')[-4]
            vn = path.split('.')[-3]
            if (vn, comp) not in self.vars_info:
                self.vars_info[(vn, comp)] = (comp, mdl, h_str)

        utils.p_success(f'>>> case.vars_info created')

    def get_paths(self, vn, comp=None, timespan=None):
        if comp is None: comp = self.get_vn_comp(vn)
        comp, mdl, h_str = self.vars_info[(vn, comp)]
        paths = utils.find_paths(self.root_dir, self.path_pattern, vn=vn, comp=comp, mdl=mdl, h_str=h_str)
        if timespan is None:
            paths_sub = paths
        else:
            syr, eyr = timespan
            paths_sub = []
            for path in paths:
                syr_tmp = int(path.split('.')[-2].split('-')[0][:4])
                eyr_tmp = int(path.split('.')[-2].split('-')[1][:4])
                if (syr_tmp >= syr and syr_tmp <= eyr) or (eyr_tmp >= syr and eyr_tmp <= eyr):
                    paths_sub.append(path)

        return paths_sub

    def get_vn_comp(self, vn):
        comps = []
        for (v, comp) in self.vars_info:
            if v == vn:
                comps.append(comp)
        
        if len(comps) == 1:
            return comps[0]
        elif len(comps) == 0:
            if f'get_{vn}' in diags.DiagCalc.__dict__:
                utils.p_warning(f'>>> {vn} is a supported derived variable.')
            else:
                raise ValueError('The input variable name is unknown.')
        else:
            utils.p_warning(f'{vn} belongs to components: {comps}')
            raise ValueError('The input variable name belongs to multiple components. Please specify via the argument `comp`.')

    
    def load(self, vn, comp=None, timespan=None, load_idx=-1, adjust_month=True, verbose=True, **kws):
        if comp is None:
            comp = self.get_vn_comp(vn)

        if (vn, comp) in self.vars_info:
            if timespan is None:
                paths = self.get_paths(vn, comp=comp)[load_idx]
            else:
                paths = self.get_paths(vn, comp=comp, timespan=timespan)

            if vn in self.ds:
                if self.ds[vn].path != paths:
                    if verbose: utils.p_warning(f'>>> case.ds["{vn}"] will be reloaded due to different paths.')
                    self.clear_ds(vn)

            if vn not in self.ds:
                comp, mdl, h_str = self.vars_info[(vn, comp)]

                _kws = {
                   'vn': vn,
                   'adjust_month': adjust_month, 
                   'comp': comp,
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
            # else:
            #     if verbose: utils.p_warning(f'>>> case.ds["{vn}"] already loaded; to reload, run case.clear_ds("{vn}") before case.load("{vn}")')

        else:
            if verbose: utils.p_warning(f'>>> Variable {vn} not existing')

    def calc(self, spell:str, comp=None, timespan=None, load_idx=-1, adjust_month=True, recalculate=False, verbose=True):
        ''' Calculate a diagnostic spell
        '''
        if spell in self.diags and not recalculate:
            utils.p_warning(f'>>> Spell `{spell}` is already calculated and the calculation is skipped.')
        else:
            S = Spell(spell)
            if S.slicing is None:
                vn = S.vn
            else:
                vn = S.vn.split('.')[0]

            if vn in self.diags:
                da = self.diags[vn]
                utils.p_warning(f'>>> Variable `{vn}` is already calculated and the calculation is skipped.')
            else:
                if comp is None: comp = self.get_vn_comp(vn)
                if f'get_{vn}' in diags.DiagCalc.__dict__:
                    da = diags.DiagCalc.__dict__[f'get_{vn}'](self, timespan=timespan, load_idx=load_idx, adjust_month=adjust_month, verbose=verbose)
                elif (vn, comp) in self.vars_info:
                    self.load(vn, comp=comp, timespan=timespan, load_idx=load_idx, adjust_month=adjust_month, verbose=verbose)
                    da = self.ds[vn].x.da
                else:
                    raise ValueError(f'Unknown diagnostic variable: {vn}')

            if S.regrid is not None:
                da = eval(f'da.x.{S.regrid}')

            if S.slicing is not None:
                cmd = f'da.{S.slicing}'
                da = eval(cmd)

            if S.plev is not None:
                self.load('PS')
                PS = self.ds['PS']['PS']
                hyam = self.ds[vn]['hyam']
                hybm = self.ds[vn]['hybm']
                _kws = {'lev_dim': 'lev'}
                if '(' in S.plev and ')' in S.plev:
                    new_levels = eval(S.plev.split('plev')[-1])
                    if type(new_levels) not in (list, tuple):
                        new_levels = [new_levels]

                    _kws.update({'new_levels': np.array(new_levels)})

                da = da.x.get_plev(ps=PS, hyam=hyam, hybm=hybm, **_kws)

            if S.ann_method is not None:
                utils.p_hint(f'>>> Timespan: [{da.time.values[0]}, {da.time.values[-1]}]')
                da = utils.ann_modifier(da, ann_method=S.ann_method, long_name=da.long_name)

            if S.sa_method is not None:
                if S.sa_method in ['gm', 'nhm', 'shm', 'zm', 'gs', 'nhs', 'shs', 'somin']:
                    da = getattr(da.x, S.sa_method)
                elif S.sa_method == 'yz':
                    if da.name == 'MOC':
                        da = da
                    else:
                        da = da.x.zm
                else:
                    raise ValueError(f'Unknown spatial average method: {S.sa_method}')

            if S.zavg is not None:
                da = eval(f'da.x.{S.zavg}')

            if da.units == 'degC':
                da.attrs['units'] = '°C'
            elif da.units == 'K':
                da -= 273.15
                da.attrs['units'] = '°C'

            if S.alias is not None:
                spell = S.alias
                da.name = S.alias

            self.diags[spell] = da.squeeze().compute()
            if verbose: utils.p_success(f'>>> case.diags["{spell}"] created')
        return self.diags[spell]

    def plot(self, spell, t_idx=None, regrid=False, gs='T', ssv='SSH', recalculate_ssv=False, timespan=None, **kws):
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
            if (ssv, 'ocn') in self.vars_info and recalculate_ssv or ('ssv' not in self.diags):
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

        for k, v in spells.items():
            utils.p_header(f'>>> Plotting {k}')
            if len(self.diags[v].dims) == 1 and 'time' in self.diags[v].dims:
                # timeseries
                if '_clim' in k:
                    self.plot(v, ax=ax[k], title=title_dict[k], color=clr_dict[k], alpha=1)
                else:
                    self.plot(v, ax=ax[k], title=title_dict[k], color=clr_dict[k], alpha=0.1)
                    vals = self.diags[v].rolling(time=roll_int, center=True).mean().data
                    ax[k].plot(self.diags[v].time, vals, color=clr_dict[k])

                if timespan is not None and 'climo_period' not in self.diags[v].attrs:
                    start_date = cftime.DatetimeNoLeap(timespan[0], 1, 1)
                    end_date = cftime.DatetimeNoLeap(timespan[1], 1, 1)
                    ax[k].set_xlim(start_date, end_date)

            elif k in ['TS']:
                self.plot(v, ax=ax[k], title=title_dict[k], cbar_kwargs={'orientation': 'horizontal', 'aspect': 20, 'pad': 0.05})
            else:
                self.plot(v, ax=ax[k], title=title_dict[k])

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
                    color=clr_dict[k],
                    fontsize=15,
                )

        return fig, ax

    def get_climo(self, vn, comp=None, timespan=None, adjust_month=True, slicing=False, regrid=False, dlat=1, dlon=1):
        ''' Generate the climatology file for the given variable

        Args:
            slicing (bool): could be problematic
        '''
        if comp is None: comp = self.get_vn_comp(vn)
        grid = self.grid_dict[comp]
        paths = self.get_paths(vn, comp=comp, timespan=timespan)
        ds = core.open_mfdataset(paths, adjust_month=adjust_month)

        if slicing: ds = ds.sel(time=slice(timespan[0], timespan[1]))
        ds_out = ds.x.climo
        ds_out.attrs['comp'] = comp
        ds_out.attrs['grid'] = grid
        if regrid: ds_out = ds_out.x.regrid(dlat=dlat, dlon=dlon)
        return ds_out

    def save_climo(self, output_dirpath, vn, comp=None, timespan=None, adjust_month=True,
                   slicing=False, regrid=False, dlat=1, dlon=1, overwrite=False):

        output_dirpath = pathlib.Path(output_dirpath)
        if not output_dirpath.exists():
            output_dirpath.mkdir(parents=True, exist_ok=True)
            utils.p_success(f'>>> output directory created at: {output_dirpath}')

        fname = f'{vn}_climo.nc' if self.casename is None else f'{self.casename}_{vn}_climo.nc'
        out_path = os.path.join(output_dirpath, fname)
        if overwrite or not os.path.exists(out_path):
            if os.path.exists(out_path): os.remove(out_path)
            if comp is None: comp = self.get_vn_comp(vn)

            climo = self.get_climo(
                vn, comp=comp, timespan=timespan, adjust_month=adjust_month,
                slicing=slicing, regrid=regrid, dlat=dlat, dlon=dlon,
            )
            climo.to_netcdf(out_path)
            climo.close()

    def gen_climo(self, output_dirpath, comp=None, timespan=None, vns=None, adjust_month=True,
                  nproc=1, slicing=False, regrid=False, dlat=1, dlon=1, overwrite=False):

        if comp is None:
            raise ValueError('Please specify component via the argument `comp`.')

        if vns is None:
            vns = [k[0] for k, v in self.vars_info.items() if v[0]==comp]

        utils.p_header(f'>>> Generating climo for {len(vns)} variables:')
        for i in range(len(vns)//10+1):
            print(vns[10*i:10*i+10])

        if nproc == 1:
            for v in tqdm(vns, total=len(vns), desc=f'Generating climo files'):
                self.save_climo(
                    output_dirpath, v, comp=comp, timespan=timespan,
                    adjust_month=adjust_month, slicing=slicing,
                    regrid=regrid, dlat=dlat, dlon=dlon,
                    overwrite=overwrite,
                )
        else:
            utils.p_hint(f'>>> nproc: {nproc}')
            with mp.Pool(processes=nproc) as p:
                arg_list = [(output_dirpath, v, comp, timespan, adjust_month, slicing, regrid, dlat, dlon, overwrite) for v in vns]
                p.starmap(self.save_climo, tqdm(arg_list, total=len(vns), desc=f'Generating climo files'))

        utils.p_success(f'>>> {len(vns)} climo files created in: {output_dirpath}')

    # def save_combined_ts(self, output_dirpath, comp, vns=None, timespan=None, adjust_month=True, overwrite=False, chunk_nt=None):
    #     output_dirpath = pathlib.Path(output_dirpath)
    #     if not output_dirpath.exists():
    #         output_dirpath.mkdir(parents=True, exist_ok=True)
    #         utils.p_success(f'>>> output directory created at: {output_dirpath}')

    #     if vns is None:
    #         vns = [k[0] for k, v in self.vars_info.items() if v[0]==comp]

    #     utils.p_header(f'>>> Combining timeseries files for {len(vns)} variables:')
    #     for i in range(len(vns)//10+1):
    #         print(vns[10*i:10*i+10])

    #     fname = f'{timespan[0]}_{timespan[1]}_ts.nc' if self.casename is None else f'{self.casename}_{timespan[0]}_{timespan[1]}_ts.nc'
    #     out_path = os.path.join(output_dirpath, fname)
    #     if overwrite or not os.path.exists(out_path):
    #         paths_list = []
    #         for vn in vns:
    #             paths = self.get_paths(vn, comp=comp, timespan=timespan)
    #             paths_list.append(*paths)

    #         if chunk_nt is None:
    #             ds = core.open_mfdataset(paths_list, adjust_month=adjust_month, coords='minimal', data_vars='minimal')
    #         else:
    #             ds = core.open_mfdataset(paths_list, adjust_month=adjust_month, coords='minimal', data_vars='minimal', chunks={'time': chunk_nt})

    #         # ds.attrs['comp'] = comp
    #         # ds.attrs['grid'] = self.grid_dict[comp]
    #         # ds.to_netcdf(out_path)
    #         # ds.close()
    #         # utils.p_success(f'>>> Combined timeseries file created at: {out_path}')

    def get_mean(self, vn, comp, months=list(range(1, 13)), timespan=None, adjust_month=True, slicing=False, regrid=False, dlat=1, dlon=1):
        grid = self.grid_dict[comp]
        paths = self.get_paths(vn, comp=comp, timespan=timespan)
        ds = core.open_mfdataset(paths, adjust_month=adjust_month)

        if slicing: ds = ds.sel(time=slice(timespan[0], timespan[1]))
        ds_out = ds.x.annualize(months=months)
        ds_out.attrs['comp'] = comp
        ds_out.attrs['grid'] = grid
        if regrid: ds_out = ds_out.x.regrid(dlat=dlat, dlon=dlon)
        return ds_out

    def get_ts(self, vn, comp, timespan=None, adjust_month=True, slicing=False, regrid=False, dlat=1, dlon=1):
        grid = self.grid_dict[comp]
        paths = self.get_paths(vn, comp=comp, timespan=timespan)
        ds = core.open_mfdataset(paths, adjust_month=adjust_month)

        if slicing: ds = ds.sel(time=slice(timespan[0], timespan[1]))

        ds_out = ds
        ds_out.attrs['comp'] = comp
        ds_out.attrs['grid'] = grid
        if regrid: ds_out = ds_out.x.regrid(dlat=dlat, dlon=dlon)
        return ds_out

    def save_means(self, vn, comp, output_dirpath, timespan, adjust_month=True, slicing=False, regrid=False, dlat=1, dlon=1, overwrite=False):
        output_dirpath = pathlib.Path(output_dirpath)
        if not output_dirpath.exists():
            output_dirpath.mkdir(parents=True, exist_ok=True)
            utils.p_success(f'>>> output directory created at: {output_dirpath}')

        ds = self.get_ts(vn, comp, timespan=timespan, adjust_month=adjust_month, slicing=slicing, regrid=False)

        sn_dict = {
            'ANN': list(range(1, 13)),
            'DJF': [12, 1, 2],
            'MAM': [1, 2, 3],
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
                ds_ann.to_netcdf(out_path)
                ds_ann.close()

    def gen_means(self, output_dirpath, comp=None, vns=None, timespan=None, adjust_month=True, slicing=False,
                  regrid=False, dlat=1, dlon=1, overwrite=False, nproc=1):

        if comp is None:
            raise ValueError('Please specify component via the argument `comp`.')

        if vns is None:
            vns = [k[0] for k, v in self.vars_info.items() if v[0]==comp]

        utils.p_header(f'>>> Generating seaonal means for {len(vns)} variables:')
        for i in range(len(vns)//10+1):
            print(vns[10*i:10*i+10])

        if nproc == 1:
            for vn in vns:
                self.save_means(
                    vn, comp, output_dirpath, timespan, adjust_month=adjust_month, slicing=slicing,
                    regrid=regrid, dlat=dlat, dlon=dlon, overwrite=overwrite, 
                )
        else:
            utils.p_hint(f'>>> nproc: {nproc}')
            with mp.Pool(processes=nproc) as p:
                arg_list = [(vn, comp, output_dirpath, timespan, adjust_month, slicing, regrid, dlat, dlon, overwrite) for vn in vns]
                p.starmap(self.save_means, tqdm(arg_list, total=len(vns), desc=f'Generating seasonal mean files'))

    def check_timespan(self, comp, vns=None, timespan=None):
        if vns is None:
            vns = [k[0] for k, v in self.vars_info.items() if v[0]==comp]
        elif type(vns) is str:
            vns = [vns]

        paths = self.get_paths(vns[0], comp=comp, timespan=timespan)
        if timespan is None:
            syr = int(paths[0].split('.')[-2].split('-')[0][:4])
            eyr = int(paths[-1].split('.')[-2].split('-')[1][:4])
        else:
            syr, eyr = timespan

        step_s = int(paths[0].split('.')[-2].split('-')[0][:4])
        step_e = int(paths[0].split('.')[-2].split('-')[1][:4])
        step = step_e - step_s + 1

        full_list = []
        for y in range(syr, eyr, step):
            full_list.append(f'{y:04d}01-{y+step-1:04d}12')

        df = pd.DataFrame(index=vns, columns=range(1, len(full_list)+1))

        for irow, vn in enumerate(vns):
            paths = self.get_paths(vn, comp=comp, timespan=timespan)
            # print(vn, paths)

            for icol, timestamp in enumerate(full_list):
                path_elements = paths[icol].split('.')
                path_elements[-2] = timestamp
                path = '.'.join(path_elements)
                if os.path.exists(path):
                    df.iloc[irow, icol] = timestamp
                else:
                    print(path)
                    df.iloc[irow, icol] = f'{timestamp}!'

        df = df.fillna('!')
        
        def style_missing(v, props=''):
            return props if '!' in v else None
        
        def remove_mark(v):
            if '!' in v:
                v = v.split('!')[0]
            return v

        df = df.style.map(style_missing, props='background-color:red;color:white').format(remove_mark)
        return df

    def clear_ds(self, vn=None):
        ''' Clear the existing `.ds` property
        '''
        if vn is not None:
            try:
                self.ds.pop(vn)
            except:
                pass
        else:
            self.ds = {}
        
    def copy(self):
        return deepcopy(self)

    # @dask.delayed
    # def save_spell(self, spell:str, vn:str, output_path:str, timespan=None, overwrite=True, long_name=None, **kws):
    #     case = self.copy()
    #     if overwrite or not os.path.exists(output_path):
    #         case.calc(spell, timespan=timespan)
    #         case.diags[spell].name = vn
    #         da = case.diags[spell]
    #         if long_name is not None: da.attrs['long_name'] = long_name
    #         if os.path.exists(output_path): os.remove(output_path)
    #         da.x.to_netcdf(output_path, **kws)

    # def gen_ts_spell(self, spell:str, vn:str, comp:str, output_dirpath:str, long_name=None, timespan=None, timestep=50, overwrite=True):
    #     ''' Generate timeseries based on a spell
    #     '''
    #     _mdl_hstr_dict = {
    #         'atm': ('cam', 'h0'),
    #         'ocn': ('pop', 'h'),
    #         'lnd': ('clm2', 'h0'),
    #         'ice': ('cice', 'h'),
    #         'rof': ('rtm', 'h0'),
    #     }
    #     mdl, hstr = _mdl_hstr_dict[comp]
    #     path_tmp = 'comp/proc/tseries/month_1/'.replace('comp', comp)
    #     output_path = os.path.join(output_dirpath, path_tmp)
    #     output_dir = pathlib.Path(os.path.dirname(output_path))
    #     output_dir.mkdir(parents=True, exist_ok=True)

    #     if timespan is None: raise ValueError('Please specify `timespan`.')

    #     syr = timespan[0]
    #     nt = (timespan[-1] - timespan[0] + 1) // timestep
    #     timespan_list = []
    #     for i in range(nt):
    #         timespan_list.append((syr, syr+timestep-1))
    #         syr += timestep 

    #     tasks = []
    #     for timespan_tmp in timespan_list:
    #         utils.p_header(f'>>> Processing timespan: {timespan_tmp}')
    #         filename = 'casename.mdl.h_str.vn.timespan.nc'.replace('casename', self.casename).replace('mdl', mdl).replace('h_str', hstr).replace('vn', vn).replace('timespan', f'{timespan_tmp[0]:04d}01-{timespan_tmp[1]:04d}12')
    #         output_path = os.path.join(output_dir, filename)
    #         tasks.append(self.save_spell(spell, vn, timespan=timespan_tmp, long_name=long_name, output_path=output_path, overwrite=overwrite))

    #     dask.compute(*tasks)

        
    
    # def gen_ts_spell(self, spell:str, vn:str, comp:str, output_dirpath:str, long_name=None, timespan=None, timestep=50, overwrite=True, nproc=1):
    #     ''' Generate timeseries based on a spell
    #     '''
    #     _mdl_hstr_dict = {
    #         'atm': ('cam', 'h0'),
    #         'ocn': ('pop', 'h'),
    #         'lnd': ('clm2', 'h0'),
    #         'ice': ('cice', 'h'),
    #         'rof': ('rtm', 'h0'),
    #     }
    #     mdl, hstr = _mdl_hstr_dict[comp]
    #     path_tmp = 'comp/proc/tseries/month_1/'.replace('comp', comp)
    #     output_path = os.path.join(output_dirpath, path_tmp)
    #     output_dir = pathlib.Path(os.path.dirname(output_path))
    #     output_dir.mkdir(parents=True, exist_ok=True)

    #     if timespan is None: raise ValueError('Please specify `timespan`.')

    #     syr = timespan[0]
    #     nt = (timespan[-1] - timespan[0] + 1) // timestep
    #     timespan_list = []
    #     for i in range(nt):
    #         timespan_list.append((syr, syr+timestep-1))
    #         syr += timestep 

    #     # generate timeseries files for each sub-timespan
    #     if nproc == 1:
    #         for timespan_tmp in timespan_list:
    #             utils.p_header(f'>>> Processing timespan: {timespan_tmp}')
    #             filename = 'casename.mdl.h_str.vn.timespan.nc'.replace('casename', self.casename).replace('mdl', mdl).replace('h_str', hstr).replace('vn', vn).replace('timespan', f'{timespan_tmp[0]:04d}01-{timespan_tmp[1]:04d}12')
    #             output_path = os.path.join(output_dir, filename)
    #             self.save_spell(spell, vn, timespan=timespan_tmp, long_name=long_name, output_path=output_path, overwrite=overwrite)
    #     else:
    #         utils.p_hint(f'>>> nproc: {nproc}')
    #         with mp.Pool(processes=nproc) as p:
    #             arg_list = []
    #             for timespan_tmp in timespan_list:
    #                 filename = 'casename.mdl.h_str.vn.timespan.nc'.replace('casename', self.casename).replace('mdl', mdl).replace('h_str', hstr).replace('vn', vn).replace('timespan', f'{timespan_tmp[0]:04d}01-{timespan_tmp[1]:04d}12')
    #                 output_path = os.path.join(output_dir, filename)
    #                 arg_list.append((spell, vn, output_path, timespan_tmp,  overwrite, long_name))
    #             p.starmap(self.save_spell, tqdm(arg_list, total=len(arg_list), desc=f'Saving "{spell}" to files'))

class Climo:
    def __init__(self, root_dir, casename):
        self.root_dir = root_dir
        self.casename = casename
        utils.p_header(f'>>> case.root_dir: {self.root_dir}')
        utils.p_header(f'>>> case.casename: {self.casename}')

    def gen_MONS_climo(self, output_dirpath, climo_period=None):
        output_dirpath = pathlib.Path(output_dirpath)
        if not output_dirpath.exists():
            output_dirpath.mkdir(parents=True, exist_ok=True)
            utils.p_header(f'>>> output directory created at: {output_dirpath}')

        paths = sorted(glob.glob(os.path.join(self.root_dir, '*_climo.nc')))
        ds = core.open_mfdataset(paths)
        if climo_period is None:
            try:
                climo_period = ds.attrs['climo_period']
            except:
                pass

        if climo_period is not None:
            fname = f'{self.casename}_{climo_period[0]}_{climo_period[1]}_MONS_climo.nc'
        else:
            fname = f'{self.casename}_MONS_climo.nc'

        output_fpath = os.path.join(output_dirpath, fname)
        ds.to_netcdf(output_fpath)
        utils.p_header(f'>>> MONS_climo generated at: {output_fpath}')
        self.MONS_climo_path = output_fpath
        utils.p_success(f'>>> case.MONS_climo_path created')

    def gen_seasons_climo(self, MONS_climo_path=None):
        if MONS_climo_path is None:
            MONS_climo_path = self.MONS_climo_path

        ds = xr.open_dataset(MONS_climo_path)

        sn_dict = {
            'ANN': list(range(1, 13)),
            'DJF': [12, 1, 2],
            'MAM': [1, 2, 3],
            'JJA': [6, 7, 8],
            'SON': [9, 10, 11],
        }

        for sn, months in sn_dict.items():
            output_fpath = MONS_climo_path.replace('MONS', sn)
            if os.path.exists(output_fpath):
                os.remove(output_fpath)

            ds_mean = ds.sel(time=months).mean('time').expand_dims('time')
            ds_mean = ds_mean.assign_coords(time=[ds.coords['time'][0]])
            ds_mean.to_netcdf(output_fpath, unlimited_dims={'time':True})
            utils.p_header(f'>>> {sn}_climo generated at: {output_fpath}')

class Means:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        utils.p_header(f'>>> case.root_dir: {self.root_dir}')

    def merge_means(self, sn, output_dirpath, overwrite=False, casetag=None):
        utils.p_header(f'>>> Processing season {sn}')
        paths = glob.glob(os.path.join(self.root_dir, sn, f'*_{sn}_means.nc'))
        if casetag is None:
            fname = f'{sn}_means.nc'
        else:
            fname = f'{casetag}_{sn}_means.nc'
        out_path = os.path.join(output_dirpath, fname)
        if overwrite or not os.path.exists(out_path):
            # if chunk_nt is not None:
            #     ds = xr.open_mfdataset(paths, compat='override', coords='minimal', data_vars='minimal', chunks={'time': chunk_nt})
            # else:
            #     ds = xr.open_mfdataset(paths, compat='override', coords='minimal', data_vars='minimal')

            ds_list = []
            for path in paths:
                ds_tmp = core.open_dataset(path)
                for k, v in ds_tmp.coords.items():
                    try:
                        if any(np.isnan(v.values)):
                            ds_tmp = ds_tmp.drop_vars(k)
                    except:
                        pass
                ds_list.append(ds_tmp)

            # ds = xr.open_dataset(paths[0])
            # for path in tqdm(paths[1:]):
            #     ds_tmp = core.open_dataset(path)
            #     ds = xr.merge([ds, ds_tmp])
            
            utils.p_header(f'>>> Merging files')
            ds = xr.merge(ds_list)
            ds.to_netcdf(out_path)
            utils.p_header(f'>>> Merged mean file saved at: {out_path}')
        else:
            utils.p_warning(f'>>> The result already exists. Skipping ...')

    def merge_means_nproc(self, output_dirpath, sns=['ANN', 'DJF', 'MAM', 'JJA', 'SON'], overwrite=False, casetag=None, nproc=1):
        if nproc == 1:
            for sn in sns:
                self.merge_means(sn, output_dirpath, overwrite=overwrite, casetag=casetag)
        else:
            utils.p_hint(f'>>> nproc: {nproc}')
            with mp.Pool(processes=nproc) as p:
                arg_list = [(sn, output_dirpath, overwrite, casetag) for sn in sns]
                p.starmap(self.merge_means, tqdm(arg_list, total=len(sns), desc=f'Merging mean files'))


    # def load(self, vn, adjust_month=True, load_idx=-1, regrid=False):
    #     ''' Load a specific variable
        
    #     Args:
    #         vn (str or list): a variable name, or a list of variable names
    #         adjust_month (bool): adjust the month of the `xarray.Dataset` (the default CESM output has a month shift)
    #         load_idx (int or slice): -1 means to load the last file
    #         regrid (bool): if True, will regrid to regular lat/lon grid
    #     '''
    #     if not isinstance(vn, (list, tuple)):
    #         vn = [vn]

    #     for v in vn:
    #         # if v in ['KMT', 'z_t', 'z_w', 'dz', 'dzw']:
    #         #     vn_tmp = 'SSH'
    #         #     comp, mdl, h_str = self.vars_info[vn_tmp]
    #         #     paths = sorted(glob.glob(
    #         #         os.path.join(
    #         #             self.root_dir,
    #         #             self.path_pattern \
    #         #                 .replace('comp', comp) \
    #         #                 .replace('casename', '*') \
    #         #                 .replace('mdl', mdl) \
    #         #                 .replace('h_str', h_str) \
    #         #                 .replace('vn', vn_tmp) \
    #         #                 .replace('timespan', '*'),
    #         #         )
    #         #     ))
    #         #     with xr.load_dataset(paths[-1], decode_cf=False) as ds:
    #         #         self.ds[v] = ds.x[v]

    #         if v in self.vars_info:
    #             comp, mdl, h_str = self.vars_info[v]
    #             paths = sorted(glob.glob(
    #                 os.path.join(
    #                     self.root_dir,
    #                     self.path_pattern \
    #                         .replace('comp', comp) \
    #                         .replace('casename', '*') \
    #                         .replace('mdl', mdl) \
    #                         .replace('h_str', h_str) \
    #                         .replace('vn', v) \
    #                         .replace('timespan', '*'),
    #                 )
    #             ))

    #             # remove loaded object
    #             if v in self.ds:
    #                 if (regrid and 'regridded' not in self.ds[v].attrs) or (not regrid and 'regridded' in self.ds[v].attrs):
    #                     self.clear_ds(v)
    #                     utils.p_warning(f'>>> case.ds["{v}"] already loaded but will be reloaded due to a different regrid status')

    #             if v in self.ds:
    #                 if (load_idx is not None) and (paths[load_idx] != self.ds[v].attrs['path']):
    #                     self.clear_ds(v)
    #                     utils.p_warning(f'>>> case.ds["{v}"] already loaded but will be reloaded due to a different `load_idx`')
                
    #             # new load
    #             if v not in self.ds:
    #                 if load_idx is not None:
    #                     # ds =  core.load_dataset(paths[load_idx], vn=v, adjust_month=adjust_month, comp=comp, grid=self.grid_dict[comp])
    #                     ds =  core.open_dataset(paths[load_idx], vn=v, adjust_month=adjust_month, comp=comp, grid=self.grid_dict[comp])
    #                 else:
    #                     ds =  core.open_mfdataset(paths, vn=v, adjust_month=adjust_month, comp=comp, grid=self.grid_dict[comp])

    #                 if regrid:
    #                     self.ds[v] = ds.x.regrid()
    #                     self.ds[v].attrs.update({'regridded': True})
    #                 else:
    #                     self.ds[v] = ds

    #                 self.ds[v].attrs['vn'] = v
    #                 utils.p_success(f'>>> case.ds["{v}"] created')

    #             elif v in self.ds:
    #                 utils.p_warning(f'>>> case.ds["{v}"] already loaded; to reload, run case.clear_ds("{v}") before case.load("{v}")')

    #         else:
    #             utils.p_warning(f'>>> Variable {v} not existed')

        
    # def calc(self, spell, load_idx=-1, adjust_month=True, **kws):
    #     ''' Calculate a diagnostic spell

    #     Args:
    #         spell (str): The diagnostic spell in the format of `plot_type:diag_name:ann_method[:sm_method]`, where
    #             The `plot_type` supports:
                
    #                 * `ts`: timeseries plots
    #                 * `map`: 2D horizonal spatial plots
    #                 * `zm`: zonal mean plots
    #                 * `yz`: 2D lat-depth spatial plots
            
    #             The `plot_type:diag_name` combination supports:

    #                 * `ts:GMST`: the global mean surface temperature (GMST) timeseries
    #                 * `ts:MOC`: the meridional ocean circulation (MOC) timeseries
    #                 * `map:TS`: the surface temperature (TS) 2D map
    #                 * `map:LST`: the land surface temperature (LST) 2D map
    #                 * `map:SST`: the sea surface temperature (SST) 2D map
    #                 * `map:MLD`: the mixed layer depth (MLD) 2D map
    #                 * `zm:LST`: the LST 2D map
    #                 * `zm:SST`: the SST 2D map
    #                 * `yz:MOC`: the lat-depth MOC 2D map

    #             The `ann_method` supports:
                
    #                 * `ann`: calendar year annual mean
    #                 * `<m>`: a number in [..., -11, -12, 1, 2, ..., 12] representing a month
    #                 * `<m1>,<m2>,...`: a list of months sepearted by commmas
                    
    #             The `sm_method` supports:
                
    #                 * `gm`: global mean
    #                 * `nhm`: NH mean
    #                 * `shm`: SH mean

    #             For example,
                
    #                 * `ts:GMST:ann`: annual mean GMST timeseries
    #                 * `ts:SHH:ann:shm`: annual mean SH mean SHH timeseries
    #                 * `map:TS:-12,1,2`: DJF mean TS 2D map
    #                 * `map:MLD:3`: March MLD 2D map
    #                 * `zm:LST:6,7,8,9`: JJAS LST zonal mean

    #     '''
    #     spell_elements = spell.split(':')
    #     if len(spell_elements) == 3:
    #         plot_type, diag_name, ann_method = spell_elements
    #     elif len(spell_elements) == 4:
    #         plot_type, diag_name, ann_method, sm_method = spell_elements
    #     else:
    #         raise ValueError('Wrong diagnostic spell.')

    #     func_name = f'calc_{plot_type}_{diag_name}'
    #     if func_name in diags.DiagCalc.__dict__:
    #         self.diags[spell] = diags.DiagCalc.__dict__[func_name](
    #             self, load_idx=load_idx,
    #             adjust_month=adjust_month,
    #             ann_method=ann_method, **kws,
    #         )
    #     else:
    #         func_name = f'calc_{plot_type}'
    #         if 'sm_method' in locals(): kws.update({'sm_method': sm_method})

    #         self.diags[spell] = diags.DiagCalc.__dict__[func_name](
    #             self, vn=diag_name, load_idx=load_idx,
    #             adjust_month=adjust_month,
    #             ann_method=ann_method, **kws,
    #         )

    #     utils.p_success(f'>>> case.diags["{spell}"] created')

    # def plot(self, spell, **kws):
    #     ''' Plot a diagnostic spell

    #     Args:
    #         spell (str): The diagnostic variable name in the format of `plot_type:diag_name:ann_method[:sm_method]`, see :func:`x4c.case.Timeseries.calc`
    #     '''
    #     spell_elements = spell.split(':')
    #     if len(spell_elements) == 3:
    #         plot_type, diag_name, ann_method = spell_elements
    #     elif len(spell_elements) == 4:
    #         plot_type, diag_name, ann_method, sm_method = spell_elements
    #     else:
    #         raise ValueError('Wrong diagnostic spell.')

    #     if 'sm_method' in locals(): kws.update({'sm_method': sm_method})
    #     return diags.DiagPlot.__dict__[f'plot_{plot_type}'](self, diag_name, ann_method=ann_method, **kws)

class Logs:
    ''' Initialize a CESM Log case

    Args:
        root_dir (str): the root directory of the CESM Timeseries output
    '''
    def __init__(self, dirpath, comp='ocn', load_num=None):
        self.dirpath = dirpath
        self.paths = sorted(glob.glob(os.path.join(dirpath, f'{comp}.log.*.gz')))
        if load_num is not None:
            if load_num < 0:
                self.paths = self.paths[load_num:]
            else:
                self.paths = self.paths[:load_num]

        utils.p_header(f'>>> Logs.dirpath: {self.dirpath}')
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

        if not isinstance(vn, (list, tuple)):
            vn = [vn]

        nf = len(self.paths)
        df_list = []
        for idx_file in range(nf):
            vars = {}
            with gzip.open(self.paths[idx_file], mode='rt') as fp:
                lines = fp.readlines()

                # find 1st timestamp
                for line in lines:
                    i = lines.index(line)
                    if line.find('This run        started from') != -1 and lines[i+1].find('date(month-day-year):') != -1:
                        start_date = lines[i+1].split(':')[-1].strip()
                        break

                mm, dd, yyyy = start_date.split('-')

                # find variable values
                for line in lines:
                    for v in vn:
                        if v not in vars:
                            vars[v] = []
                        elif line.strip().startswith(f'{v}:'):
                            val = float(line.strip().split(':')[-1])
                            vars[v].append(val)

            df_tmp = pd.DataFrame(vars)
            dates = xr.cftime_range(start=f'{yyyy}-{mm}-{dd}', freq='MS', periods=len(df_tmp), calendar='noleap')
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