from . import utils
import xarray as xr
import numpy as np

class DiagCalc:
    # General calculations
    # def calc_ts(case, vn, load_idx=-1, adjust_month=True, sm_method='gm', ann_method='ann', long_name=None, units=None):
    #     ''' General timeseries calculation
    #     '''
    #     case.load(vn, load_idx=load_idx, adjust_month=adjust_month)
    #     if long_name is None:
    #         if 'long_name' in case.ds[vn].x.da.attrs:
    #             long_name = case.ds[vn].x.da.attrs['long_name']
    #         else:
    #             long_name = vn

    #     da_tmp = case.ds[vn].x.da
    #     da_ann = utils.ann_modifier(da_tmp, ann_method, long_name=long_name)
    #     da = getattr(da_ann.x, sm_method)
    #     da = utils.convert_units(da, units)
    #     return da

    # def calc_map(case, vn, load_idx=-1, adjust_month=True, ann_method='ann', clim=True, long_name=None, units=None):
    #     ''' General map calculation
    #     '''
    #     case.load(vn, load_idx=load_idx, adjust_month=adjust_month, regrid=True)
    #     if long_name is None:
    #         if 'long_name' in case.ds[vn].x.da.attrs:
    #             long_name = case.ds[vn].x.da.attrs['long_name']
    #         else:
    #             long_name = vn

    #     da_tmp = case.ds[vn].x.da
    #     da = utils.ann_modifier(da_tmp, ann_method, long_name=long_name)
    #     if clim: da = da.mean('time')
    #     da = utils.convert_units(da, units)
    #     return da

    # # Specific calculations
    # def calc_ts_GMST(case, load_idx=-1, adjust_month=True, ann_method='ann'):
    #     vn = 'TS'
    #     case.load(vn, load_idx=load_idx, adjust_month=adjust_month)
    #     da_degC = case.ds[vn].x.da - 273.15
    #     da_degC.attrs['units'] = '°C'
    #     da_ann = utils.ann_modifier(da_degC, ann_method, long_name='Surface Temperature')
    #     da = da_ann.x.gm
    #     return da

    # def calc_ts_d18Osw(case, load_idx=-1, adjust_month=True, ann_method='ann'):
    #     vn = 'R18O'
    #     case.load(vn, load_idx=load_idx, adjust_month=adjust_month, regrid=True)
    #     R18O = case.ds[vn].x.da.isel(z_t=0)
    #     d18O = (R18O - 1)*1e3
    #     da_ann = utils.ann_modifier(d18O, ann_method, long_name=r'Sea Surface $\delta^{18}$O')
    #     da = da_ann.x.gm
    #     da.name = 'd18Osw'
    #     da.attrs['units'] = 'permil'
    #     return da

    # def calc_map_TS(case, load_idx=-1, adjust_month=True, ann_method='ann', clim=True):
    #     vn = 'TS'
    #     case.load(vn, load_idx=load_idx, adjust_month=adjust_month, regrid=True)
    #     da_degC = case.ds[vn].x.da - 273.15
    #     da_degC.attrs['units'] = '°C'
    #     da = utils.ann_modifier(da_degC, ann_method, long_name='Surface Temperature')
    #     if clim: da = da.mean('time')
    #     return da

    # def calc_map_LST(case, load_idx=-1, adjust_month=True, ann_method='ann', clim=True):
    #     vn = 'TS'
    #     case.load(vn, load_idx=load_idx, adjust_month=adjust_month, regrid=True)
    #     da_degC = case.ds[vn].x.da - 273.15
    #     da_degC.attrs['units'] = '°C'
    #     da_ann = utils.ann_modifier(da_degC, ann_method, long_name='Land Surface Temperature')

    #     case.load('LANDFRAC', load_idx=load_idx, adjust_month=adjust_month, regrid=True)
    #     landfrac = case.ds[vn].x.da.x.annualize().mean('time')

    #     da = da_ann.where(landfrac>0.5)
    #     if clim: da = da.mean('time')
    #     da.name = 'LST'
    #     return da

    # def calc_map_SST(case, load_idx=-1, adjust_month=True, ann_method='ann', clim=True):
    #     vn = 'TEMP'
    #     case.load(vn, load_idx=load_idx, adjust_month=adjust_month, regrid=True)
    #     case.ds[vn].attrs['units'] = '°C'
    #     sst = case.ds[vn].x.da.isel(z_t=0)
    #     da = utils.ann_modifier(sst, ann_method, long_name='Sea Surface Temperature')
    #     if clim: da = da.mean('time')
    #     da.name = 'SST'
    #     return da

    # def calc_map_d18Osw(case, load_idx=-1, adjust_month=True, ann_method='ann', clim=True):
    #     vn = 'R18O'
    #     case.load(vn, load_idx=load_idx, adjust_month=adjust_month, regrid=True)
    #     R18O = case.ds[vn].x.da.isel(z_t=0)
    #     d18O = (R18O - 1)*1e3
    #     da = utils.ann_modifier(d18O, ann_method, long_name=r'Sea Surface $\delta^{18}$O')
    #     if clim: da = da.mean('time')
    #     da.name = 'd18Osw'
    #     da.attrs['units'] = 'permil'
    #     return da

    # def calc_map_d18Op(case, load_idx=-1, adjust_month=True, ann_method='ann', clim=True):
    #     case.load('PRECRC_H216Or', load_idx=load_idx, adjust_month=adjust_month)
    #     case.load('PRECSC_H216Os', load_idx=load_idx, adjust_month=adjust_month)
    #     case.load('PRECRL_H216OR', load_idx=load_idx, adjust_month=adjust_month)
    #     case.load('PRECSL_H216OS', load_idx=load_idx, adjust_month=adjust_month)

    #     case.load('PRECRC_H218Or', load_idx=load_idx, adjust_month=adjust_month)
    #     case.load('PRECSC_H218Os', load_idx=load_idx, adjust_month=adjust_month)
    #     case.load('PRECRL_H218OR', load_idx=load_idx, adjust_month=adjust_month)
    #     case.load('PRECSL_H218OS', load_idx=load_idx, adjust_month=adjust_month)

    #     p16O = case.ds['PRECRC_H216Or'].x.da + case.ds['PRECSC_H216Os'].x.da + case.ds['PRECRL_H216OR'].x.da + case.ds['PRECSL_H216OS'].x.da
    #     p18O = case.ds['PRECRC_H218Or'].x.da + case.ds['PRECSC_H218Os'].x.da + case.ds['PRECRL_H218OR'].x.da + case.ds['PRECSL_H218OS'].x.da

    #     p16O = p16O.where(p16O > 1e-18, 1e-18)
    #     p18O = p18O.where(p18O > 1e-18, 1e-18)

    #     d18Op = (p18O / p16O - 1)*1000
    #     d18Op.name = 'd18Op'
    #     d18Op = d18Op.x.regrid()
    #     da = utils.ann_modifier(d18Op, ann_method, long_name=r'Precipitation $\delta^{18}$O')
    #     da.attrs['units'] = 'permil'
    #     if clim: da = da.mean('time')
    #     return da

    # def calc_map_d18Op_clm(case, load_idx=-1, adjust_month=True, ann_method='ann', clim=True):
    #     case.load('RAIN_H218O', load_idx=load_idx, adjust_month=adjust_month)
    #     case.load('RAIN_H2OTR', load_idx=load_idx, adjust_month=adjust_month)

    #     p16O = case.ds['RAIN_H2OTR'].x.da
    #     p18O = case.ds['RAIN_H218O'].x.da

    #     p16O = p16O.where(p16O > 1e-18, 1e-18)
    #     p18O = p18O.where(p18O > 1e-18, 1e-18)

    #     d18Op = (p18O / p16O - 1)*1000
    #     d18Op.name = 'd18Op'
    #     d18Op = d18Op.x.regrid()
    #     da = utils.ann_modifier(d18Op, ann_method, long_name=r'Precipitation $\delta^{18}$O in CLM')
    #     da.attrs['units'] = 'permil'
    #     if clim: da = da.mean('time')
    #     return da

    # def calc_map_d18Os_clm(case, load_idx=-1, lev_idx=0, adjust_month=True, ann_method='ann', clim=True):
    #     case.load('H2OSOI_H2OTR', load_idx=load_idx, adjust_month=adjust_month)
    #     case.load('H2OSOI_H218O', load_idx=load_idx, adjust_month=adjust_month)

    #     p16O = case.ds['H2OSOI_H2OTR'].x.da
    #     p18O = case.ds['H2OSOI_H218O'].x.da

    #     p16O = p16O.where(p16O > 1e-18, 1e-18)
    #     p18O = p18O.where(p18O > 1e-18, 1e-18)

    #     d18Os = (p18O / p16O - 1) * 1000
    #     d18Os.name = 'd18Os'
    #     d18Os = d18Os.x.regrid()
    #     da = utils.ann_modifier(d18Os, ann_method, long_name=r'Soil $\delta^{18}$O in CLM')
    #     da.attrs['units'] = 'permil'
    #     if clim: da = da.mean('time')
    #     return da[lev_idx]

    # def calc_map_MLD(case, load_idx=-1, adjust_month=True, ann_method='ann', clim=True):
    #     vn = 'XMXL'
    #     case.load(vn, load_idx=load_idx, adjust_month=adjust_month, regrid=True)
    #     da_tmp = case.ds[vn].x.da / 100
    #     da = utils.ann_modifier(da_tmp, ann_method, long_name='Mixed Layer Depth')
    #     if clim: da = da.mean('time')
    #     da.name = 'MLD'
    #     da.attrs['units'] = 'm'
    #     return da

    # def calc_3d_PD(case, load_idx=-1, adjust_month=True, ann_method='ann'):
    #     vn = 'PD'
    #     case.load(vn, load_idx=load_idx, adjust_month=adjust_month, regrid=True)
    #     da_tmp = case.ds[vn].x.da
    #     da_ann = utils.ann_modifier(da_tmp, ann_method, long_name='Potential Density')
    #     da = da_ann.mean('time')
    #     da.name = 'PD'
    #     return da

    # def calc_yz_PD(case, load_idx=-1, adjust_month=True, ann_method='ann'):
    #     da_tmp = DiagCalc.calc_3d_PD(case, load_idx=load_idx, adjust_month=adjust_month, ann_method=ann_method)
    #     da = da_tmp.x.zm
    #     da['z_t'] = da_tmp['z_t'] / 1e5  # unit: cm -> km
    #     da['z_t'].attrs['units'] = 'km'
    #     da.name = 'PD'
    #     return da

    # def calc_zm_LST(case, load_idx=-1, adjust_month=True, ann_method='ann'):
    #     da_tmp = DiagCalc.calc_map_LST(case, load_idx=load_idx, adjust_month=adjust_month, ann_method=ann_method)
    #     da = da_tmp.x.zm
    #     da.attrs['long_name'] = f'Zonal Mean {da.attrs["long_name"]}'
    #     da.name = 'LST'
    #     return da

    # def calc_zm_SST(case, load_idx=-1, adjust_month=True, ann_method='ann'):
    #     da_tmp = DiagCalc.calc_map_SST(case, load_idx=load_idx, adjust_month=adjust_month, ann_method=ann_method)
    #     da = da_tmp.x.zm
    #     da.attrs['long_name'] = f'Zonal Mean {da.attrs["long_name"]}'
    #     da.name = 'SST'
    #     return da

    # def calc_3d_MOC(case, load_idx=-1, adjust_month=True, ann_method='ann'):
    #     vn = 'MOC'
    #     case.load(vn, load_idx=load_idx, adjust_month=adjust_month, regrid=False)
    #     da_tmp = case.ds[vn]
    #     da_ann = utils.ann_modifier(da_tmp, ann_method, long_name='Meridional Ocean Circulation')
        
    #     da = da_ann.copy()
    #     da['moc_z'] = da_ann['moc_z'] / 1e5  # unit: cm -> km
    #     da['moc_z'].attrs['units'] = 'km'
    #     return da

    # def calc_ts_MOC(case, load_idx=-1, adjust_month=True, ann_method='ann', transport_reg=0, moc_z=slice(0.5, None), lat_aux_grid=slice(-90, -28)):
    #     da_tmp = DiagCalc.calc_3d_MOC(case, load_idx=load_idx, adjust_month=adjust_month, ann_method=ann_method)
    #     da = da_tmp.isel(transport_reg=transport_reg, moc_comp=0).sel(moc_z=moc_z, lat_aux_grid=lat_aux_grid).min(('moc_z', 'lat_aux_grid'))
    #     return da

    # def calc_yz_MOC(case, load_idx=-1, adjust_month=True, ann_method='ann', transport_reg=0):
    #     da_tmp = DiagCalc.calc_3d_MOC(case, load_idx=load_idx, adjust_month=adjust_month, ann_method=ann_method)
    #     da = da_tmp.isel(transport_reg=transport_reg, moc_comp=0).mean('time')
    #     return da

    # def calc_ysig2_MOC(case, load_idx=-1, adjust_month=True, ann_method='ann', transport_reg=0,
    #                    refz=2000, sigma_mid=None, sigma_edge=None):
    #     '''  Compute MOC with the isopycnal (constant density surfaces) sigma-2 vertical coordinate

    #     Reference: https://github.com/sgyeager/POP_MOC/blob/main/notebooks/pop_MOCsig2_1deg.ipynb
    #     '''
    #     case.load('SALT', load_idx=load_idx, adjust_month=adjust_month, regrid=False)
    #     case.load('TEMP', load_idx=load_idx, adjust_month=adjust_month, regrid=False)
    #     case.load('KMT')
    #     sigma2_T = pop_tools.eos(salt=case.ds['SALT'], temp=case.ds['TEMP'], depth=xr.DataArray(refz)) - 1000
    #     sigma2_T = sigma2_T.assign_attrs({'long_name':'Sigma referenced to {}m'.format(refz),'units':'kg/m^3'})

    #     if sigma_mid is None:
    #         sigma_mid = np.array([
    #             28.  , 28.2 , 28.4 , 28.6 , 28.8 , 29.  , 29.2 , 29.4 , 29.6 ,
    #             29.8 , 30.  , 30.2 , 30.4 , 30.6 , 30.8 , 31.  , 31.2 , 31.4 ,
    #             31.6 , 31.8 , 32.  , 32.2 , 32.4 , 32.6 , 32.8 , 33.  , 33.2 ,
    #             33.4 , 33.6 , 33.8 , 34.  , 34.2 , 34.4 , 34.6 , 34.8 , 35.  ,
    #             35.1 , 35.2 , 35.3 , 35.4 , 35.5 , 35.6 , 35.7 , 35.8 , 35.9 ,
    #             36.  , 36.05, 36.1 , 36.15, 36.2 , 36.25, 36.3 , 36.35, 36.4 ,
    #             36.45, 36.5 , 36.55, 36.6 , 36.65, 36.7 , 36.75, 36.8 , 36.85,
    #             36.9 , 36.95, 37.  , 37.05, 37.1 , 37.15, 37.2 , 37.25, 37.3 ,
    #             37.35, 37.4 , 37.45, 37.5 , 37.55, 37.6 , 37.65, 37.7 , 37.75,
    #             37.8 , 37.85, 37.9 , 37.95, 38.,
    #         ])

    #     if sigma_edge is None:
    #         sigma_edge = np.array([
    #             0.   , 28.1  , 28.3  , 28.5  , 28.7  , 28.9  , 29.1  , 29.3  ,
    #             29.5  , 29.7  , 29.9  , 30.1  , 30.3  , 30.5  , 30.7  , 30.9  ,
    #             31.1  , 31.3  , 31.5  , 31.7  , 31.9  , 32.1  , 32.3  , 32.5  ,
    #             32.7  , 32.9  , 33.1  , 33.3  , 33.5  , 33.7  , 33.9  , 34.1  ,
    #             34.3  , 34.5  , 34.7  , 34.9  , 35.05 , 35.15 , 35.25 , 35.35 ,
    #             35.45 , 35.55 , 35.65 , 35.75 , 35.85 , 35.95 , 36.025, 36.075,
    #             36.125, 36.175, 36.225, 36.275, 36.325, 36.375, 36.425, 36.475,
    #             36.525, 36.575, 36.625, 36.675, 36.725, 36.775, 36.825, 36.875,
    #             36.925, 36.975, 37.025, 37.075, 37.125, 37.175, 37.225, 37.275,
    #             37.325, 37.375, 37.425, 37.475, 37.525, 37.575, 37.625, 37.675,
    #             37.725, 37.775, 37.825, 37.875, 37.925, 37.975, 50.,
    #         ])

    #     # Here, test histogram by counting cells in each density bin. Vertical sum should be same as KMT.
    #     iso_count = histogram(sigma2_T, bins=[sigma_edge.values],dim=['z_t'],density=False)
    #     iso_count = iso_count.rename({'density_bin':'sigma'}).assign_coords({'sigma':sigma_mid})

    #     kmtdiff = iso_count.sum('sigma') - case.ds['KMT']

    #     # Use histogram to compute layer thickness. Vertical sum should be same as HT.
    #     dzwgts = (ds['dz']/100.).assign_attrs({'units':'m'})
    #     iso_thick = histogram(sigma2_T, bins=[sigma_edge.values], weights=dzwgts,dim=['z_t'],density=False)
    #     iso_thick = iso_thick.rename({'density_bin':'sigma'}).assign_coords({'sigma':sigma_mid})
    #     iso_thick = iso_thick.rename('iso_thick').assign_attrs({'units':'m','long_name':'Isopycnal Layer Thickness'}).rename({'sigma':'sigma_mid'})
    #     iso_thick = iso_thick.transpose('time','sigma_mid','nlat','nlon')

    #     htdiff = iso_thick.sum('sigma_mid') - (ds['HT']/100.).assign_attrs({'units':'m'})


    #     return da

    # Get specific diagnostic variables
    def get_SST(case, **kws):
        if ('SST', 'ocn') not in case.vars_info:
            vn = 'TEMP'
            case.load(vn, **kws)
            sst = case.ds[vn].x.da.isel(z_t=0)
        else:
            case.load('SST', **kws)
            sst = case.ds['SST'].x.da

        sst.attrs['units'] = '°C'
        sst.attrs['long_name'] = 'Sea Surface Temperature'
        sst.name = 'SST'
        return sst

    def get_SSS(case, **kws):
        if ('SSS', 'ocn') not in case.vars_info:
            vn = 'SALT'
            case.load(vn, **kws)
            sss = case.ds[vn].x.da.isel(z_t=0)
        else:
            case.load('SSS', **kws)
            sss = case.ds['SSS'].x.da

        sss.attrs['units'] = 'gram/kilogram'
        sss.attrs['long_name'] = 'Sea Surface Salinity'
        sss.name = 'SSS'
        return sss

    def get_LST(case, **kws):
        vn = 'TS'
        case.load(vn, **kws)
        ts = case.ds[vn].x.da

        vn = 'LANDFRAC'
        case.load(vn, **kws)
        landfrac = case.ds[vn].x.da

        lst = ts.where(landfrac>0.5)

        lst.attrs['long_name'] = 'Land Surface Temperature'
        lst.name = 'LST'
        return lst

    def get_MLD(case, **kws):
        vn = 'XMXL'
        case.load(vn, **kws)
        da = case.ds[vn].x.da / 100
        da.name = 'MLD'
        da.attrs['units'] = 'm'
        return da

    def get_PRECT(case, **kws):
        case.load('PRECC', **kws)
        case.load('PRECL', **kws)
        da = case.ds['PRECC'].x.da + case.ds['PRECL'].x.da
        da.name = 'PRECT'
        da.attrs['long_name'] = 'Total precipitation rate (convective + large-scale; liq + ice)'
        return da

    def get_dD(case, **kws):
        case.load('PRECRC_H2Or', **kws)
        case.load('PRECSC_H2Os', **kws)
        case.load('PRECRL_H2OR', **kws)
        case.load('PRECSL_H2OS', **kws)

        case.load('PRECRC_HDOr', **kws)
        case.load('PRECSC_HDOs', **kws)
        case.load('PRECRL_HDOR', **kws)
        case.load('PRECSL_HDOS', **kws)

        h2o = case.ds['PRECRC_H2Or'].x.da + case.ds['PRECSC_H2Os'].x.da + case.ds['PRECRL_H2OR'].x.da + case.ds['PRECSL_H2OS'].x.da
        hdo = case.ds['PRECRC_HDOr'].x.da + case.ds['PRECSC_HDOs'].x.da + case.ds['PRECRL_HDOR'].x.da + case.ds['PRECSL_HDOS'].x.da

        h2o = h2o.where(h2o > 1e-18, 1e-18)
        hdo = hdo.where(hdo > 1e-18, 1e-18)

        dD = (hdo / h2o - 1)*1000
        dD.name = 'dD'
        dD.attrs['long_name'] = 'Precipitation dD'
        dD.attrs['units'] = 'permil'

        return dD
        
    def get_d18Op(case, **kws):
        case.load('PRECRC_H216Or', **kws)
        case.load('PRECSC_H216Os', **kws)
        case.load('PRECRL_H216OR', **kws)
        case.load('PRECSL_H216OS', **kws)

        case.load('PRECRC_H218Or', **kws)
        case.load('PRECSC_H218Os', **kws)
        case.load('PRECRL_H218OR', **kws)
        case.load('PRECSL_H218OS', **kws)

        p16O = case.ds['PRECRC_H216Or'].x.da + case.ds['PRECSC_H216Os'].x.da + case.ds['PRECRL_H216OR'].x.da + case.ds['PRECSL_H216OS'].x.da
        p18O = case.ds['PRECRC_H218Or'].x.da + case.ds['PRECSC_H218Os'].x.da + case.ds['PRECRL_H218OR'].x.da + case.ds['PRECSL_H218OS'].x.da

        p16O = p16O.where(p16O > 1e-18, 1e-18)
        p18O = p18O.where(p18O > 1e-18, 1e-18)

        d18Op = (p18O / p16O - 1)*1000
        d18Op.name = 'd18Op'
        d18Op.attrs['long_name'] = 'Precipitation d18O'
        d18Op.attrs['units'] = 'permil'

        return d18Op

    def get_d18Osw(case, **kws):
        case.load('R18O', **kws)
        R18O = case.ds['R18O'].x.da
        d18Osw = (R18O - 1)*1e3
        d18Osw.name = 'd18Osw'
        d18Osw.attrs['long_name'] = 'Sea-water d18O'
        d18Osw.attrs['units'] = 'permil'
        return d18Osw

    def get_d18Oc(case, **kws):
        ''' Calculate d18Oc = f(TEMP, d18Osw)

        Reference: Marchitto et al. (2014)
        '''
        case.load('R18O', **kws)
        case.load('TEMP', **kws)
        R18O = case.ds['R18O'].x.da
        d18Osw = (R18O - 1)*1e3
        T = case.ds['TEMP'].x.da

        d18Osw_PDB = d18Osw - 0.27         #VSMOW to VPDB conversion
        d18Oc = (-0.245*T + 0.0011*T*T + 3.58) + d18Osw_PDB
        d18Oc.name = 'd18Oc'
        d18Oc.attrs['long_name'] = 'Calcite d18O'
        d18Oc.attrs['units'] = 'permil'
        return d18Oc


    def get_RESTOM(case, **kws):
        ''' Calculate RESTOM = FSNT - FLNT
        '''
        case.load('FSNT', **kws)
        case.load('FLNT', **kws)

        RESTOM = case.ds['FSNT'].x.da - case.ds['FLNT'].x.da
        RESTOM.name = 'RESTOM'
        RESTOM.attrs['long_name'] = 'Net Radiation Flux'
        RESTOM.attrs['units'] = 'W/m$^2$'
        return RESTOM
    



    # def get_d18Oc(case, **kws):
    #     ''' Calculate d18Oc = f(TEMP, d18Osw) based on the Eq (1) of the Ref.:
    #     T = 16.1 - 4.64 (d18Oc - d18Osw) + 0.09 (d18Oc - d18Osw)^2

    #     Reference: Hollis, C.J., Dunkley Jones, T., Anagnostou, E., Bijl, P.K., Cramwinckel, M.J., Cui, Y., Dickens, G.R., Edgar, K.M., Eley, Y., Evans, D., Foster, G.L., Frieling, J., Inglis, G.N., Kennedy, E.M., Kozdon, R., Lauretano, V., Lear, C.H., Littler, K., Lourens, L., Meckler, A.N., Naafs, B.D.A., Pälike, H., Pancost, R.D., Pearson, P.N., Röhl, U., Royer, D.L., Salzmann, U., Schubert, B.A., Seebeck, H., Sluijs, A., Speijer, R.P., Stassen, P., Tierney, J., Tripati, A., Wade, B., Westerhold, T., Witkowski, C., Zachos, J.C., Zhang, Y.G., Huber, M., Lunt, D.J., 2019. The DeepMIP contribution to PMIP4: methodologies for selection, compilation and analysis of latest Paleocene and early Eocene climate proxy data, incorporating version 0.1 of the DeepMIP database. Geoscientific Model Development 12, 3149–3206. https://doi.org/10.5194/gmd-12-3149-2019
    #     '''
    #     case.load('R18O', **kws)
    #     case.load('TEMP', **kws)
    #     R18O = case.ds['R18O'].x.da
    #     d18Osw = (R18O - 1)*1e3
    #     T = case.ds['TEMP'].x.da
    #     D = 4.64**2 - 4*0.09*(16.1 - T)

    #     ds = xr.Dataset()

    #     d18Oc = d18Osw + (4.64 + np.sqrt(D))/0.18
    #     d18Oc.name = 'd18Oc'
    #     d18Oc.attrs['long_name'] = 'Calcite d18O (solution 1)'
    #     d18Oc.attrs['units'] = 'permil'
    #     ds['d18Oc_s1'] = d18Oc

    #     d18Oc = d18Osw + (4.64 - np.sqrt(D))/0.18
    #     d18Oc.name = 'd18Oc'
    #     d18Oc.attrs['long_name'] = 'Calcite d18O (solution 2)'
    #     d18Oc.attrs['units'] = 'permil'
    #     ds['d18Oc_s2'] = d18Oc

    #     utils.p_warning('>>> There are two solutions: "d18Oc_s1" and "d18Oc_s2".')
    #     return ds

    def get_MOC(case, **kws):
        vn = 'MOC'
        case.load(vn, **kws)
        da = case.ds[vn].x.da.isel(transport_reg=0, moc_comp=0)
        da['moc_z'] = da['moc_z'] / 1e5  # unit: cm -> km
        da['moc_z'].attrs['units'] = 'km'
        da = da.rename({'moc_z': 'z_t', 'lat_aux_grid': 'lat'})
        da.name = 'MOC'
        da.attrs['lon_name'] = 'Meridional Ocean Circulation'
        return da

    # def get_SOMOC(case, **kws):
    #     vn = 'MOC'
    #     case.load(vn, **kws)
    #     da = case.ds[vn].x.da.isel(transport_reg=0, moc_comp=0)
    #     da['moc_z'] = da['moc_z'] / 1e5  # unit: cm -> km
    #     da['moc_z'].attrs['units'] = 'km'
    #     da = da.sel(moc_z=slice(0.5, None), lat_aux_grid=slice(-90, -28)).min(('moc_z', 'lat_aux_grid'))
    #     da.name = 'MOC'
    #     da.attrs['lon_name'] = 'Southern Ocean (90°S-28°S) MOC'
    #     return da

    def get_ICEFRAC(case, **kws):
        vn = 'aice'
        case.load(vn, **kws)
        convert_factor = 4*np.pi*6.37122**2 / case.ds[vn].gw.sum().values / 100  # 1e6 km^2
        da = case.ds[vn].x.da * convert_factor
        da.attrs['units'] = '10$^6$ km$^2$'
        da.attrs['long_name'] = 'Sea Ice Area'
        return da


class DiagPlot:
    kws_ts = {}
    kws_map = {}
    kws_zm = {}
    kws_yz = {}

    # ==========
    #  kws_ts
    # ----------
    kws_ts['GMST'] = {'ylim': [20, 30]}

    # ==========
    #  kws_map
    # ----------
    kws_map['TS'] = {'levels': np.linspace(0, 40, 21), 'cbar_kwargs': {'ticks': np.linspace(0, 40, 11)}}
    kws_map['LST'] = {'levels': np.linspace(0, 40, 21), 'cbar_kwargs': {'ticks': np.linspace(0, 40, 11)}}
    kws_map['SST'] = {'levels': np.linspace(0, 40, 21), 'cbar_kwargs': {'ticks': np.linspace(0, 40, 11)}}

    kws_map['MLD'] = {
        # 'levels': np.linspace(0, 800, 17),
        # 'cbar_kwargs': {'ticks': np.linspace(0, 800, 9)},
        'levels': np.linspace(0, 500, 11),
        'cbar_kwargs': {'ticks': np.linspace(0, 500, 6)},
        'extend': 'max',
        # 'central_longitude': -30,
        'central_longitude': 180,
        'cyclic': True,
        # 'log': True,
        # 'levels': np.logspace(0, 3, 28),
        # 'cbar_kwargs': {'ticks': np.logspace(0, 3, 4)},
        # 'vmin': 1,
        # 'vmax': 1000,
        # 'cmap': 'GnBu',
    }

    kws_map['d18Osw'] = {
        'levels': np.linspace(-1, 1, 21),
        'cbar_kwargs': {'ticks': np.linspace(-1, 1, 11)},
    }

    kws_map['d18Op'] = {
        'levels': np.linspace(-20, 0, 21),
        'cbar_kwargs': {'ticks': np.linspace(-20, 0, 11)},
    }
    kws_map['d18Op_clm'] = kws_map['d18Op']
    kws_map['d18Os_clm'] = kws_map['d18Op']

    # ==========
    #  kws_zm
    # ----------
    kws_zm['LST'] = {'ylim': (-35, 40)}
    kws_zm['SST'] = {'ylim': (-5, 40)}

    # ==========
    #  kws_yz
    # ----------
    kws_yz['MOC'] = {'levels': np.linspace(-20, 20, 21), 'cbar_kwargs': {'ticks': np.linspace(-20, 20, 5)}}
    kws_yz['PD'] = {'levels': 20}


    # base function for timeseries (ts) plotting
    # def plot_ts(case, diag_name, ann_method='ann', **kws):
    #     _kws = DiagPlot.kws_ts[diag_name].copy() if diag_name in DiagPlot.kws_ts else {}
    #     _kws = utils.update_dict(_kws, kws)
    #     spell = f'ts:{diag_name}:{ann_method}'
    #     if 'sm_method' in _kws:
    #         sm_method = _kws.pop('sm_method')
    #         spell = f'ts:{diag_name}:{ann_method}:{sm_method}'
    #     fig_ax =  case.diags[spell].x.plot(**_kws)
    #     return fig_ax

    # base function for map plotting
    # def plot_map(case, diag_name, ann_method='ann', cyclic=False, t_idx=-1, clim=True, **kws):
    #     ''' Base function for map plotting

    #     Args:
    #         case (x4c.Timeseries): a CESM timeseries case object
    #         diag_name (str): a diagnostics name
    #         ann_method (str): a annualization method that supports:
                
    #                 * `ann`: calendar year annual mean
    #                 * `<m>`: a number in [..., -11, -12, 1, 2, ..., 12] representing a month
    #                 * `<m1>,<m2>,...`: a list of months sepearted by commmas
    #         cyclic (bool): if True, will add cyclic points to the data array to avoid a blank line in contourf plots
    #     '''
    #     _kws = DiagPlot.kws_map[diag_name].copy() if diag_name in DiagPlot.kws_map else {}
    #     _kws = utils.update_dict(_kws, kws)
    #     da = case.diags[f'map:{diag_name}:{ann_method}']
    #     if 'time' in da.coords and clim:
    #         da = da.mean('time')
    #     elif 'time' in da.coords and not clim:
    #         da = da.isel(time=t_idx)
            
    #     if 'cyclic' in _kws: cyclic = _kws.pop('cyclic')

    #     if cyclic:
    #         da_original = da.copy()
    #         da = utils.add_cyclic_point(da_original)
    #         da.name = da_original.name
    #         da.attrs = da_original.attrs

    #     if 'SSH' in case.vars_info:
    #         case.load('SSH', regrid=True)
    #         da_ssv = case.ds['SSH'].x.da.mean('time')
    #         if cyclic: da_ssv = utils.add_cyclic_point(da_ssv)
    #         fig_ax =  da.x.plot(ssv=da_ssv, **_kws)
    #     else:
    #         fig_ax =  da.x.plot(**_kws)

    #     return fig_ax

    # base function for vertical slice (yz) plotting
    # def plot_yz(case, diag_name, ann_method='ann', **kws):
    #     _kws = DiagPlot.kws_yz[diag_name].copy() if diag_name in DiagPlot.kws_yz else {}
    #     _kws = utils.update_dict(_kws, kws)

    #     fig_ax =  case.diags[f'yz:{diag_name}:{ann_method}'].x.plot(**_kws)
    #     ax = fig_ax[-1] if isinstance(fig_ax, tuple) else fig_ax

    #     ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    #     ax.set_xticklabels(['90°S', '60°S', '30°S', 'EQ', '30°N', '60°N', '90°N'])
    #     ax.set_xlim([-90, 90])
    #     ax.set_xlabel('Latitude')

    #     ax.invert_yaxis()
    #     ax.set_yticks([0, 2, 4])
    #     ax.set_ylabel('Depth [km]')
    #     return fig_ax

    # base function for zonal mean (zm) plotting
    # def plot_zm(case, diag_name, ann_method='ann', **kws):
    #     _kws = DiagPlot.kws_zm[diag_name].copy() if diag_name in DiagPlot.kws_zm else {}
    #     _kws = utils.update_dict(_kws, kws)

    #     fig_ax = case.diags[f'zm:{diag_name}:{ann_method}'].x.plot(**_kws)
    #     ax = fig_ax[-1] if isinstance(fig_ax, tuple) else fig_ax

    #     ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    #     ax.set_xticklabels(['90°S', '60°S', '30°S', 'EQ', '30°N', '60°N', '90°N'])
    #     ax.set_xlim([-90, 90])
    #     ax.set_xlabel('Latitude')

    #     return fig_ax