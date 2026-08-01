from . import utils
import xarray as xr
import numpy as np

class Registry:
    funcs = {}

    @classmethod
    def get_F(cls, name):
        '''Retrieve a diagnostic function by name.'''
        return cls.funcs.get(name)


def F(func=None, *, name=None):
    '''Decorator to register a diagnostic function, with optional custom key.'''
    def decorator(f):
        if name is not None:
            key = name  # use explicit override, e.g. "NINO3.4"
        else:
            n = f.__name__
            key = n[4:] if n.startswith('get_') else n
        Registry.funcs[key] = f
        return f

    if func is not None:
        # Called as @F (no arguments)
        return decorator(func)
    else:
        # Called as @F(name='...') 
        return decorator


class DiagCalc:
    # Get specific diagnostic variables
    @F
    def get_SST(case, **kws):

        vn = 'TEMP'
        case.load(vn, **kws)
        sst = case.ds[vn].x.da.isel(z_t=0)

        sst.attrs['units'] = '°C'
        sst.attrs['long_name'] = 'Sea Surface Temperature'
        sst.name = 'SST'
        return sst

    @F
    def get_SSS(case, **kws):

        vn = 'SALT'
        case.load(vn, **kws)
        sss = case.ds[vn].x.da.isel(z_t=0)
        sss.attrs['units'] = 'gram/kilogram'
        sss.attrs['long_name'] = 'Sea Surface Salinity'
        sss.name = 'SSS'
        return sss

    @F
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

    @F
    def get_MLD(case, **kws):
        vn = 'XMXL'
        case.load(vn, **kws)
        da = case.ds[vn].x.da / 100
        da.name = 'MLD'
        da.attrs['units'] = 'm'
        return da

    @F
    def get_PRECT(case, **kws):
        case.load('PRECC', **kws)
        case.load('PRECL', **kws)
        da = case.ds['PRECC'].x.da + case.ds['PRECL'].x.da
        # arithmetic between two DataArrays drops the DataArray-valued attrs, so the
        # grid metadata has to be put back or `.x.gm` fails
        utils.copy_grid_attrs(da, case.ds['PRECC'].x.da)
        da.name = 'PRECT'
        da.attrs['long_name'] = 'Total precipitation rate (convective + large-scale; liq + ice)'
        return da

    @F
    def get_dDp(case, **kws):
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

        dDp = (hdo / h2o - 1)*1000
        utils.copy_grid_attrs(dDp, case.ds['PRECRC_H2Or'].x.da)
        dDp.name = 'dDp'
        dDp.attrs['long_name'] = 'Precipitation dD'
        dDp.attrs['units'] = 'permil'

        return dDp
        
    @F
    def get_d18Op(case, **kws):
        case.load('PRECRC_H216Or', **kws)
        case.load('PRECSC_H216Os', **kws)
        case.load('PRECRL_H216OR', **kws)
        case.load('PRECSL_H216OS', **kws)

        case.load('PRECRC_H218Or', **kws)
        case.load('PRECSC_H218Os', **kws)
        case.load('PRECRL_H218OR', **kws)
        case.load('PRECSL_H218OS', **kws)

        precrc_h216or = case.ds['PRECRC_H216Or'].x.da.clip(max=2e-3)
        precsc_h216os = case.ds['PRECSC_H216Os'].x.da.clip(max=1e-3)
        precrl_h216or = case.ds['PRECRL_H216OR'].x.da.clip(max=2e-3)
        precsl_h216os = case.ds['PRECSL_H216OS'].x.da.clip(max=1e-3)

        precrc_h218or = case.ds['PRECRC_H218Or'].x.da.clip(max=1e-5)
        precsc_h218os = case.ds['PRECSC_H218Os'].x.da.clip(max=5e-6)
        precrl_h218or = case.ds['PRECRL_H218OR'].x.da.clip(max=1e-5)
        precsl_h218os = case.ds['PRECSL_H218OS'].x.da.clip(max=5e-6)

        p16O = precrc_h216or + precsc_h216os + precrl_h216or + precsl_h216os
        p18O = precrc_h218or + precsc_h218os + precrl_h218or + precsl_h218os

        p16O = p16O.where(p16O > 1e-18, 1e-18)
        p18O = p18O.where(p18O > 1e-18, 1e-18)

        d18Op = (p18O/p16O - 1)*1e3
        utils.copy_grid_attrs(d18Op, case.ds['PRECRC_H216Or'].x.da)
        d18Op.name = 'd18Op'
        d18Op.attrs['long_name'] = 'Precipitation d18O'
        d18Op.attrs['units'] = 'permil'

        return d18Op

    @F
    def get_d18Osw(case, **kws):
        case.load('R18O', **kws)
        R18O = case.ds['R18O'].x.da
        d18Osw = (R18O - 1)*1e3
        d18Osw.name = 'd18Osw'
        d18Osw.attrs['long_name'] = 'Sea-water d18O'
        d18Osw.attrs['units'] = 'permil'
        return d18Osw

    @F
    def get_dDsw(case, **kws):
        case.load('RHDO', **kws)
        RHDO = case.ds['RHDO'].x.da
        dDsw = (RHDO - 1)*1e3
        dDsw.name = 'dDsw'
        dDsw.attrs['long_name'] = 'Sea-water dD'
        dDsw.attrs['units'] = 'permil'
        return dDsw

    @F
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
        utils.copy_grid_attrs(d18Oc, T)
        d18Oc.name = 'd18Oc'
        d18Oc.attrs['long_name'] = 'Calcite d18O'
        d18Oc.attrs['units'] = 'permil'
        return d18Oc

    @F
    def get_RESTOM(case, **kws):
        ''' Calculate RESTOM = FSNT - FLNT
        '''
        case.load('FSNT', **kws)
        case.load('FLNT', **kws)

        RESTOM = case.ds['FSNT'].x.da - case.ds['FLNT'].x.da
        utils.copy_grid_attrs(RESTOM, case.ds['FSNT'].x.da)
        RESTOM.name = 'RESTOM'
        RESTOM.attrs['long_name'] = 'Net Radiation Flux'
        RESTOM.attrs['units'] = 'W/m$^2$'
        return RESTOM
    
    @F
    def get_DP(case, **kws):
        ''' Calculate pressure thickness
        '''
        vn = 'PS'
        case.load(vn, **kws)
        da_pressure = case.ds[vn]['hyai']*case.ds[vn]['P0'] + case.ds[vn]['hybi']*case.ds[vn][vn]
        da = da_pressure.diff('ilev').rename({'ilev': 'lev'})
        da['lev'] = case.ds[vn]['lev']
        utils.copy_grid_attrs(da, case.ds[vn].x.da)
        da.name = 'DP'
        da.attrs['long_name'] = 'Pressure Thickness'
        da.attrs['units'] = 'Pa'
        return da

    @F(name='NINO3.4')
    def get_NINO34(case, **kws):
        ''' Calculate NINO3.4
        '''
        vn = 'SST'
        case.load(vn, **kws)
        da_sst = case.ds[vn]
        if 'lat' not in da_sst.coords or 'lon' not in da_sst.coords:
            da_sst = da_sst.x.regrid()

        da = da_sst.x.geo_mean(ind='nino3.4')
        da.name = 'NINO3.4'
        da.attrs['long_name'] = 'NINO3.4 Index'
        # `SST` is already in °C (see `get_SST`), so this must not be labelled 'K':
        # `Timeseries.calc` converts anything tagged 'K' by subtracting 273.15, which
        # would silently offset the index by that amount.
        da.attrs['units'] = '°C'
        return da


    @F
    def get_MOC(case, **kws):
        vn = 'MOC'
        kws.update({'verbose': False, 'reload': True})
        case.load(vn, vtype='raw', **kws)  # due to the same variable name in POP
        da = case.ds[vn].x.da.isel(transport_reg=0, moc_comp=0)
        da['moc_z'] = da['moc_z'] / 1e5  # unit: cm -> km
        da['moc_z'].attrs['units'] = 'km'
        da = da.rename({'moc_z': 'z_t', 'lat_aux_grid': 'lat'})
        da.name = 'MOC'
        da.attrs['lon_name'] = 'Meridional Ocean Circulation'
        return da


    @F
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

