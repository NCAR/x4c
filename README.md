# x4c: Xarray for CESM
x4c (xarray4cesm) is an Xarray extension that aims to support efficient and intuitive CESM output analysis and visualization:
- Analysis features: regrid, various of mean calculation, annualization/seasonalization, etc.
- Visualization features: timeseries, horizontal and vertical spatial plots, etc.

> **_Disclaimer:_**  This package is still in its early stage and under active development, and its API could be changed frequently.

## Installation

```bash
# dependencies
conda install -c conda-forge jupyter notebook xesmf

# x4c
pip install git+https://https://github.com/NCAR/x4c.git
```


## License
GPL-2.0