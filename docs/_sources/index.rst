******************************
x4c: Xarray for CESM
******************************

x4c (xarray4cesm) is an Xarray plugin that aims to support efficient and intuitive CESM output analysis and visualization:

+ **Analysis** features: regrid, various of mean calculation, annualization/seasonalization, etc.
+ **Visualization** features: timeseries plots, zonal mean plots, horizontal and vertical 2D spatial plots, etc.

.. warning ::
    This package is still in its early stage and under active development, and its API could be changed frequently.

|

.. grid:: 1 1 2 2
    :gutter: 2

    .. grid-item-card::  Installation of `x4c`
        :class-title: custom-title
        :class-body: custom-body
        :img-top: _static/installation.png
        :link: ug-installation
        :link-type: doc

        Installation instructions.

    .. grid-item-card::  Core Features
        :class-title: custom-title
        :class-body: custom-body
        :img-top: _static/setup.png
        :link: ug-core
        :link-type: doc

        Examples on core features.

    .. grid-item-card::  CESM Diagnostics
        :class-title: custom-title
        :class-body: custom-body
        :img-top: _static/run.png
        :link: ug-diags
        :link-type: doc

        Examples on CESM diagnostics.

    .. grid-item-card::  API
        :class-title: custom-title
        :class-body: custom-body
        :img-top: _static/api.png
        :link: ug-api
        :link-type: doc

        The essential API.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   ug-installation
   ug-core
   ug-diags
   ug-api