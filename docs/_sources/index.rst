******************************
x4c: Xarray for CESM
******************************

x4c (xarray4cesm) is an Xarray extension that aims to support efficient and intuitive CESM output postprocessing, analysis, and visualization:

+ **Postprocessing** features: time series generation, seasonal cycle climatology generation, etc.
+ **Analysis** features: regrid, various of mean calculation, annualization/seasonalization, etc.
+ **Visualization** features: timeseries plots, zonal mean plots, horizontal and vertical 2D spatial plots, etc.

.. warning ::
    This package is still in its early stage and under active development, and its API could be changed frequently.

|

.. grid:: 1 1 2 2
    :gutter: 2

    .. grid-item-card::  Installation
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

        Examples on core features; `x4c` as an Xarray extension.

    .. grid-item-card::  CESM Postprocessing
        :class-title: custom-title
        :class-body: custom-body
        :img-top: _static/postprocessing.png
        :link: ug-post
        :link-type: doc

        Examples on CESM postprocessing, including time series and climatology generation.

    .. grid-item-card::  CESM Diagnostics
        :class-title: custom-title
        :class-body: custom-body
        :img-top: _static/diags.png
        :link: ug-diags
        :link-type: doc

        Examples on CESM diagnostics using the `Timeseries` case system and the `spell` magics.

    .. grid-item-card::  Paleoclimate Applications
        :class-title: custom-title
        :class-body: custom-body
        :img-top: _static/paleo.png
        :link: ug-paleo
        :link-type: doc

        Examples on paleoclimate applications.

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