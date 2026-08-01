---
title: "x4c: Xarray for CESM"
---

`x4c` is an Xarray extension that aims to support efficient and intuitive CESM
output postprocessing, analysis, and visualization:

+ **Postprocessing** features: time series generation, seasonal cycle climatology generation, etc.
+ **Analysis** features: regrid, various of mean calculation, annualization/seasonalization, etc.
+ **Visualization** features: timeseries plots, zonal mean plots, horizontal and vertical 2D spatial plots, etc.

:::{warning}
This package is still in its early stage and under active development, and its API
could be changed frequently.
:::

<div class="x4c-navgrid">

  <a class="x4c-navcard" href="ug-installation">
  <img src="assets/installation.png" alt="">
  <div class="x4c-navcard-title">Installation</div>
  <div class="x4c-navcard-body">Installation instructions.</div>
  </a>

  <a class="x4c-navcard" href="ug-core">
  <img src="assets/setup.png" alt="">
  <div class="x4c-navcard-title">Core Features</div>
  <div class="x4c-navcard-body"><code>x4c</code> as an Xarray extension.</div>
  </a>

  <a class="x4c-navcard" href="ug-post">
  <img src="assets/postprocessing.png" alt="">
  <div class="x4c-navcard-title">CESM Postprocessing</div>
  <div class="x4c-navcard-body">Timeseries generation using the <code>History</code> class.</div>
  </a>

  <a class="x4c-navcard" href="ug-diags">
  <img src="assets/diags.png" alt="">
  <div class="x4c-navcard-title">CESM Diagnostics</div>
  <div class="x4c-navcard-body">Diagnostics using the <code>Timeseries</code> class and the <code>spell</code> magics.</div>
  </a>

</div>

<div class="x4c-section">
  <h2>Made with <code>.x.plot()</code></h2>
  <p class="x4c-lede">Every figure below comes from a notebook in this documentation, and every one is reachable in a line or two. Click any to open the notebook that produced it.</p>

  <div class="x4c-hero-shot">
  <img src="assets/hero-sst-djf.png" alt="Global sea surface temperature, December-January-February, Robinson projection">
  <div class="x4c-hero-code"><code>case.plot("SST:-12,1,2")</code></div>
  </div>

  <div class="x4c-gallery">

  <a class="x4c-shot" href="diags-variables">
  <img src="assets/gallery-isotopes.png" alt="Four water-isotope maps">
  <div class="x4c-shot-meta">
  <div class="x4c-shot-title">Water isotopes, annual mean</div>
  <div class="x4c-shot-code">da.x.regrid(2,2).x.annualize().mean('time').x.plot(…)</div>
  </div>
  </a>

  <a class="x4c-shot" href="core-analysis">
  <img src="assets/gallery-eof.png" alt="First two EOFs of sea surface height">
  <div class="x4c-shot-meta">
  <div class="x4c-shot-title">EOFs of sea-surface height</div>
  <div class="x4c-shot-code">m.x.plot(ax=axd[k], ssv=ssv, cmap='RdBu_r', …)</div>
  </div>
  </a>

  <a class="x4c-shot" href="diags-variables">
  <img src="assets/gallery-ocean-land.png" alt="SST, SSS, mixed layer depth and land surface temperature">
  <div class="x4c-shot-meta">
  <div class="x4c-shot-title">Ocean and land state</div>
  <div class="x4c-shot-code">ocean['SST'].x.regrid(2,2).mean('time').x.plot(…)</div>
  </div>
  </a>

  <a class="x4c-shot" href="diags-quickview">
  <img src="assets/gallery-quickview.png" alt="Multi-panel diagnostics dashboard">
  <div class="x4c-shot-meta">
  <div class="x4c-shot-title">A whole diagnostics dashboard</div>
  <div class="x4c-shot-code">case.quickview(timespan=(1, 10))</div>
  </div>
  </a>

  <a class="x4c-shot" href="core-regridding">
  <img src="assets/gallery-regrid.png" alt="Native spectral element grid beside regridded output">
  <div class="x4c-shot-meta">
  <div class="x4c-shot-title">Native grid to regular lat/lon</div>
  <div class="x4c-shot-code">ds.x.da.isel(time=0).x.regrid().x.plot(…)</div>
  </div>
  </a>

  <a class="x4c-shot" href="diags-new-vars">
  <img src="assets/gallery-precip.png" alt="Annual mean precipitation">
  <div class="x4c-shot-meta">
  <div class="x4c-shot-title">A variable you defined yourself</div>
  <div class="x4c-shot-code">case.calc('PRECT_MM|regrid(2,2):ann')</div>
  </div>
  </a>

  <a class="x4c-shot" href="core-visualization">
  <img src="assets/gallery-moc.png" alt="Global meridional overturning circulation section">
  <div class="x4c-shot-meta">
  <div class="x4c-shot-title">Vertical sections, same call</div>
  <div class="x4c-shot-code">moc.x.plot(levels=np.linspace(-20, 20, 21))</div>
  </div>
  </a>

  <a class="x4c-shot" href="core-visualization">
  <img src="assets/gallery-styles.png" alt="Global mean surface temperature in a journal style">
  <div class="x4c-shot-meta">
  <div class="x4c-shot-title">Publication styles, built in</div>
  <div class="x4c-shot-code">x4c.set_style('journal')</div>
  </div>
  </a>

  </div>
</div>
