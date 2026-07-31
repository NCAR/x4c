import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import cartopy.crs as ccrs
import pathlib
import string

def showfig(fig, close=True):
    '''Show the figure

    Parameters
    ----------
    fig : matplotlib.pyplot.figure
        The matplotlib figure object

    close : bool
        if True, close the figure automatically

    '''

    plt.show()

    if close:
        closefig(fig)

def closefig(fig=None):
    '''Show the figure

    Parameters
    ----------
    fig : matplotlib.pyplot.figure
        The matplotlib figure object

    '''
    if fig is not None:
        plt.close(fig)
    else:
        plt.close()

def savefig(fig, path, verbose=True, **kws):
    ''' Save a figure to a path

    Parameters
    ----------
    fig : matplotlib.pyplot.figure
        the figure to save
    path : str
        the path to save the figure, can be ignored and specify in "settings" instead
    settings : dict
        the dictionary of arguments for plt.savefig(); some notes below:
        - "path" must be specified in settings if not assigned with the keyword argument;
          it can be any existed or non-existed path, with or without a suffix;
          if the suffix is not given in "path", it will follow "format"
        - "format" can be one of {"pdf", "eps", "png", "ps"}
        
    '''
    savefig_args = {'bbox_inches': 'tight', 'path': path}
    savefig_args.update(**kws)

    path = pathlib.Path(savefig_args['path'])
    savefig_args.pop('path')

    dirpath = path.parent
    if not dirpath.exists():
        dirpath.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f'Directory created at: "{dirpath}"')

    # append the default suffix, then save to *that* path. Previously the updated
    # `path` was computed but `path_str` (the original, extension-less) was passed to
    # savefig, so the `.pdf` default never took effect and the reported path pointed
    # at a file that did not exist.
    if path.suffix not in ['.eps', '.pdf', '.png', '.ps']:
        path = path.with_suffix(path.suffix + '.pdf')

    fig.savefig(str(path), **savefig_args)
    plt.close(fig)

    if verbose:
        print(f'Figure saved at: "{str(path)}"')

def set_style(style='journal', font_scale=1.0):
    ''' Modify the visualization style

    This function is inspired by [Seaborn](https://github.com/mwaskom/seaborn).
    See a demo in the example_notebooks folder on GitHub to look at the different styles

    Parameters
    ----------

    style : {journal, web, nature, agu, presentation, dark, minimal, matplotlib, _spines, _nospines, _grid, _nogrid}
        set the styles for the figure:
            - journal (default): fonts appropriate for paper
            - web: web-like font (e.g. ggplot)
            - nature: clean style inspired by Nature/Science figures with Helvetica-like fonts
            - agu: style following AGU journal conventions with minor ticks and enclosed axes
            - presentation: bold, high-contrast style for talks and posters
            - dark: modern dark theme for screen display and dashboards
            - minimal: ultra-clean style with thin lines and open layout
            - matplotlib: the original matplotlib style
            In addition, the following options are available:
            - _spines/_nospines: allow to show/hide spines
            - _grid/_nogrid: allow to show gridlines (default: _grid)

    font_scale : float
        Default is 1. Corresponding to 12 Font Size.

    '''
    font_dict = {
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
    }

    style_dict = {}
    inline_rc = mpl.rcParamsDefault.copy()
    inline_rc.update({
        'interactive': True,
    })
    mpl.rcParams.update(inline_rc)

    if 'journal' in style:
        style_dict.update({
            'axes.axisbelow': True,
            'axes.facecolor': 'white',
            'axes.edgecolor': 'black',
            'axes.grid': True,
            'grid.color': 'lightgrey',
            'grid.linestyle': '--',
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],

            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.right': False,
            'axes.spines.top': False,

            'legend.frameon': False,

            'axes.linewidth': 1,
            'grid.linewidth': 1,
            'lines.linewidth': 2,
            'lines.markersize': 6,
            'patch.linewidth': 1,

            'xtick.major.width': 1.25,
            'ytick.major.width': 1.25,
            'xtick.minor.width': 0,
            'ytick.minor.width': 0,
        })
    elif 'web' in style:
        style_dict.update({
            'figure.facecolor': 'white',

            'axes.axisbelow': True,
            'axes.facecolor': 'whitesmoke',
            'axes.edgecolor': 'lightgrey',
            'axes.grid': True,
            'grid.color': 'white',
            'grid.linestyle': '-',
            'xtick.direction': 'out',
            'ytick.direction': 'out',

            'text.color': 'grey',
            'axes.labelcolor': 'grey',
            'xtick.color': 'grey',
            'ytick.color': 'grey',

            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],

            'axes.spines.left': False,
            'axes.spines.bottom': False,
            'axes.spines.right': False,
            'axes.spines.top': False,

            'legend.frameon': False,

            'axes.linewidth': 1,
            'grid.linewidth': 1,
            'lines.linewidth': 2,
            'lines.markersize': 6,
            'patch.linewidth': 1,

            'xtick.major.width': 1.25,
            'ytick.major.width': 1.25,
            'xtick.minor.width': 0,
            'ytick.minor.width': 0,
        })
    elif 'nature' in style:
        style_dict.update({
            'figure.facecolor': 'white',
            'figure.dpi': 150,

            'axes.axisbelow': True,
            'axes.facecolor': 'white',
            'axes.edgecolor': '#333333',
            'axes.grid': False,
            'axes.linewidth': 0.8,
            'axes.prop_cycle': mpl.cycler('color', [
                '#0072B2', '#D55E00', '#009E73', '#CC79A7',
                '#F0E442', '#56B4E9', '#E69F00', '#000000',
            ]),

            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.top': True,
            'ytick.right': True,

            'font.family': 'sans-serif',
            'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif'],

            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.right': True,
            'axes.spines.top': True,

            'legend.frameon': False,
            'legend.handlelength': 1.5,

            'lines.linewidth': 1.5,
            'lines.markersize': 5,
            'patch.linewidth': 0.8,

            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.minor.width': 0.5,
            'ytick.minor.width': 0.5,
            'xtick.major.size': 4,
            'ytick.major.size': 4,
            'xtick.minor.size': 2,
            'ytick.minor.size': 2,
            'xtick.minor.visible': True,
            'ytick.minor.visible': True,
        })
        font_dict.update({
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.titlesize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
        })
    elif 'agu' in style:
        style_dict.update({
            'figure.facecolor': 'white',
            'figure.dpi': 150,

            'axes.axisbelow': True,
            'axes.facecolor': 'white',
            'axes.edgecolor': 'black',
            'axes.grid': False,
            'axes.linewidth': 1.0,

            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.top': True,
            'ytick.right': True,

            'font.family': 'sans-serif',
            'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif'],

            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.right': True,
            'axes.spines.top': True,

            'legend.frameon': True,
            'legend.fancybox': False,
            'legend.edgecolor': 'black',
            'legend.framealpha': 1.0,

            'lines.linewidth': 1.5,
            'lines.markersize': 5,
            'patch.linewidth': 1,

            'xtick.major.width': 1.0,
            'ytick.major.width': 1.0,
            'xtick.minor.width': 0.6,
            'ytick.minor.width': 0.6,
            'xtick.major.size': 5,
            'ytick.major.size': 5,
            'xtick.minor.size': 3,
            'ytick.minor.size': 3,
            'xtick.minor.visible': True,
            'ytick.minor.visible': True,

            'grid.color': '#CCCCCC',
            'grid.linestyle': ':',
            'grid.linewidth': 0.5,
        })
    elif 'presentation' in style:
        style_dict.update({
            'figure.facecolor': 'white',
            'figure.dpi': 100,

            'axes.axisbelow': True,
            'axes.facecolor': 'white',
            'axes.edgecolor': '#222222',
            'axes.grid': True,
            'axes.linewidth': 1.5,
            'axes.prop_cycle': mpl.cycler('color', [
                '#2176AE', '#E84855', '#57A773', '#F9A620',
                '#8B5CF6', '#06D6A0', '#EF476F', '#118AB2',
            ]),

            'grid.color': '#E5E5E5',
            'grid.linestyle': '-',
            'grid.linewidth': 0.8,

            'xtick.direction': 'out',
            'ytick.direction': 'out',

            'font.family': 'sans-serif',
            'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif'],

            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.right': False,
            'axes.spines.top': False,

            'legend.frameon': True,
            'legend.fancybox': True,
            'legend.edgecolor': 'none',
            'legend.facecolor': 'white',
            'legend.framealpha': 0.8,

            'lines.linewidth': 2.5,
            'lines.markersize': 8,
            'patch.linewidth': 1.5,

            'xtick.major.width': 1.5,
            'ytick.major.width': 1.5,
            'xtick.minor.width': 0,
            'ytick.minor.width': 0,
            'xtick.major.size': 6,
            'ytick.major.size': 6,
        })
        font_dict.update({
            'font.size': 16,
            'axes.labelsize': 18,
            'axes.titlesize': 20,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
        })
    elif 'dark' in style:
        style_dict.update({
            'figure.facecolor': '#1a1a2e',
            'figure.edgecolor': '#1a1a2e',
            'figure.dpi': 100,

            'axes.axisbelow': True,
            'axes.facecolor': '#16213e',
            'axes.edgecolor': '#4a4a6a',
            'axes.grid': True,
            'axes.linewidth': 0.8,
            'axes.labelcolor': '#c8c8d8',
            'axes.prop_cycle': mpl.cycler('color', [
                '#4CC9F0', '#F72585', '#7209B7', '#4361EE',
                '#3A0CA3', '#06D6A0', '#FFD166', '#EF476F',
            ]),

            'grid.color': '#2a2a4a',
            'grid.linestyle': '-',
            'grid.linewidth': 0.5,

            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'xtick.color': '#8888a8',
            'ytick.color': '#8888a8',

            'text.color': '#d0d0e0',

            'font.family': 'sans-serif',
            'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif'],

            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.right': False,
            'axes.spines.top': False,

            'legend.frameon': True,
            'legend.fancybox': True,
            'legend.edgecolor': 'none',
            'legend.facecolor': '#16213e',
            'legend.framealpha': 0.9,
            'legend.labelcolor': '#c8c8d8',

            'lines.linewidth': 2,
            'lines.markersize': 6,
            'patch.linewidth': 1,

            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.minor.width': 0,
            'ytick.minor.width': 0,

            'savefig.facecolor': '#1a1a2e',
            'savefig.edgecolor': '#1a1a2e',
        })
    elif 'minimal' in style:
        style_dict.update({
            'figure.facecolor': 'white',
            'figure.dpi': 150,

            'axes.axisbelow': True,
            'axes.facecolor': 'white',
            'axes.edgecolor': '#AAAAAA',
            'axes.grid': False,
            'axes.linewidth': 0.5,
            'axes.prop_cycle': mpl.cycler('color', [
                '#264653', '#2A9D8F', '#E9C46A', '#F4A261',
                '#E76F51', '#606C38', '#283618', '#DDA15E',
            ]),

            'xtick.direction': 'out',
            'ytick.direction': 'out',

            'font.family': 'sans-serif',
            'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif'],

            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.right': False,
            'axes.spines.top': False,

            'legend.frameon': False,
            'legend.handlelength': 1.2,

            'lines.linewidth': 1.5,
            'lines.markersize': 5,
            'patch.linewidth': 0.5,

            'xtick.major.width': 0.5,
            'ytick.major.width': 0.5,
            'xtick.minor.width': 0,
            'ytick.minor.width': 0,
            'xtick.major.size': 3,
            'ytick.major.size': 3,
        })
        font_dict.update({
            'font.size': 11,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
        })
    elif 'matplotlib' in style or 'default' in style:
        mpl.rcParams.update(inline_rc)
    else:
        print(f'Style [{style}] not available! Setting to `matplotlib` ...')
        mpl.rcParams.update(inline_rc)

    if '_spines' in style:
        style_dict.update({
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.right': True,
            'axes.spines.top': True,
        })
    elif '_nospines' in style:
        style_dict.update({
            'axes.spines.left': False,
            'axes.spines.bottom': False,
            'axes.spines.right': False,
            'axes.spines.top': False,
        })

    if '_grid' in style:
        style_dict.update({
            'axes.grid': True,
        })
    elif '_nogrid' in style:
        style_dict.update({
            'axes.grid': False,
        })

    # modify font size based on font scale
    font_dict.update({k: v * font_scale for k, v in font_dict.items()})

    for d in [style_dict, font_dict]:
        mpl.rcParams.update(d)

def infer_cmap(da):
    if 'long_name' in da.attrs:
        ln_lower = da.attrs['long_name'].lower()
        if 'temperature' in ln_lower:
            cmap = 'RdBu_r'
        elif 'pressure' in ln_lower:
            cmap = 'bwr_r'
        elif 'precipitation' in ln_lower:
            cmap = 'BrBG'
        elif 'correlation' in ln_lower:
            cmap = 'RdBu_r'
        elif 'R2' in ln_lower:
            cmap = 'Reds'
        elif 'salinity' in ln_lower:
            cmap = 'PiYG'
        elif 'circulation' in ln_lower:
            cmap = 'RdBu_r'
        elif 'depth' in ln_lower:
            cmap = 'GnBu'
        elif 'height' in ln_lower:
            cmap = 'PiYG'
        elif 'kmt' in ln_lower:
            cmap = 'BrBG'
        elif 'ice' in ln_lower:
            cmap = 'Blues'
        else:
            cmap = 'viridis'
    else:
        cmap = 'viridis'
    
    return cmap

def subplots(nrow:int, ncol:int, ax_loc:dict, projs=None, projs_kws=None, figsize=None, wspace=None, hspace=None,
             annotation=False, annotation_kws=None, annotation_separate=False, annotation_skip=None):

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrow, ncol)
    gs.update(wspace=wspace, hspace=hspace)
    ax = {}
    for k, i in ax_loc.items():
        if projs is not None and k in projs:
            projs_kws = {} if projs_kws is None else projs_kws
            if k not in projs_kws: projs_kws[k] = {}
            ax[k] = plt.subplot(gs[i], projection=ccrs.__dict__[projs[k]](**projs_kws[k]))
        else:
            ax[k] = plt.subplot(gs[i])

    if annotation:
        if annotation_separate:
            for i, k in enumerate(list(ax_loc)):
                if annotation_skip and k in annotation_skip:
                    continue
                annotation_kws = {} if annotation_kws is None else annotation_kws
                _annotation_kws = {'style': ')'}
                _annotation_kws.update(annotation_kws[k])
                add_annotation(ax[k], start=i, **_annotation_kws)
        else:
            annotation_kws = {} if annotation_kws is None else annotation_kws
            _annotation_kws = {'style': ')'}
            _annotation_kws.update(annotation_kws)
            add_annotation(ax, **_annotation_kws)

    return fig, ax

def add_annotation(ax, fs=20, loc_x=-0.15, loc_y=1.03, start=0, style=None):
    if type(ax) is dict:
        ax = ax.values()
    else:
        ax = [ax]

    if type(fs) is not list:
        fs = [fs] * len(ax)

    for i, v in enumerate(ax):
        letter_str = string.ascii_lowercase[i+start]

        if style == ')':
            letter_str = f'{letter_str})'
        elif style == '()':
            letter_str = f'({letter_str})'

        v.text(
            loc_x, loc_y, letter_str,
            transform=v.transAxes, 
            size=fs[i], weight='bold',
        )