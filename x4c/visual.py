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
    # if in_notebook:
    #     try:
    #         from IPython.display import display
    #     except ImportError as error:
    #         # Output expected ImportErrors.
    #         print(f'{error.__class__.__name__}: {error.message}')

    #     display(fig)

    # else:
    #     plt.show()

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

    path_str = str(path)
    if path.suffix not in ['.eps', '.pdf', '.png', '.ps']:
        path = pathlib.Path(f'{path_str}.pdf')

    fig.savefig(path_str, **savefig_args)
    plt.close(fig)

    if verbose:
        print(f'Figure saved at: "{str(path)}"')

def set_style(style='journal', font_scale=1.0):
    ''' Modify the visualization style
    
    This function is inspired by [Seaborn](https://github.com/mwaskom/seaborn).
    See a demo in the example_notebooks folder on GitHub to look at the different styles
    
    Parameters
    ----------
    
    style : {journal, web, matplotlib, _spines, _nospines,_grid,_nogrid}
        set the styles for the figure:
            - journal (default): fonts appropriate for paper
            - web: web-like font (e.g. ggplot)
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
        else:
            cmap = 'viridis'
    else:
        cmap = 'viridis'
    
    return cmap

def subplots(nrow:int, ncol:int, ax_loc:dict, projs=None, projs_kws=None, figsize=None, wspace=None, hspace=None,
             annotation=False, annotation_kws=None, annotation_separate=False,):

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