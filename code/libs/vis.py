import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
import pandas as pd
import numpy as np
from pandas.plotting import parallel_coordinates
from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib.patches as mpatches

from libs import constants
from libs.metrics import aggregators

def sns_reset():
    sns.reset_orig()

def sns_paper_style(font_scale=1.51):
    sns.set_context("paper", font_scale=font_scale) #rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":5}) 
    rc('font', family = 'serif')
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False

def _finish_plot(fig, fn = None):
    plt.tight_layout()
    
    if fn is not None:
        fig.savefig(fn, dpi=constants.FIG_DPI, bbox_inches='tight')

    plt.show()
    plt.close()

def _polish_plot(fig, ax, x_col=None, y_col=None, group_col=None, x_order=None, **kwargs):
    xlabel_rotation = kwargs.get('xlabel_rotation', None)
    xlabel = kwargs.get('xlabel', x_col)
    ylabel = kwargs.get('ylabel', y_col) 
    yscale = kwargs.get('yscale', 'linear')
    xscale = kwargs.get('xscale', 'linear')
    xticks = kwargs.get('xticks', False)
    xticklabels_rename = kwargs.pop('xticklabels_rename', None)
    ylim = kwargs.get('ylim', None)
    xtickrot = kwargs.get('xtickrot', None)
    title = kwargs.get('title', None)
    axhline = kwargs.get('axhline', None)

    legend_kwargs = kwargs.get('legend_kwargs', {})
    legend = kwargs.get('legend', True)

    if title:
        fig.suptitle(title, ha='center', va='bottom')

    if axhline:
        ax.axhline(**axhline)

    ytickmax = kwargs.get('ytickval', None)
    if ytickmax:
        # Customize radial ticks
        yticks = ax.get_yticks()
        ax.set_yticks(yticks)  # Ensure ticks are updated
        if ytickmax < 0.5:
            yticklabels = [f"{tick:.2f}"  for i, tick in enumerate(yticks)]
        else:
            yticklabels = [f"{tick:.1f}" if i % 2 == 0 else "" for i, tick in enumerate(yticks)]
        ax.set_yticklabels(yticklabels)  # Show every second tick label

    # Rotate labels
    if xtickrot:
        for tick in ax.get_xticklabels():
            tick.set_rotation(xtickrot)
            # tick.set_ha('right')
            # tick.set_rotation(45)

    # Add labels and legend
    if x_col:
        ax.set_xlabel(xlabel)
        
    if y_col:
        ax.set_ylabel(ylabel)
    
    if xlabel or xlabel is None:
        ax.set_xlabel(xlabel)

    if ylabel or ylabel is None:
        ax.set_ylabel(ylabel)
    
    if xscale:
        ax.set_xscale(xscale)

    if yscale:
        ax.set_yscale(yscale)

    if (group_col and legend) or legend:
        # plt.legend(title=legend_title, **legend_kwargs)
        fig.legend(**legend_kwargs)

    # Adjust x-axis order if provided
    if x_order is not None:
        plt.xticks(ticks=np.arange(len(x_order)), labels=x_order)

    # Apply rotation to x-axis labels if specified
    if xlabel_rotation is not None:
        plt.xticks(rotation=xlabel_rotation)

    if xticks is None:
        ax.set_xticklabels([])

    if xticklabels_rename:
        xticks = ax.get_xticklabels()
        newxticks = [xticklabels_rename[t.get_text()] for t in xticks]
        ax.set_xticklabels(newxticks)
    
    if ylim:
        ax.set_ylim(ylim)

def plot_barplot(data, x_col, y_col, group_col=None, x_order=None, group_order=None, fn=None, **kwargs):
    """
    Plots a barplot using matplotlib.

    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - x_col (str): Column name for x-axis.
    - y_col (str): Column name for y-axis.
    - group_col (str, optional): Column name for grouping (to stack bars). Default is None.
    - x_order (list, optional): Order of categories for the x-axis. Default is None (no specific order).
    - group_order (list, optional): Order of group categories. Default is None (no specific order).
    - **kwargs: Additional keyword arguments:
        - xlabel_rotation (int, optional): Rotation angle for x-axis labels. Default is None (no rotation).
        - xlabel (str, optional): Custom label for the x-axis. Default is the x_col name.
        - ylabel (str, optional): Custom label for the y-axis. Default is the y_col name.
        - legend_title (str, optional): Custom title for the legend. Default is the group_col name.
        - colors (dict, optional): Dictionary mapping group names to colors. Default is None.
        - std_col (str, optional): Column name for the standard deviation values. Default is None.

    Returns:
    - None
    """
    
    df = data.copy()
    figsize = kwargs.get('figsize', (10,5))
    colors = kwargs.get('colors', None)
    err_col = kwargs.get('err_col', None)
    bar_width = kwargs.get('bar_width', 0.5)
    ax = kwargs.pop('ax', None)
    fig = kwargs.pop('fig', None)
    finish = kwargs.pop('finish', True)

    # Set x-axis order
    if x_order is not None:
        df[x_col] = pd.Categorical(df[x_col], categories=x_order, ordered=True)
        df = df.sort_values(by=x_col)

    # Set group order
    if group_col and group_order is not None:
        df[group_col] = pd.Categorical(df[group_col], categories=group_order, ordered=True)

    if ax is None or fig is None:
        fig, ax = plt.subplots(1,1,figsize=figsize)

    if group_col:
        groups = group_order if group_order is not None else df[group_col].unique()
        x_positions = np.arange(len(df[x_col].unique()))

        
        for i, group in enumerate(groups):
            group_data = df[df[group_col] == group]
            if group_data.empty:
                continue
            group_data = group_data if x_order is None else group_data.sort_values(by=x_col).reset_index(drop=True)
            x_data = group_data.groupby(x_col, observed=False)[y_col].sum()
            err_data = group_data.groupby(x_col, observed=False)[err_col].sum() if err_col else None
            bottom_values = np.zeros(len(x_data))

            ax.bar(
                x=x_data.index,
                height=x_data,
                bottom=bottom_values,
                width=bar_width,
                label=str(group),
                color=colors[group] if colors and group in colors else None
            )

            if err_col:
                ax.errorbar(
                    x=x_data.index,
                    y=x_data + bottom_values,
                    yerr=err_data,
                    fmt='none',
                    ecolor='black',
                    capsize=5
                )

            bottom_values += x_data.reindex(x_order if x_order else x_data.index, fill_value=0).values
    else:
        # No grouping
        df = df if x_order is None else df.sort_values(by=x_col).reset_index(drop=True)
        y_data = df[y_col]
        err_data = df[err_col] if err_col else None

        ax.bar(
            x=y_data.index,
            height=y_data,
            width=bar_width,
        )

        if err_col:
            ax.errorbar(
                x=y_data.index,
                y=y_data,
                yerr=err_data,
                fmt='none',
                ecolor='black',
                capsize=5
            )

    # Display the plot
    _polish_plot(fig, ax, x_col, y_col, group_col, x_order, **kwargs)
    _finish_plot(fig, fn)



def _plot_spiderweb(categories, values, label=None, color='blue', fig=None, ax=None, **kwargs):
    """
    Plots a spiderweb (radar) plot.

    Parameters:
    - categories (list): List of category names.
    - values (list): List of values corresponding to the categories.
    - title (str, optional): Title of the plot. Default is "".
    - color (str, optional): Color for the plot. Default is 'blue'.

    Returns:
    - None
    """
    
    figsize = kwargs.get('figsize', (10,5))
    title = kwargs.get('title', '')
    ylim = kwargs.get('ylim', None)

    # Ensure the data is circular (start and end points match)
    categories = list(categories)
    values = list(values)
    values += values[:1]

    # Calculate angles for the axes
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    # Create the plot
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw=dict(polar=True))
        
    # Draw the spiderweb
    ax.fill(angles, values, color=color, alpha=0.1)
    ax.plot(angles, values, label=label, color=color, linewidth=2)

    # Style grid
    ax.set_rscale('linear')
    ax.grid(color='gray', linestyle='--', linewidth=0.5)

    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, ha='center')
    
    # Remove polar labels
    # ax.yaxis.grid(True)

    # Add category labels
    # ax.set_xticks(angles[:-1])
    # ax.set_xticklabels(categories)

    # Set radial limit
    ax.set_ylim(ylim)

    # Customize circular gridlines
    #ax.yaxis.grid(color='gray', linestyle='solid', linewidth=0.4)  # Lighter gridlines
    
    # Remove the outermost border (spine)
    ax.spines['polar'].set_visible(False)
    
    return fig, ax

def plot_spiderweb(data, x_col, x_order, y_col, hue, hue_order, hue_colors, fn=None, **kwargs):
    
    figsize = kwargs.get('figsize', (10,5))
    ylim = kwargs.get('ylim', (-0.05,1.05))
    df_data = data.copy()

    fig, ax = None, None
    for group in hue_order:
        df = df_data.query(f'{hue}==@group').copy()
        try:
            values = [df.query(f"{x_col}==@x").iloc[0][y_col] for x in x_order]
            fig, ax = _plot_spiderweb(x_order, values, label=group, color=hue_colors[group], fig=fig, ax=ax, 
                                        figsize=figsize, 
                                        ylim=ylim)
            
            # ax.set_xticklabels(x_order, fontsize=10, ha='center')
        except Exception as e:
            print(group, df.shape, e)

    # _polish_plot(fig, ax, x_col=x_col, group_col=hue, group_order=hue_order, x_order=x_order, **kwargs)
    _finish_plot(fig, fn)


def plot_parallel_coords(df, hue, hue_order, hue_colors, df_err=None, fn=None, **kwargs):

    figsize = kwargs.get('figsize', (10,5))
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    baselines = kwargs.pop('baselines', None)
    
    
    parallel_coordinates(df, hue, color=[hue_colors[m] for m in hue_order if m in df.model.unique()], axvlines=False, ax=ax, lw=2,)
    plt.gca().legend_.remove()

    # Add error shading
    if df_err is not None and hue:
        for group, tmp in df_err.groupby(hue, observed=False):

            if tmp.empty:
                continue
            
            v_err = tmp.drop(columns=hue).T
            index = v_err.index.to_list()
            v_val = df.set_index(hue).loc[group].loc[index].values.astype(float)
            v_err = v_err.values.reshape(1,-1)[0].astype(float)

            ax.fill_between(
                index,
                v_val - v_err,  # Lower bound
                v_val + v_err,  # Upper bound
                color=hue_colors[group],
                alpha=0.1,  # Transparency
            )


    if baselines is not None:
        for bname, obj in baselines.items():
            baseline = obj['values']
            color = obj.get('color', 'black')
            ls = obj.get('ls', '-')
            x = baseline.index.categories.to_list()
            y = [baseline[i] if i in baseline else None for i in x ]
            ax.plot(x, y, color=color, linestyle=ls, lw=1.5, label=bname)

    
    ax.grid(axis='y')

    _polish_plot(fig, ax, x_col=None, group_col=hue, group_order=hue_order, x_order=df.columns[1:], **kwargs)
    _finish_plot(fig, fn)


def plot_components_by_model_including_population(results, colors, col_order, fn=None, **kwargs):
    
    colors.update({'APS': constants.COMPONENT_POPULATION_COLOR})

    ncols = len(col_order)
    nrows = 1
    w = kwargs.pop('width', 2)
    h = kwargs.pop('height', 2)
    figsize = (ncols * w, nrows * h)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)

    for i, model in enumerate(col_order):
        obj = results.get(model, None)

        if obj is None:
            continue

        r = i//ncols
        c = i%ncols
        ax = axes[r,c] if nrows > 1 else axes[c]
        
        combined_data = obj['reduction']

        y = 1.0
        for label, data in combined_data.groupby('label'):    
            col = colors[label]    
            ax.scatter(data['dim1'], data['dim2'], label=model if label==model else None, 
                    alpha=1 if label==model else 0.5, 
                    color=col, rasterized=True)
            
            pop = data.id_author_oa.nunique()
            ax.text(s=f"{pop} people", x=0.6, y=y, color=col,
                    fontsize=8, ha='right', va='top', transform=ax.transAxes) 
            
            y -= 0.08

        ax.set_title(model)

        for artist in ax.artists:
            artist.set_rasterized(True)
        #ax.set_rasterized(True)

        if kwargs.get('xlabel', True):
            ax.set_xlabel('PC 1')

        y_col = 'PC2' if c==0 else ''

        _polish_plot(fig, ax, x_col=None, y_col=y_col, group_col=None, group_order=None, **kwargs)

    plt.subplots_adjust(wspace=0.0, hspace=0.01)
    _finish_plot(fig, fn)
 


def plot_components_by_model_and_param_value_including_population(results, col_order, hue_order, fn=None, **kwargs):
    
    colors = constants.COMPPONENT_TASK_PARAM_COLORS
    color_map = {task_param: colors[i] for i, task_param in enumerate(hue_order)}

    markers = constants.COMPPONENT_TASK_PARAM_MARKERS
    marker_map = {task_param: markers[i] for i, task_param in enumerate(hue_order)}

    annotated_flag = {model:{} for model in col_order}

    for mode in col_order:
        annotated_flag[mode] = {task_param: False for task_param in hue_order}

    xticks = kwargs.get('xticks', False)
    title = kwargs.pop('title', None)

    ncols = len(col_order)
    nrows = 1
    w = kwargs.pop('width', 2)
    h = kwargs.pop('height', 1.7 if xticks is None else 2.2)
    
    figsize = (ncols * w, nrows * h)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)

    for (model, task_name, task_param), obj in results.items():
        
        c = col_order.index(model)

        ax = axes[c]

        if obj is not None:

            combined_data = obj['reduction']

            sample_size = combined_data.query('label != "APS"').groupby('task_param', observed=False).size()
            
            for label, data in combined_data.groupby('label', observed=True): 
                if label == 'APS':
                    ax.scatter(data['dim1'], data['dim2'], alpha=0.1, color=constants.COMPONENT_POPULATION_COLOR, zorder=-1, rasterized=True) 

                    # PLOT REFERENCE (TWINS)
                    if task_name == 'twins':

                        flag = sum([int(c in task_param) for c in ['famous', 'random']])
                        if flag > 0:
                            if 'famous' in task_param:
                                # Réka Albert: 43067
                                # Albert‐László Barabási: 44475
                                id = 43067 if 'female' in task_param else 44475
                                _tmp = data.loc[[id]]
                            elif 'random' in task_param:
                                # female: 130493
                                # male: 55759
                                id = 130493 if 'female' in task_param else 55759
                                _tmp = data.loc[[id]]

                            # plot the reference
                            ax.scatter(_tmp['dim1'], _tmp['dim2'], 
                                label=task_param, 
                                color='black',
                                alpha=1.0,
                                marker=10 if 'female' in task_param else 11, # up 10, down 11
                                zorder=10e10,
                                rasterized=True,
                                )
                        
                else:
                    
                    for i, (task_param, df) in enumerate(data.groupby('task_param', observed=False)):
                        
                        color = color_map[task_param]
                        marker = marker_map[task_param]

                        ax.scatter(df['dim1'], df['dim2'], 
                                label=task_param, 
                                color=color,
                                alpha=0.7,
                                marker=marker,
                                zorder=1/sample_size[task_param], # the smallest group on top
                                rasterized=True
                                )

                                       
                        y = 0.01 + hue_order.index(task_param) * 0.08
                        ax.text(s=f"{df.shape[0]} people", x=0.6, y=y, color=color,fontsize=8, ha='right', va='bottom', transform=ax.transAxes) 
                        annotated_flag[model][task_param] = True

        for artist in ax.artists:
            artist.set_rasterized(True)
        #ax.set_rasterized(True)

    
    # if param was not found in the data
    for model, obj_a in annotated_flag.items():
        c = col_order.index(model)
        ax = axes[c]

        if title:
            ax.set_title(model)

        y_col = 'PC2' if c==0 else False
        kwargs['legend'] = False    
        _polish_plot(fig, ax, x_col='PC1', y_col=y_col, group_col=None, group_order=None, **kwargs)

        for task_param, flag in obj_a.items():
            if not flag:
                color = color_map[task_param]
                y = 0.01 + hue_order.index(task_param) * 0.08
                ax.text(s=f"0 people", x=0.6, y=y, color=color,fontsize=8, ha='right', va='bottom', transform=ax.transAxes) 
   

    # add legend
    c = 2
    custom_legend = [Line2D([0], [0], 
                            color=color, marker=marker_map[label], 
                            markersize=6, 
                            label=label.split('_')[-1].replace('career','early_career') if not label.startswith('top_') else label, 
                            linestyle="") for label, color in color_map.items()] 
    axes[c].legend(
        handles=custom_legend,  # Add custom legend handles
        loc="upper right",      # Position at the top-right corner
        frameon=True,           # Add a frame around the legend
        borderpad=0.2,          # Adjust padding inside the legend box
        handlelength=0.5,       # Adjust the length of the colored marker
        fontsize="small"        # Adjust text size
    )

    plt.subplots_adjust(wspace=0.0, hspace=0.01)
    _finish_plot(fig, fn)



def plot_error_box_plot_comparison(df_sample, df_full, metric, fn=None, **kwargs):

    figsize = kwargs.get('figsize', (5, 4))
    group_col = 'model'
    categories_yaxis = constants.LLMS
    colors_categories_yaxis = constants.LLMS_COLORS   
    yticks = kwargs.get('yticks', True)

    #########################################
    # Filter out NaN values
    data_llm = df_sample.copy()
    data_llm.dropna(subset=[metric], inplace=True)

    data_gt = df_full.copy()
    data_gt.dropna(subset=[metric], inplace=True)

    #########################################
    # Create the figure and gridspec
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.0)

    #########################################
    # Top Panel: Population Distribution
    ax_top = fig.add_subplot(gs[0])
    sns.boxplot(data=data_gt,  x=metric, ax=ax_top, color='lightgray')
    if yticks:
        ax_top.set_ylabel('APS')
    for artist in ax_top.artists:  # Rasterize individual boxes
        artist.set_rasterized(True)
    ax_top.set_rasterized(True)
    ax_top.set_xlabel('')
    ax_top.set_yticks([])
    ax_top.axes.get_xaxis().set_visible(False)

    #########################################
    # Main Panel: Violin Plot or Box Plot
    ax_main = fig.add_subplot(gs[1], sharex=ax_top)
    sns.boxplot(data=data_llm, y=group_col, x=metric, ax=ax_main, hue=group_col, palette=constants.LLMS_COLORS)
    ax_main.set_xlabel(metric)
    ax_main.set_ylabel('')
    if yticks:
        ax_main.set_yticklabels(categories_yaxis)
    else:
        ax_main.set_yticklabels([])
    
    #########################################
    # legend
    legend = kwargs.get('legend', False)
    colors_categories_yaxis.update({'APS': 'lightgray'})
    if legend:
        legend_patches = [mpatches.Patch(color=colors_categories_yaxis[g], label=g) for g in  ['APS'] + categories_yaxis]
        # ax_main.legend(
        #         handles=legend_patches,  # Add custom legend handles
        #         loc="upper right",      # Position at the top-right corner
        #         frameon=True,           # Add a frame around the legend
        #         borderpad=0.4,          # Adjust padding inside the legend box
        #         handlelength=0.6,       # Adjust the length of the colored marker
        #         fontsize="small"        # Adjust text size
        # )
        legend_kwargs = kwargs.get('legend_kwargs', {'loc': 'upper right', 'frameon': True, 'borderpad': 0.4, 'handlelength': 0.6, 'fontsize': 'small'})    
        legend_kwargs['handles'] = legend_patches
        fig.legend(**legend_kwargs)

    # Adjust aesthetics
    ax_top.spines['bottom'].set_visible(False)
    ax_main.spines['top'].set_visible(False)
    ax_main.set_xscale('log')
    for artist in ax_main.artists:  # Rasterize individual boxes
        artist.set_rasterized(True)
    ax_main.set_rasterized(True)
    ax_main.set_xlabel('')
    ax_main.set_yticks([])

    #########################################
    _finish_plot(fig, fn)




def plot_task_param_comparison_bars(df_result, metric, all_labels, model, color_map_attribute, fn=None, **kwargs):
    

    # Step 1: Group and Aggregate Counts
    grouped = df_result.query("model==@model").groupby(["task_name", "task_param", "attribute_label"]).sum().reset_index()

    # Step 2: Prepare Data for Plotting
    # Pivot data to organize counts by task_name and task_param/gender
    pivoted = grouped.pivot_table(
        index="task_name",
        columns=["ax", "attribute_label"],
        values=metric,
        fill_value=0,
    )

    # Define the unique task_names and task_params
    task_names = df_result.task_name.unique()
    task_params = [0,1]

    # sort index
    pivoted = pivoted.reindex(task_names)

    #################################################
    # Plot setup
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 4.5), gridspec_kw={'width_ratios': [2.5, 2.5, 2.5]})
    #################################################

    #################################################
    # Left plot: First task_param
    #################################################
    task_param1 = task_params[0]
    bottom = np.zeros(len(task_names))  # Initialize stacking
    for label in all_labels:
        if (task_param1, label) in pivoted.columns:
            axes[0].barh(
                task_names,
                pivoted[(task_param1, label)].apply(lambda x: x),  # Invert to stack from top to bottom (remove - sign if x-axis LR)
                # label=gender,
                left=bottom,  # Add previous stacks
                alpha=0.7,
                color=color_map_attribute[label]
            )
            bottom += pivoted[(task_param1, label)].apply(lambda x: x)  # Invert to stack from top to bottom (remove - sign if x-axis LR)
    axes[0].invert_yaxis()
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['left'].set_visible(False)


    #################################################
    # Middle subplot: Names
    #################################################
    axes[1].set_yticks([])
    axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['left'].set_visible(False)
    axes[1].spines['bottom'].set_visible(False)
    axes[1].invert_yaxis()  # Match the left subplot
    axes[1].set_xlim(-1, 1)

    y = 0.1035
    for i, name in enumerate(task_names):
        axes[1].text(s=name, x=0, y=0.09 + (i*y), ha='center', va='center', color='black')


    #################################################
    # Right plot: Second task_param
    #################################################
    task_param2 = task_params[1]
    bottom = np.zeros(len(task_names))  # Reset stacking for the second subplot
    for label in all_labels:
        if (task_param2, label) in pivoted.columns:
            axes[2].barh(
                task_names,
                pivoted[(task_param2, label)],
                label=label.replace('Unisex','Neutral').replace(constants.ETHNICITY_BLACK,'Black').replace(constants.ETHNICITY_LATINO,'Latino'),
                left=bottom,  # Add previous stacks
                alpha=0.7,
                color=color_map_attribute[label]
            )
            bottom += pivoted[(task_param2, label)]
    axes[2].invert_yaxis()
    axes[2].set_yticks([])
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    axes[2].spines['left'].set_visible(False)


    #################################################
    # Y-ticks
    #################################################
    # Synchronize x-axis limits
    if metric == 'counts':
        max_x = max(axes[0].get_xlim()[0]*-1, axes[2].get_xlim()[1])
    else:
        max_x = 1
    # axes[0].set_xlim(-max_x,0)   # comment if x-axis L-R
    axes[0].set_xlim(0, max_x) # uncomment if x-axis L-R
    axes[2].set_xlim(0, max_x)
     

    for i in [0,1]:
        j = i * 2
        ax = axes[j]

        # add task_param values
        y_ticks = [df_result.groupby(['ax','task_name']).task_param.unique()[i,task_name][0].replace(f"{task_name.split('_')[-1]}_",'') for task_name in task_names]
        y_ticks = [t.replace('male','(male)').replace('fe(male)','(female)') for t in y_ticks]
        ax = axes[j]
        if i == 1:
            ax2 = ax.twinx()
            ax2.set_yticks(range(task_names.shape[0]))
            ax2.set_yticklabels(y_ticks, color='grey')
            # ax2.invert_yaxis() # comment if x-axis L-R
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.set_ylim(ax.get_ylim())
        else:
            ax.set_yticks(range(task_names.shape[0]))
            ax.set_yticklabels(y_ticks, color='grey')
        
        # remove sign
        ticks = ax.get_xticks()
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(abs((tick))) if max_x==1 else str(abs(int(tick))) for tick in ticks])

    # Adjust subplot appearance
    xlabel = "Fraction" if metric=='percentage' else "Unique counts"
    axes[0].set_xlabel(xlabel)
    axes[2].set_xlabel(xlabel)

    bbox = (0.06, 0.93, 0.5, 0.5) if constants.GENDER_FEMALE in all_labels else [0.0, 0.93, 0.5, 0.5]
    ncols = 5  
    fig.legend(loc="lower left", ncols=ncols, bbox_to_anchor=bbox)

    # Overall layout
    plt.subplots_adjust(wspace=0.05)  # Reduce horizontal space between subplots
    _finish_plot(fig, fn)




def plot_model_comparison_bars(df_result, metric, all_labels, task_name, color_map_attribute, twins_pair=None, minority_baselines=None, fn=None, **kwargs):
    
    # Step 1: Group and Aggregate Counts
    if task_name == constants.EXPERIMENT_TASK_TWINS:
        if twins_pair is None:
            raise ValueError("The twins task requires a twins_pair: famous, random, tv, ficticious, politic")
        if twins_pair not in constants.TASK_TWINS_GROUP_ORDER:
            raise ValueError(f"Twins_pair must be: {constants.TASK_TWINS_GROUP_ORDER}")
        
        grouped = df_result.query(f"task_name=='{task_name}_{twins_pair}'").groupby(["model", "task_param", "attribute_label"]).sum().reset_index()
    else: 
        grouped = df_result.query("task_name==@task_name").groupby(["model", "task_param", "attribute_label"]).sum().reset_index()
        
    # Step 2: Prepare Data for Plotting
    # Pivot data to organize counts by task_name and task_param/gender
    pivoted = grouped.pivot_table(
        index="model",
        columns=["ax", "attribute_label"],
        values=metric,
        fill_value=0,
    )

    # Define the unique task_names and task_params
    row_names = df_result.model.unique()
    task_params = [0,1]

    # sort index
    pivoted = pivoted.reindex(row_names)

    #################################################
    # Plot setup
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 4.), gridspec_kw={'width_ratios': [0.4, 0.2, 0.4]})

    #################################################

    #################################################
    # Left plot: First task_param
    #################################################
    task_param1 = task_params[0]
    bottom = np.zeros(len(row_names)) # Initialize stacking
    bottom_gt = 0
    ax = axes[0]
    for label in all_labels:
        if (task_param1, label) in pivoted.columns:
            ax.barh(
                row_names,
                pivoted[(task_param1, label)].apply(lambda x: x),  # Invert to stack from top to bottom (remove - sign if x-axis LR)
                left=bottom,  # Add previous stacks
                alpha=0.7,
                color=color_map_attribute[label]
            )
            bottom += pivoted[(task_param1, label)].apply(lambda x: x)  # Invert to stack from top to bottom (remove - sign if x-axis LR)
        
        # Ground-truth
        if minority_baselines is not None and task_name in minority_baselines:
            yy = minority_baselines[task_name].iloc[task_param1][label]
            ax.barh(
                ['Ground-truth'],
                yy,
                left=bottom_gt,  # Add previous stacks
                alpha=1.0,
                color=color_map_attribute[label]
            )
            bottom_gt += yy
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)


    #################################################
    # Middle subplot: Names (models)
    #################################################
    axes[1].set_yticks([])
    axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['left'].set_visible(False)
    axes[1].spines['bottom'].set_visible(False)
    axes[1].invert_yaxis()  # Match the left subplot
    axes[1].set_xlim(-1, 1)

    # Models
    y = 0.158 if minority_baselines is None or task_name not in minority_baselines.keys() else 0.135 
    for i, name in enumerate(row_names):
        axes[1].text(s=name, x=0, y=0.105 + (i*y), ha='center', va='center', color='black')
    if minority_baselines is not None and task_name in minority_baselines.keys():
        axes[1].text(s='ground-truth', x=0, y=0.105 + ((i+1)*y), ha='center', va='center', color='black', weight='bold')


    #################################################
    # Right plot: Second task_param
    #################################################
    task_param2 = task_params[1]
    bottom = np.zeros(len(row_names)) # Reset stacking for the second subplot
    bottom_gt = 0
    ax = axes[2]
    for label in all_labels:
        if (task_param2, label) in pivoted.columns:
            ax.barh(
                row_names,
                pivoted[(task_param2, label)],
                label=label.replace('Unisex','Neutral').replace(constants.ETHNICITY_BLACK,'Black').replace(constants.ETHNICITY_LATINO,'Latino'),
                left=bottom,  # Add previous stacks
                alpha=0.7,
                color=color_map_attribute[label]
            )
            bottom += pivoted[(task_param2, label)]
        # Ground-truth
        if minority_baselines is not None and task_name in minority_baselines:
            yy = minority_baselines[task_name].iloc[task_param2][label]
            ax.barh(
                ['Ground-truth'],
                yy,
                left=bottom_gt,  # Add previous stacks
                alpha=1.0,
                color=color_map_attribute[label]
            )
            bottom_gt += yy
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    #################################################
    # Minority Baselines
    #################################################
    ### When minority_baselines is a pd.Series per task (only the minority group)
    if minority_baselines is not None:
        _label = constants.GENDER_LIST[0] if constants.GENDER_FEMALE in all_labels else constants.ETHNICITY_LIST[0]
        if task_name in minority_baselines:
            fb = minority_baselines[task_name][_label]
            for axid in [0,1]:
                axes[axid * 2].axvline(fb.iloc[axid], color=color_map_attribute[_label], ls='--', lw=1)

    # Task params
    df_task_params = grouped[['ax','task_param']].drop_duplicates()
    for id, row in df_task_params.iterrows():
        ax = axes[row.ax * 2]
        ax.text(s=row.task_param, x=0.5, y=1.05, transform=ax.transAxes, ha='center', va='top')

    # Adjust subplot appearance
    xlabel = "Fraction" if metric=='percentage' else "Unique counts"
    axes[0].set_xlabel(xlabel)
    axes[2].set_xlabel(xlabel)
    axes[0].set_yticks([])
    axes[2].set_yticks([])
    axes[0].set_xlim(0, 1)
    axes[2].set_xlim(0, 1)

    bbox = (0.08, 0.95, 0.5, 0.5) if constants.GENDER_FEMALE in all_labels else [0.0, 0.93, 0.5, 0.5]
    ncols = 4 if constants.GENDER_FEMALE in all_labels else 5
    fig.legend(loc="lower left", ncols=ncols, bbox_to_anchor=bbox)

    # Overall layout
    plt.subplots_adjust(wspace=0.05)  # Reduce horizontal space between subplots
    _finish_plot(fig, fn)




def plot_task_param_comparison_line(df_result, col_val, col_err, model, fn=None, **kwargs):
    

    # Step 1: Group and Aggregate Counts
    grouped = df_result.query("model==@model").groupby(["task_name", "task_param", "attribute_label"]).sum().reset_index()

    # Step 2: Prepare Data for Plotting
    # Pivot data to organize counts by task_name and task_param/gender
    pivoted_vals = grouped.pivot_table(
        index="task_name",
        columns=["ax", "attribute_label"],
        values=col_val,
        fill_value=0,
    )

    pivoted_errs = [None,None]
    if col_err:
        pivoted_errs = grouped.pivot_table(
            index="task_name",
            columns=["ax", "attribute_label"],
            values=col_err,
            fill_value=0,
        )

    # Define the unique task_names and task_params
    task_names = df_result.task_name.unique()

    # sort index
    pivoted_vals = pivoted_vals.reindex(task_names)
    pivoted_errs = pivoted_errs.reindex(task_names) if pivoted_errs is not None else None
    
    #################################################
    # Plot setup
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 4.5), gridspec_kw={'width_ratios': [1, 5, 5]})

    #################################################

    #################################################
    # Left plot: First task_param
    #################################################

    index = 0
    axes[1].errorbar(np.squeeze(pivoted_vals[index].values), task_names, xerr=np.squeeze(pivoted_errs[index].values), fmt='o', color='red', label="Mean ± SD")
    axes[1].invert_yaxis()
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['left'].set_visible(False)
    axes[1].set_yticks([])

    #################################################
    # Middle subplot: Names
    #################################################
    axes[0].set_yticks([])
    axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['left'].set_visible(False)
    axes[0].spines['bottom'].set_visible(False)
    axes[0].invert_yaxis()  # Match the left subplot
    axes[0].set_xlim(-1, 1)

    y = 0.1035
    for i, name in enumerate(task_names):
        axes[0].text(s=name, x=0, y=0.09 + (i*y), ha='right', va='center', color='black')


    #################################################
    # Right plot: Second task_param
    #################################################
    index = 1
    axes[2].errorbar(np.squeeze(pivoted_vals[index].values), task_names, xerr=np.squeeze(pivoted_errs[index].values), fmt='o', color='red', label="Mean ± SD")
    axes[2].invert_yaxis()
    axes[2].set_yticks([])
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    axes[2].spines['left'].set_visible(False)


    #################################################
    # Y-ticks
    #################################################
    # Synchronize x-axis limits
    max_x = max(axes[1].get_xlim()[1], axes[2].get_xlim()[1])
    min_x = min(axes[1].get_xlim()[0], axes[2].get_xlim()[0])
    smooth = 0
    axes[1].set_xlim(min_x-smooth, max_x+smooth)
    axes[2].set_xlim(min_x-smooth, max_x+smooth)
     

    for i in [0,1]:
        j = i + 1
        ax = axes[j]

        # add task_param values
        y_ticks = [df_result.groupby(['ax','task_name']).task_param.unique()[i,task_name][0].replace(f"{task_name.split('_')[-1]}_",'') for task_name in task_names]
        y_ticks = [t.replace('male','(male)').replace('fe(male)','(female)') for t in y_ticks]
        
        ax2 = ax.twinx()
        ax2.set_yticks(range(task_names.shape[0]))
        ax2.set_yticklabels(y_ticks, color='grey')
        ax2.invert_yaxis()
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.set_ylim(ax.get_ylim())
    

    # Adjust subplot appearance
    xlabel = "Mean ± SD"
    axes[1].set_xlabel(xlabel)
    axes[2].set_xlabel(xlabel)

    # Overall layout
    plt.subplots_adjust(wspace=0.05)  # Reduce horizontal space between subplots
    _finish_plot(fig, fn)




def plot_gt_demographics(df_gt_stats, attribute, fn=None, **kwargs):
    from postprocessing import bias

    figsize = kwargs.get('figsize', (10, 5))

    df_c = bias.get_data_by_attribute_and_metric(df_gt_stats, attribute, 'counts')
    df_f = bias.get_data_by_attribute_and_metric(df_gt_stats, attribute, 'fractions')

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True, sharex=False)

    category_order = constants.GENDER_LIST if attribute == constants.DEMOGRAPHIC_ATTRIBUTE_GENDER else constants.ETHNICITY_LIST
    category_dict = constants.GENDER_COLOR_DICT if attribute == constants.DEMOGRAPHIC_ATTRIBUTE_GENDER else constants.ETHNICITY_COLOR_DICT
    colors = [category_dict[c] for c in category_order]
    
    # counts
    ax = axes[0]
    ax = df_c.plot.barh(stacked=True,  color=colors, ax=ax, legend=False)   
    ax.set_ylabel('')

    # Customize specific labels in the legend
    handles, labels = ax.get_legend_handles_labels()
    if attribute == constants.DEMOGRAPHIC_ATTRIBUTE_GENDER:
        custom_labels = {
            "Female": "Female",
            "Male": "Male",
            "Unisex": "Neutral", #"Non-binary",
            "Unknown": "Unknown",
        }
    else:
        custom_labels = {
        "Black or African American": "Black",
        "Asian": "Asian",
        "White": "White",
        "Hispanic or Latino": "Latino",
        "Unknown": "Unknown",
    }
    new_labels = [custom_labels.get(label, label) for label in labels]
    ax.legend(handles, new_labels)


    # fractions
    ax = axes[1]
    ax = df_f.plot.barh(stacked=True,  color=colors, ax=ax, legend=False)
    ax.set_ylabel('')

    # Make specific ytick labels bold
    bold_labels = ["Physics Education Research", "Condensed Matter & Materials Physics"]  # Specify labels to make bold
    for label in axes[0].get_yticklabels():
        if label.get_text() in bold_labels:
            label.set_fontweight("bold")  # Make the label bold


    # Layout adjustments
    _finish_plot(fig, fn)


def plot_temperature_consistency(df, fn=None, **kwargs):
    ncols = kwargs.get('ncols', 6)  
    width = kwargs.get('width', 3.)
    height = kwargs.get('height', 2.)
    df_best_temperature = kwargs.get('df_best_temperature', None)
    df_factuality = kwargs.get('df_factuality', None)
    
    df = df.copy()
    groups = df.result_valid_flag.unique().categories

    # prepare plotting
    models = df['model'].unique()
    n_models = len(models)

    nrows = int(np.ceil(n_models / ncols))
    width = width * ncols
    height = height * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), sharex=True, sharey=True)
    
    if n_models == 1:
        axes = [axes]  # make iterable

    for row in range(nrows):
        for col in range(ncols):
            idx = row * ncols + col
            if idx >= n_models:
                fig.delaxes(axes[row, col])  # remove unused axes

            ax = axes[row, col]
            model = models[idx]
            sub = df[df['model'] == model]
            ax.set_title(model)

            # pivot to temperature × flag (values: normalized_counts), sort temperatures numerically
            pivot = (
                sub.pivot_table(
                    index='temperature',
                    columns='result_valid_flag',
                    values='normalized_counts',
                    aggfunc='sum',
                    fill_value=0.0,
                    observed=False
                )
                .reindex(columns=groups)
                .sort_index(key=lambda s: pd.to_numeric(s, errors='coerce'))  # ensures numeric order
            )

            # reorder columns
            order_cols = [c for c in constants.EXPERIMENT_OUTPUTS_ORDER if c in pivot.columns]
            pivot = pivot.reindex(columns=order_cols)

            x = np.arange(len(pivot.index))
            bottoms = np.zeros(len(x), dtype=float)
            xtick_labels = pivot.index.tolist()

            # draw stacked bars
            bar_width = 0.8
            for flag in groups:
                vals = pivot[flag].to_numpy() if flag in pivot.columns else np.zeros(len(x))
                ax.bar(x, vals, bottom=bottoms, width=bar_width, color=constants.EXPERIMENT_OUTPUT_COLORS[flag], edgecolor='none', label=flag)
                bottoms += vals

            if df_factuality is not None:
                # overlay factuality line
                df_fact_model = df_factuality.groupby(['model','temperature'])[['mean','std']].mean().reset_index().query("model==@model").copy()
                df_fact_model = df_fact_model.sort_values(by='temperature', key=lambda s: pd.to_numeric(s, errors='coerce'))
                # ax.plot(np.arange(len(df_fact_model)), df_fact_model['mean'], 
                #         color='white', marker='o', linestyle='-', zorder=10e10)
                
                ax.errorbar(np.arange(len(df_fact_model)), 
                            df_fact_model['mean'], 
                            yerr = df_fact_model['std'], 
                            fmt='o-', capsize=5,
                            color='lightgray',
                            label = 'Factuality',
                            zorder=10e10)
        
            if df_best_temperature is not None:
                tmp = df_best_temperature.query("model==@model")
                # ax.scatter([xtick_labels.index(tmp.iloc[0]['temperature'])], [tmp.iloc[0]['mean']], marker='D', 
                #            color='yellow', s=100, zorder=10e100)
                # ax.set_ylim(df_best_temperature['mean'].min(), df_best_temperature['mean'].max()   )
                rect = plt.Rectangle((xtick_labels.index(tmp.iloc[0]['temperature']) - bar_width/2, 0.0), 
                                        bar_width, 
                                        1.0,
                                        ls='-',
                                        fill=False, edgecolor='black', linewidth=4.0)
                ax.add_patch(rect)
            else:
                # choose a single best temperature: max 'a', ties → lowest temperature
                valid_series = pivot[constants.EXPERIMENT_OUTPUT_VALID] if constants.EXPERIMENT_OUTPUT_VALID in pivot.columns else pd.Series(0.0, index=pivot.index)
                max_valid = valid_series.max()
                if pd.isna(max_valid):
                    best_idx = None
                else:
                    best_idx = valid_series[valid_series == max_valid].index[0]  # first after numeric sort = lowest temp
        
                # single rectangle around the chosen temperature, height = 1.0
                if best_idx is not None:
                    j = np.where(pivot.index == best_idx)[0][0]  # bar position
                    rect = plt.Rectangle((x[j] - bar_width/2, 0.0), 
                                        bar_width, 
                                        1.0,
                                        ls='-',
                                        fill=False, edgecolor='black', linewidth=4.0)
                    ax.add_patch(rect)

            # y-limit to ensure the rectangle is fully visible up to 1.0
            ymax = max(1.0, float(bottoms.max()))
            ax.set_ylim(0.0, ymax+0.05)

    # cosmetics
    for ax in axes[-1,:]:
        ax.set_xlabel("temperature")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=0)
        # ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.6)

    # put a single legend at the top
    handles, labels = axes[0,0].get_legend_handles_labels()
    handles_ordered = []
    labels_ordered = []
    for l in constants.EXPERIMENT_OUTPUTS_ORDER:
        if l in labels:
            idx = labels.index(l)
            handles_ordered.append(handles[idx])
            labels_ordered.append(labels[idx])

    plus = int(df_factuality is not None)
    fig.legend(handles_ordered, labels_ordered, title="result_valid_flag", ncol=len(groups)+plus, loc='upper center', bbox_to_anchor=(0.5, 1.04))
    
    # final
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.subplots_adjust(hspace=0.4, wspace=0.05)

    # save
    if fn is not None:
        plt.savefig(fn, dpi=constants.FIG_DPI, bbox_inches='tight')

    # close
    plt.show()
    plt.close()



def plot_temperature_factuality_per_task(df, fn=None, **kwargs):
    ncols = kwargs.get('ncols', 6)
    width = kwargs.get('width', 3.)
    height = kwargs.get('height', 2.)

    nmodels = df['model'].nunique()
    nrows = int(np.ceil(nmodels / ncols))
    width = width * ncols
    height = height * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), sharex=True, sharey=True)
    groups = df.task_name.unique().categories

    for idx, model in enumerate(df['model'].unique()):
        for task_name in constants.EXPERIMENT_TASKS:

            col = idx % ncols
            row = idx // ncols
            ax = axes[row, col]

            df_subplot = df.query("model == @model and task_name == @task_name").copy()

            ax.set_title(model)
            ax.errorbar(df_subplot['temperature'], df_subplot['mean'], yerr=df_subplot['std'], fmt='o-', capsize=5, label=task_name)
            ax.set_ylim(0, 1)
            ax.grid(linestyle=':', linewidth=0.6, alpha=0.6)
            
    # cosmetics
    for ax in axes[-1,:]:
        ax.set_xlabel("temperature")
        ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.6)

    for ax in axes[:,0]:
        ax.set_ylabel("factuality")

    # put a single legend at the top
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, title="task_name", ncol=len(groups), loc='upper center', bbox_to_anchor=(0.5, 1.05))

    # final
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.subplots_adjust(hspace=0.4, wspace=0.05)

    # save
    if fn is not None:
        plt.savefig(fn, dpi=constants.FIG_DPI, bbox_inches='tight')

    # close
    plt.show()
    plt.close()

def plot_temperature_factuality_per_model(df, fn=None, **kwargs):
    ncols = kwargs.get('ncols', 6)  
    width = kwargs.get('width', 3.)
    height = kwargs.get('height', 2.)

    nmodels = df['model'].nunique()
    nrows = int(np.ceil(nmodels / ncols))
    width = width * ncols
    height = height * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), sharex=True, sharey=True)

    df_grouped = df.groupby(['model','temperature'])[['mean','std']].mean().reset_index()

    for idx, model in enumerate(df_grouped['model'].unique()):
        
        col = idx % ncols
        row = idx // ncols
        ax = axes[row, col]

        df_subplot = df_grouped.query("model == @model").copy()

        ax.set_title(model)
        ax.errorbar(df_subplot['temperature'], df_subplot['mean'], yerr=df_subplot['std'], fmt='o-', capsize=5)
        
        ax.set_ylim(0, 1)
        ax.grid(linestyle=':', linewidth=0.6, alpha=0.6)

    # cosmetics
    for ax in axes[-1,:]:
        ax.set_xlabel("temperature")
        ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.6)

    for ax in axes[:,0]:
        ax.set_ylabel("factuality")
        
    # final
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.subplots_adjust(hspace=0.4, wspace=0.05)

    # save
    if fn is not None:
        plt.savefig(fn, dpi=constants.FIG_DPI, bbox_inches='tight')
    
    # close
    plt.show()
    plt.close()



def plot_temperature_by_size(df, fn=None, **kwargs):
    ngroups = len(constants.LLMS_SIZE_CATEGORIES.keys())

    ncols = 1
    nrows = 1
    width = 5. * ncols
    height = 3 * nrows

    fig, ax = plt.subplots(nrows, ncols, figsize=(width, height), sharex=True, sharey=True)

    for id, (group_size, llms) in enumerate(constants.LLMS_SIZE_CATEGORIES.items()):

        df_subset = df.query("model in @llms").copy()
        color = constants.LLMS_SIZE_COLORS.get(group_size, None)

        # main: aggregate over models in the group
        df_grouped = df_subset.groupby(['temperature'])[['mean','std']].mean().reset_index()
        ax.errorbar(df_grouped['temperature'], df_grouped['mean'], yerr=df_grouped['std'], fmt='o-', 
                    capsize=5, label=group_size, color=color, linewidth=2.0)

        # # each model
        # for model, df_subplot in df_subset.groupby('model'):
        #     ax.errorbar(df_subplot['temperature'], df_subplot['mean'], yerr=df_subplot['std'], fmt='o-', capsize=5, label=model, alpha=0.2, zorder=0)
            
        # legend
        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles, labels, title="model", loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1)

    ax.legend(ncol=2)
    ax.set_ylabel("factuality")
        
    # settings
    ax.set_ylim(-0.05, 1.05)
    ax.grid(linestyle=':', linewidth=0.6, alpha=0.6)
    ax.set_xlabel("temperature")
    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.6)

    # final
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.subplots_adjust(hspace=0.4, wspace=0.05)

    # save
    if fn is not None:
        plt.savefig(fn, dpi=constants.FIG_DPI, bbox_inches='tight')
        
    # close
    plt.show()
    plt.close()



def plot_temperature_vs_bias(df_authors, df_all_authors_demographics, cat_col='gender', parity=True, fn=None, **kwargs):
    from postprocessing import bias

    # data
    all_colors = constants.GENDER_COLOR_DICT if cat_col == 'gender' else constants.ETHNICITY_COLOR_DICT if cat_col == 'ethnicity' else None
    all_values = constants.GENDER_LIST if cat_col == 'gender' else constants.ETHNICITY_LIST if cat_col == 'ethnicity' else None
    df_model_demographic, df_task_demographic = bias.get_mean_percentages(df_authors, cat_col, {cat_col:all_values})
    
    # setup
    nmodels = df_model_demographic['model'].nunique()
    ncols = 6
    nrows = int(np.ceil(nmodels / ncols))
    width = 3. * ncols
    height = 3. * nrows

    # baseliines
    g_baselines = df_all_authors_demographics.groupby(cat_col).size() / df_all_authors_demographics.shape[0]

    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), sharex=True, sharey=True)
    groups = []

    for group, df_data in df_model_demographic.groupby(cat_col, observed=False):

        gcolor = all_colors[group]
        groups.append(group)

        for idx, model in enumerate(df_data['model'].unique()):
            
            col = idx % ncols
            row = idx // ncols
            ax = axes[row, col]

            df_subplot = df_data.query("model == @model").copy()

            ax.set_title(model)

            ax.errorbar(df_subplot['temperature'],  df_subplot['mean'] - g_baselines[group] if parity else df_subplot['mean'], 
                        yerr=df_subplot['std'], fmt='o-', capsize=5, label=group, color=gcolor)
            ax.hlines(0 if parity else g_baselines[group], 
                      xmin=df_subplot['temperature'].min(), xmax=df_subplot['temperature'].max(), color='black', linestyle='--', linewidth=1., alpha=0.7, zorder=10e100)

            ax.set_ylim(-0.5, 0.75)
            ax.grid(linestyle=':', linewidth=0.6, alpha=0.6)

        # cosmetics
        for ax in axes[-1,:]:
            ax.set_xlabel("temperature")
            ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.6)

    # put a single legend at the top
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, title=cat_col, ncol=len(groups), loc='upper center', bbox_to_anchor=(0.5, 1.02))

    # final
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.subplots_adjust(hspace=0.4, wspace=0.05)

    # save
    if fn is not None:
        plt.savefig(fn, dpi=constants.FIG_DPI, bbox_inches='tight')
    
    # final
    plt.show()
    plt.close()



def plot_temperature_vs_bias_by_size(df_authors, df_all_authors_demographics, cat_col='gender', parity=True, fn=None, **kwargs):
    from postprocessing import bias

    # data
    all_values = constants.GENDER_LIST if cat_col == 'gender' else constants.ETHNICITY_LIST if cat_col == 'ethnicity' else None
    df_model_demographic, df_task_demographic = bias.get_mean_percentages(df_authors, cat_col, {cat_col:all_values}, by_size=True)
    groups = []

    # setup
    maxcols = min(6, len(all_values))
    ncols = len(all_values)
    nrows = int(np.ceil(maxcols / ncols))
    width = 3. * ncols
    height = 3. * nrows

    # baseliines
    g_baselines = df_all_authors_demographics.groupby(cat_col).size() / df_all_authors_demographics.shape[0]

    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), sharex=True, sharey=True)

    for idx, (group, df_data) in enumerate(df_model_demographic.groupby(cat_col, observed=False)):

        # for each demographic group, plot each model_size in a subplot
        col = idx % ncols
        row = idx // ncols
        ax = axes[row, col] if nrows > 1 else axes[col]
        ax.set_title(group)

        for size_class in constants.LLMS_SIZE_ORDER:
            df_subplot = df_data.query("size_class == @size_class").copy()
            gcolor = constants.LLMS_SIZE_COLORS.get(size_class, None)
            ax.errorbar(df_subplot['temperature'],  
                        df_subplot['mean'] - g_baselines[group] if parity else df_subplot['mean'], 
                        yerr=df_subplot['std'], fmt='o-', capsize=5, label=size_class, color=gcolor)
            ax.hlines(0 if parity else g_baselines[group], 
                      xmin=df_subplot['temperature'].min(), xmax=df_subplot['temperature'].max(), 
                      color='black', linestyle='--', linewidth=1., alpha=0.7, zorder=10e100)
            groups.append(size_class)

        # cosmetics
        if row == nrows - 1:
            ax.set_xlabel("temperature")
            ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.6)

    # put a single legend at the top
    handles, labels = ax.get_legend_handles_labels()
    labels = [constants.LLMS_SIZE_SHORT_NAMES[l] for l in labels if l in groups]
    fig.legend(handles, labels, title='model_size', ncol=len(labels), loc='upper center', bbox_to_anchor=(0.5, 1.2))

    # final
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.subplots_adjust(hspace=0.4, wspace=0.05)

    # save
    if fn is not None:
        plt.savefig(fn, dpi=constants.FIG_DPI, bbox_inches='tight')
    
    # final
    plt.show()
    plt.close()



def boxpanel(ax, df_attempt, df_group, group_col, ycol, order=None, continuous=True, **kwargs):
    
    data = df_attempt.copy() if continuous else df_group.copy()

    # Determine category order
    cats = order if order is not None else list(data[group_col].dropna().unique())
    
    if not continuous:
        data = data.set_index(group_col).loc[cats].reset_index()

        x = np.arange(data.shape[0])

        yerr = np.vstack([
            data["mean"] - data["ci_low"],
            data["ci_high"] - data["mean"]
        ])

        ax.bar(
            x,
            data["mean"],
            width=0.6,
            color="white",
            edgecolor="black",
            tick_label=cats
        )

        ax.errorbar(
            x,
            data["mean"],
            yerr=yerr,
            fmt="none",
            color="black",
            capsize=3
        )

        ax.set_xticks(x)
        ax.set_xticklabels(data[group_col])
    
    else:
        data = [data.loc[data[group_col] == c, ycol].dropna().to_numpy() for c in cats]
        bp = ax.boxplot(
            data,
            tick_labels=cats,
            patch_artist=True, # fill with color
            showfliers=True, # show outliers
            widths=0.6, # width of the box  
            whis=(5, 95) # whiskers at 5% and 95% of the data
        )

        # set means
        for i, y in enumerate(data, start=1):
            ax.plot(i, np.mean(y), marker="o", color="tab:red", markersize=4, zorder=3)

        # Minimal styling; rasterize only the outlier dots for fast rendering/export
        for box in bp["boxes"]:
            box.set(facecolor="white", edgecolor="black")

        # set lines (lines, caps, medians)
        for k in ("whiskers", "caps", "medians"):
            for line in bp[k]:
                line.set(color="black", linewidth=1.)

        for line in bp["whiskers"] + bp["caps"]:
            line.set_linewidth(0.5)

        # set fliers (outliers)
        for flier in bp["fliers"]:
            flier.set(marker="o", markersize=1, alpha=0.1, rasterized=True, color="lightgray")

    group_name = group_col.replace('_', ' ').capitalize()

    # title
    show_title = kwargs.pop('show_title', False)

    if show_title:
        ax.set_title(group_name)
    else:
        ax.set_title(None)

    # set xlabel
    show_xlabel = kwargs.pop('show_xlabel', False)
    if show_xlabel:
        # set xlabel
        xlabel = kwargs.get('xlabel', None)
        xlabel = group_name if xlabel is None else xlabel
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(None)

     # set ylabel
    ylabel = kwargs.get('ylabel', None)
    if ylabel is not None:
        if 'diversity_' in ylabel or 'factuality_' in ylabel or 'parity_' in ylabel:
            k = f"{ylabel.split('_')[0]}_"
            v = ylabel.split(k)[-1].replace('prominence_','')

            k = k.replace('diversity','div.').replace('factuality','fact.') if 'diversity' in ylabel or 'factuality' in ylabel else k

            ylabel = k.replace('_','').title() + u"$_{" + v + "}$"
        else:
            ylabel = ylabel.replace('_pct','').title() # u"$_{" + "ratio" + "}$"
        ax.set_ylabel(ylabel)
        

    # xticks
    show_xticks = kwargs.pop('show_xticks', False)
    if not show_xticks:
        ax.set_xticklabels([])

def plot_infrastructural_conditions(per_attempt, fn=None, continuous=True, **kwargs):

    figsize = kwargs.pop('figsize', (10, 2.5))
    aggregator_kwargs = kwargs.pop('aggregator_kwargs', {})

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True, sharex=False)

    ycol = 'metric'
    
    # access
    ax = axes[0]
    key = 'model_access'
    order = ["open", "proprietary"]
    per_group = aggregators.aggregate_per_group(per_attempt, key, **aggregator_kwargs)
    boxpanel(ax, per_attempt, per_group, key, ycol, order=order, continuous=continuous, **kwargs)

    # model size
    ax = axes[1]
    key = 'model_size'
    order = [c for c in ['S', 'S (P)', 'M', 'M (P)', 'L', 'XL'] if c in per_attempt.model_size.unique()]
    per_group = aggregators.aggregate_per_group(per_attempt, key, **aggregator_kwargs)
    boxpanel(ax, per_attempt, per_group, key, ycol, order=order, continuous=continuous, **kwargs)
    ax.set_ylabel(None)

    # model class
    ax = axes[2]
    key = 'model_class'
    order = ['non-reasoning', 'reasoning']
    per_group = aggregators.aggregate_per_group(per_attempt, key, **aggregator_kwargs)
    boxpanel(ax, per_attempt, per_group, key, ycol, order=order, continuous=continuous, **kwargs)
    ax.set_ylabel(None)
    
    ylim = kwargs.pop('ylim', None)
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    
    if fn is not None:
        fig.savefig(fn, dpi=constants.FIG_DPI, bbox_inches='tight')

    plt.show()
    plt.close()

def pointplot(ax, df_group, group_col, hue_order, x_order, **kwargs):

    data = df_group.copy()
    xvar = group_col[0]
    gvar = group_col[1]

    # Load colormaps
    tab20 = plt.get_cmap("tab20")
    tab20c = plt.get_cmap("tab20c")
    colors = {'model_access': [tab20(0), tab20(1)][::-1],
              'model_size': [tab20c(i) for i in range(8,12)][::-1],
              'model_class': [tab20(2), tab20(3)][::-1]}

    show_legend = kwargs.pop('show_legend', False)

    # Determine category order (hue)
    gcats = hue_order if hue_order is not None else list(data[gvar].dropna().unique())
    xcats = x_order if x_order is not None else list(data[xvar].dropna().unique())

    for i, hue in enumerate(gcats):
        color = colors[gvar][i]
        df_hue = data.query(f"{gvar} == @hue").copy()
        df_hue = df_hue.set_index(xvar).loc[xcats].reset_index()
        x = np.arange(df_hue.shape[0])

        yerr = np.vstack([
            df_hue["mean"] - df_hue["ci_low"],
            df_hue["ci_high"] - df_hue["mean"]
        ])

        ax.plot(x, df_hue["mean"], marker="o", markersize=4, zorder=3, label=hue, color=color)

        ax.errorbar(
            x,
            df_hue["mean"],
            yerr=yerr,
            fmt="none",
            color=color,
            capsize=3,
            linewidth=0.5
        )

        ax.set_xticks(x)
        ax.set_xticklabels(df_hue['temperature'])

        if show_legend:
            legend_kwargs = kwargs.get('legend_kwargs', {})
            legend_kwargs['ncol'] = 2 if len(gcats) > 2 else 1
            ax.legend(**legend_kwargs)


    group_name = gvar.replace('_', ' ').capitalize()

    # title
    show_title = kwargs.pop('show_title', False)

    if show_title:
        ax.set_title(group_name)
    else:
        ax.set_title(None)

    # set xlabel
    show_xlabel = kwargs.pop('show_xlabel', False)
    if show_xlabel:
        # set xlabel
        ax.set_xlabel(xvar)
    else:
        ax.set_xlabel(None)

    # set ylabel
    ylabel = kwargs.get('ylabel', None)
    if ylabel is not None:
        if 'diversity_' in ylabel or 'factuality_' in ylabel or 'parity_' in ylabel:
            k = f"{ylabel.split('_')[0]}_"
            v = ylabel.split(k)[-1].replace('prominence_','')

            k = k.replace('diversity','div.').replace('factuality','fact.') if 'diversity' in ylabel or 'factuality' in ylabel else k

            ylabel = k.replace('_','').title() + u"$_{" + v + "}$"
        else:
            ylabel = ylabel.replace('_pct','').title() # u"$_{" + "ratio" + "}$"
        ax.set_ylabel(ylabel)

        
    # xticks
    show_xticks = kwargs.pop('show_xticks', False)
    if not show_xticks:
        ax.set_xticklabels([])


def barplot_diff(ax, df_group, df_group_intervention, group_col, hue_order, x_order, **kwargs):

    data = df_group.copy()
    data_interventions = df_group_intervention.copy()

    xvar = group_col[0]
    gvar = group_col[1]
    width = kwargs.get('width_bar', 2)

    # Load colormaps
    tab20 = plt.get_cmap("tab20")
    tab20c = plt.get_cmap("tab20c")
    colors = {'model_access': [tab20(0), tab20(1)][::-1],
              'model_size': [tab20c(i) for i in range(8,12)][::-1],
              'model_class': [tab20(2), tab20(3)][::-1]}

    show_legend = kwargs.pop('show_legend', False)

    # Determine category order (hue)
    gcats = hue_order if hue_order is not None else list(data[gvar].dropna().unique())
    xcats = x_order if x_order is not None else list(data_interventions[xvar].dropna().unique())
    x = np.arange(len(xcats))
    k = len(gcats)

    ax.axhline(0, color='black', linestyle='--', linewidth=0.5, zorder=10e100)
    ax.set_xticks(x)

    for i, hue in enumerate(gcats):
        color = colors[gvar][i]

        df_hue = data.query(f"{gvar} == @hue").copy()
        df_hue = pd.concat([df_hue]*len(xcats), axis=0).reset_index(drop=True)
        df_hue.loc[:,xvar] = xcats
        df_hue = df_hue.set_index(xvar)
        

        df_hue_interventions = data_interventions.query(f"{gvar} == @hue").reset_index(drop=True).set_index(xvar).copy()
        
        df_hue_diff = df_hue_interventions.drop(columns=[gvar]) - df_hue.drop(columns=[gvar])

        # yerr = np.vstack([
        #     df_hue["mean"] - df_hue["ci_low"],
        #     df_hue["ci_high"] - df_hue["mean"]
        # ])

        ax.bar(x + (i - k/2)*width + width/2, df_hue_diff["mean"], width=width, label=hue, color=color)

        # ax.errorbar(
        #     x,
        #     df_hue["mean"],
        #     yerr=yerr,
        #     fmt="none",
        #     color=color,
        #     capsize=3,
        #     linewidth=0.5
        # )

        
    if 'xticklabels_map' in kwargs:
        xticklabels_map = kwargs.get('xticklabels_map', {})
        xcat_labels = [xticklabels_map[x] for x in xcats]
    else:
        xcat_labels = xcats
    ax.set_xticklabels(xcat_labels)

    if show_legend:
        legend_kwargs = kwargs.get('legend_kwargs', {})
        legend_kwargs['ncol'] = 1 if len(gcats) <= 2 else 2
        ax.legend(**legend_kwargs)

    group_name = gvar.replace('_', ' ').capitalize()

    # title
    show_title = kwargs.get('show_title', False)

    if show_title:
        ax.set_title(group_name)
    else:
        ax.set_title(None)

    # set xlabel
    show_xlabel = kwargs.get('show_xlabel', False)
    if show_xlabel:
        # set xlabel
        xlabel = kwargs.get('xlabel', xvar)
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(None)

    # set ylabel
    ylabel = kwargs.get('ylabel', None)
    if ylabel is not None:
        if 'diversity_' in ylabel or 'factuality_' in ylabel or 'parity_' in ylabel:
            k = f"{ylabel.split('_')[0]}_"
            v = ylabel.split(k)[-1].replace('prominence_','')

            k = k.replace('diversity','div.').replace('factuality','fact.') if 'diversity' in ylabel or 'factuality' in ylabel else k

            ylabel = k.replace('_','').title() + u"$_{" + v + "}$"
        else:
            ylabel = ylabel.replace('_pct','').title() # u"$_{" + "ratio" + "}$"
        ax.set_ylabel(ylabel)

    # xticks
    show_xticks = kwargs.get('show_xticks', False)
    if not show_xticks:
        ax.set_xticklabels([])



def plot_infrastructural_conditions_comparison_intervention(per_attempt, per_attempt_intervention, intervention_col, fn=None, **kwargs):

    figsize = kwargs.pop('figsize', (10, 2.5))
    aggregator_kwargs = kwargs.pop('aggregator_kwargs', {})

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True, sharex=False)

    x_order = per_attempt_intervention[intervention_col].unique()

    # access
    ax = axes[0]
    key = 'model_access'
    key_intervention = [intervention_col,key]
    order = ["open", "proprietary"]
    per_group = aggregators.aggregate_per_group(per_attempt, key, **aggregator_kwargs)
    per_group_intervention = aggregators.aggregate_per_group(per_attempt_intervention, key_intervention, **aggregator_kwargs)
    barplot_diff(ax, per_group, per_group_intervention, key_intervention, hue_order=order, x_order=x_order, **kwargs)

    # model size
    ax = axes[1]
    key = 'model_size'
    key_intervention = [intervention_col,key]
    order = [c for c in ['S', 'S (P)', 'M', 'M (P)', 'L', 'XL'] if c in per_attempt.model_size.unique()]
    per_group = aggregators.aggregate_per_group(per_attempt, key, **aggregator_kwargs)
    per_group_intervention = aggregators.aggregate_per_group(per_attempt_intervention, key_intervention, **aggregator_kwargs)
    barplot_diff(ax, per_group, per_group_intervention, key_intervention, hue_order=order, x_order=x_order, **kwargs)
    ax.set_ylabel(None)

    # model class
    ax = axes[2]
    key = 'model_class'
    key_intervention = [intervention_col,key]
    order = ['non-reasoning', 'reasoning']
    per_group = aggregators.aggregate_per_group(per_attempt, key, **aggregator_kwargs)
    per_group_intervention = aggregators.aggregate_per_group(per_attempt_intervention, key_intervention, **aggregator_kwargs)
    barplot_diff(ax, per_group, per_group_intervention, key_intervention, hue_order=order, x_order=x_order, **kwargs)
    ax.set_ylabel(None)
    
    ylim = kwargs.pop('ylim', None)
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, top=0.80)

    if fn is not None:
        fig.savefig(fn, dpi=constants.FIG_DPI, bbox_inches='tight')

    plt.show()
    plt.close()

def plot_infrastructural_conditions_by_intervention(per_attempt, xvar, fn=None, **kwargs):

    figsize = kwargs.pop('figsize', (10, 2.5))
    aggregator_kwargs = kwargs.pop('aggregator_kwargs', {})

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True, sharex=False)

    x_order = per_attempt[xvar].unique()

    # access
    ax = axes[0]
    key = 'model_access'
    key = [xvar,key]
    order = ["open", "proprietary"]
    per_group = aggregators.aggregate_per_group(per_attempt, key, **aggregator_kwargs)
    pointplot(ax, per_group, key, hue_order=order, x_order=x_order, **kwargs)

    # model size
    ax = axes[1]
    key = 'model_size'
    key = [xvar,key]
    order = [c for c in ['S', 'S (P)', 'M', 'M (P)', 'L', 'XL'] if c in per_attempt.model_size.unique()]
    per_group = aggregators.aggregate_per_group(per_attempt, key, **aggregator_kwargs)
    pointplot(ax, per_group, key, hue_order=order, x_order=x_order, **kwargs)
    ax.set_ylabel(None)

    # model class
    ax = axes[2]
    key = 'model_class'
    key = [xvar,key]
    order = ['non-reasoning', 'reasoning']
    per_group = aggregators.aggregate_per_group(per_attempt, key, **aggregator_kwargs)
    pointplot(ax, per_group, key, hue_order=order, x_order=x_order, **kwargs)
    ax.set_ylabel(None)
    
    ylim = kwargs.pop('ylim', None)
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    
    if fn is not None:
        fig.savefig(fn, dpi=constants.FIG_DPI, bbox_inches='tight')

    plt.show()
    plt.close()


def plot_metric_bars_by_groups(df, x_col='task_name', hue_col="model", metric_col="mean", fn=None, **kwargs):
    """
    Plot grouped bar chart:
    - x-axis: metric
    - y-axis: mean
    - hue: model (two bars per metric)
    """

    figsize = kwargs.pop('figsize', (10, 2.5))
    bar_width = kwargs.pop('bar_width', 0.35)
    show_legend = kwargs.pop('show_legend', False)
    x_order = kwargs.pop('x_order', df[x_col].unique())

    fig, ax = plt.subplots(1, 1, figsize=figsize, sharey=True, sharex=False)

    hue_groups = df[hue_col].unique()

    x = np.arange(len(x_order))

    tab20 = plt.get_cmap("tab20")
    colors = [tab20(6), tab20(7)][::-1]

    for i, hue in enumerate(hue_groups):
        values = (
            df[df[hue_col] == hue]
            .set_index(x_col)
            .loc[x_order, metric_col]
            .values
        )
        
        ax.bar(
            x + (i - (len(hue_groups)-1)/2) * bar_width,
            values,
            width=bar_width,
            label=hue,
            color=colors[i]
        )

    ax.axhline(0, color='black', linestyle='--', linewidth=0.5, zorder=10e100)
    ax.set_xticks(x)

    if 'xticklabels_map' in kwargs:
        xticklabels_map = kwargs.get('xticklabels_map', {})
        xcat_labels = [xticklabels_map[x] for x in x_order]
    else:
        xcat_labels = x_order
    ax.set_xticklabels(xcat_labels)

    if show_legend:
        legend_kwargs = kwargs.get('legend_kwargs', {})
        legend_kwargs['ncol'] = 1 if len(hue_groups) <= 2 else 2
        ax.legend(**legend_kwargs)

    ylim = kwargs.pop('ylim', None)
    if ylim is not None:
        ax.set_ylim(ylim)

    # set xlabel
    show_xlabel = kwargs.get('show_xlabel', False)
    if show_xlabel:
        # set xlabel
        xlabel = kwargs.get('xlabel', x_col)
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(None)

    
    # set ylabel
    ylabel = kwargs.get('ylabel', None)
    if ylabel is not None:
        if 'diversity_' in ylabel or 'factuality_' in ylabel or 'parity_' in ylabel:
            k = f"{ylabel.split('_')[0]}_"
            v = ylabel.split(k)[-1].replace('prominence_','')

            k = k.replace('diversity','div.').replace('factuality','fact.') if 'diversity' in ylabel or 'factuality' in ylabel else k

            ylabel = k.replace('_','').title() + u"$_{" + v + "}$"
        else:
            ylabel = ylabel.replace('_pct','').title() # u"$_{" + "ratio" + "}$"
        ax.set_ylabel(ylabel)


    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    
    if fn is not None:
        fig.savefig(fn, dpi=constants.FIG_DPI, bbox_inches='tight')

    plt.show()
    plt.close()




