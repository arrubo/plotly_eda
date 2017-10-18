import pandas as pd
# from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

import numpy as np
# import math
import scipy

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as mpatches
# import matplotlib.mlab as mlab

import seaborn as sns

from IPython.display import HTML
import colorlover as cl

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.figure_factory as ff
init_notebook_mode(connected=True)


def describe_info(dataframe):
    df_describe = dataframe.describe()
    print("DataFrame shape:")
    print("  Rows:", dataframe.shape[0], "( from", min(dataframe.index), "to", max(dataframe.index), ")")
    print("  Columns:", dataframe.shape[1])
    for col in dataframe.columns:
        df_describe.loc['count', col] = dataframe[dataframe.notnull()].count()[col]
        df_describe.loc['type', col] = dataframe[col].dtype
        if df_describe.loc['type', col] in ["float64", "int64"]:
            df_describe.loc['median', col] = dataframe[col].median()
            df_describe.loc['mad', col] = dataframe[col].mad()
            df_describe.loc['mode', col] = str(dataframe[col].mode()[0])
        else:
            df_describe.loc['mode', col] = dataframe[col].mode()[0]
    df_describe = df_describe.reindex(
        ["type", "count", "mean", "std", "median", "mad", "mode", "min", "25%", "50%", "75%", "max"])
    return df_describe


def describe_categorical(dataframe, field, **kwargs):
    valid_kwargs = ['category_map', 'totals', 'sorting']
    if 'help' in kwargs and kwargs['help'] == 'args':
        return print(valid_kwargs)
    for kwarg in kwargs:
        if kwarg not in valid_kwargs:
            raise Exception("'{}' is not a valid kwarg for this function. Did you spell something wrong?".format(kwarg))
    n_total = dataframe.shape[0]
    categories = sorted(dataframe[field].unique())
    output = pd.DataFrame([], index=categories, columns=['n', '%'])
    for category in categories:
        n_category = dataframe[dataframe[field] == category].shape[0]
        output.loc[category, 'n'] = n_category
        output.loc[category, '%'] = round(n_category * 100 / n_total, 2)
    if kwargs:
        if 'category_map' in kwargs:
            output = output.rename(index=kwargs['category_map'])
        if 'sorting' in kwargs:
            if kwargs['sorting'] == 'ascending':
                output = output.sort_values(by='n', ascending=True)
            if kwargs['sorting'] == 'descending':
                output = output.sort_values(by='n', ascending=False)
        if 'totals' in kwargs and kwargs['totals'] is True:
            output.loc['totals'] = output.apply(np.sum, axis='index')
    return output


# helpers for controlling colors, labels, colormaps, colorscales, ...
def to_new_scale(item, new_range, old_range=[]):
    new_max = max(new_range)
    new_min = min(new_range)
    item = np.array(list(item))
    if old_range == []:
        old_max = item.max()
        old_min = item.min()
    else:
        old_max = max(old_range)
        old_min = min(old_range)
    return (new_max - new_min) * (item - old_min) / (old_max - old_min) + new_min


def color_norm_to_rgb(norm_colors):
    return ["rgb({rgb[0]},{rgb[1]},{rgb[2]})".format(rgb=np.array(list(rgb_norm[:3]*255)).astype(int))
            for rgb_norm in list(norm_colors)]


# creates the handles for the plot legend
def set_handles(row):
    return mpatches.Patch(color=row['color_norm'], label=row['lab'])


# creates a dict gathering the basic setup for color mapping
def set_color_setup(data, **kwargs):
    # Color info
    # plotly     -> https://plot.ly/ipython-notebooks/color-scales/
    # matplotlib -> https://matplotlib.org/xkcd/examples/color/colormaps_reference.html
    valid_kwargs = ['colormap_name', 'norm_min', 'norm_max', 'matplotlib_style', 'plotly_style']
    if 'help' in kwargs and kwargs['help'] == 'args':
        return print(valid_kwargs)
    if not list(data):
        raise Exception("No 'data' has been provided. Make sure you provide the parameter to prepare the setup.")
    for kwarg in kwargs:
        if kwarg not in valid_kwargs:
            raise Exception("'{}' is not a valid kwarg for this function. Did you spell something wrong?".format(kwarg))
    if isinstance(data[0], str):  # categorical data
        norm_min = 0
        norm_max = len(data) - 1
    else:  # numerical data
        norm_min = min(data)
        norm_max = max(data)
    if kwargs:
        if 'norm_min' in kwargs:
            norm_min = kwargs['norm_min']
        if 'norm_max' in kwargs:
            norm_max = kwargs['norm_max']
        if 'colormap_name' not in kwargs:
            kwargs['colormap_name'] = 'seismic'
    color_setup = {
        'colormap': plt.get_cmap(kwargs['colormap_name']),
        'colormapNorm': mcolors.Normalize(vmin=norm_min, vmax=norm_max)}
    color_setup['scalarMap'] = cm.ScalarMappable(norm=color_setup['colormapNorm'], cmap=color_setup['colormap'])
    return color_setup


# creates a dataframe gathering some basics for plotting
def set_plot_setup(data, color_setup):
    if not list(data):
        raise Exception("No 'data' has been provided. Make sure you provide the parameter to prepare the setup.")
    color_setup_error = '''
    No 'color_setup' has been provided. Make sure you provide a variable to prepare the setup.
    Color_setup parameter (2 options):
     1 - list: color pallete, i.e. sns.color_palette(["lawngreen", "r","b"]) 
               [(0.48627450980392156, 0.9882352941176471, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
     2 - color_setup: output from set_color_setup'''
    if not color_setup:
        raise Exception(color_setup_error)
    if isinstance(color_setup, sns.palettes._ColorPalette):
        colors = color_setup
    elif isinstance(color_setup, dict):
        colors = list(color_setup['scalarMap'].to_rgba(data))
    else:
        raise Exception(color_setup_error)
    plot_setup = pd.DataFrame(index=range(0, len(data)),
                              columns=['lab', 'color_norm', 'color_rgb', 'handle'],
                              data={
                                'lab': list(map(str, list(data))),
                                'color_norm': colors,
                                'color_rgb': color_norm_to_rgb(colors)
        })
    plot_setup.handle = plot_setup.apply(lambda row: set_handles(row), axis=1)
    return plot_setup


def custom_colorscale(data, original_bounds=[-1, 1], colormap_name='seismic'):
    if not list(data): raise Exception(
        "No 'data' has been provided. Make sure you provide the parameter to prepare the setup.")
    norm = (np.array(range(21)) / 20).round(2)
    color_setup = set_color_setup(norm, colormap_name=colormap_name)
    plot_setup = set_plot_setup(norm, color_setup)
    converter = pd.DataFrame({'norm': norm, 'rgb': plot_setup.color_rgb})
    data_norm = to_new_scale(data, [0, 1], [-1, 1]).round(2)
    filter_selected = (converter.norm >= round(min(data_norm), 2)) & (converter.norm <= round(max(data_norm), 2))
    color_selected = converter[filter_selected].rgb
    norm_selected = to_new_scale(converter[filter_selected].norm, [0, 1], original_bounds).round(2)
    output = {
        'filter_selected': filter_selected,
        'color_selected': color_selected,
        'norm_selected': norm_selected,
        'colorscale': [list(colorscale) for colorscale in zip(norm_selected, color_selected)]
    }
    # Color info
    # plotly     -> https://plot.ly/ipython-notebooks/color-scales/
    # matplotlib -> https://matplotlib.org/xkcd/examples/color/colormaps_reference.html
    # import colorlover as cl
    # from IPython.display import HTML
    # HTML(cl.to_html(list(color_selected)))
    return output


def plot_pearsons(dataframe, columns_array=[], **kwargs):
    valid_kwargs = ['hover_text', 'plot_title', 'margin', 'height', 'width']
    if 'help' in kwargs and kwargs['help'] == 'args':
        return print(valid_kwargs)
    for kwarg in kwargs:
        if kwarg not in valid_kwargs:
            raise Exception("'{}' is not a valid kwarg for this function. Did you spell something wrong?".format(kwarg))
    if not isinstance(dataframe, pd.core.frame.DataFrame):
        raise Exception("No valid 'dataframe' has been provided. Make sure you provide a proper dataframe.")
    # columns_array
    if len(columns_array) > 0:
        df_pearsons = dataframe[columns_array]
    else:
        df_pearsons = dataframe
     # as type float
    for column in df_pearsons.columns:
        if not is_numeric_dtype(dataframe[column]):
            raise Exception("Column '{}' has to be float type,".format(column))
    correlation_matrix = df_pearsons.corr()
    np.fill_diagonal(correlation_matrix.values, 0)
    if kwargs and 'hover_text' in kwargs and kwargs['hover_text']:
        text = kwargs['hover_text']
    else:
        correlation_matrix_label = correlation_matrix.copy()
        for column in correlation_matrix_label.columns:
            correlation_matrix_label[column] = pd.cut(correlation_matrix_label[column], right=False,
                bins= [-1, -0.8, -0.6, -0.4, -0.2, -0.01, 0.01, 0.2, 0.4, 0.6, 0.8, 1],
                labels= ['very strong -', 'strong -', 'moderate -', 'weak -', 'very weak -', '', 'very weak +',
                          'weak +', 'moderate +', 'strong +', 'very strong +'])
        xy = correlation_matrix.apply(lambda c: "x: "+c.index+"<br>y: "+c.name+"<br>z: ")
        z = correlation_matrix.applymap("{:1.2f}<br>corr: ".format)
        text = xy + z + correlation_matrix_label.applymap(str)
        np.fill_diagonal(text.values, "")
    correlation_matrix_values = np.nan_to_num(correlation_matrix.values.flatten())
    lower_color = custom_colorscale(correlation_matrix_values)["colorscale"][0][1]
    middle_value = to_new_scale([0], [0, 1], [correlation_matrix_values.min(), correlation_matrix_values.max()])
    upper_color = custom_colorscale(correlation_matrix_values)["colorscale"][-1][1]
    trace = go.Heatmap(
        z=np.nan_to_num(correlation_matrix.values),
        x=columns_array,
        y=columns_array,
        text=text.values,
        hoverinfo="text",
        colorscale=[[0, lower_color],
                    [middle_value, 'rgb(255,255,255)'],
                    [1, upper_color]],
        colorbar={'tickvals': [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
                               0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
    )
    layout_json = {
        'margin': {
            't': 50,
            'r': 50,
            'b': 120,
            'l': 120}
    }
    if kwargs and 'height' in kwargs:
        layout_json['height'] = kwargs['height']
    if kwargs and 'width' in kwargs:
        layout_json['width'] = kwargs['width']
    if kwargs and 'margin' in kwargs:
        layout_json['margin'] = kwargs['margin']
    if kwargs and 'plot_title' in kwargs:
        layout_json['title'] = kwargs['plot_title']
    figure = dict(data=[trace], layout=layout_json)
    iplot(figure)


def plot_attribute(df, field, **kwargs):
    valid_kwargs = [
        # common
        'mode', 'plot_title', 'x_title', 'y_title', 'colors',
        'help', 'margin', 'height', 'width', 'x_showticklabels',
        # categorical
        'barmode', 'width', 'sorting', 'x_labels_map', 'text_hover',
        # numerical
        'start_bin', 'end_bin', 'width_bins', 'n_bins', 'verbose'
    ]
    if not kwargs:
        kwargs = {}
    # - help
    if 'help' in kwargs:
        return print(valid_kwargs)
    # - check field exists
    if field not in df.columns.values:
        raise Exception("'{}' is not a feature of the dataframe. Did you spell something wrong?".format(field))
    # if sorted(sorting) != df[field].unique()
    # if xlabels_map.keys() != df[field].unique()
    # - check valid kwargs
    for kwarg in kwargs:
        if kwarg not in valid_kwargs:
            raise Exception("'{}' is not a valid kwarg for this function. Did you spell something wrong?".format(kwarg))
    # layout
    layout = {
        'title': '',
        'xaxis': {
            'title': ''},
        'yaxis': {
            'title': ''}}
    if 'height' in kwargs:
        layout['height'] = kwargs['height']
    if 'width' in kwargs:
        layout['width'] = kwargs['width']
    if 'margin' in kwargs:
        layout['margin'] = kwargs['margin']
    if 'plot_title' in kwargs:
        layout['title'] = kwargs['plot_title']
    if 'x_title' in kwargs:
        layout['xaxis']['title'] = kwargs['x_title']
    if 'x_showticklabels' in kwargs:
        layout['xaxis']['showticklabels'] = kwargs['x_showticklabels']
    if 'y_title' in kwargs:
        layout['yaxis']['title'] = kwargs['y_title']
    # - mode : ['categorical,'numerical']
    # if not given, it will be guessed from the column data type
    if 'mode' not in kwargs:
        if isinstance(df[field].iloc[0], str):
            kwargs['mode'] = 'categorical'
        else:
            kwargs['mode'] = 'numerical'
    if kwargs['mode'] == 'categorical':
        # categorical
        # - barmode  : ['group','stack','relative']
        barmode = 'group'
        if 'barmode' in kwargs and kwargs['barmode']:
            barmode = kwargs['barmode']
        # sorting x axis
        dict_data = df[field].value_counts()
        keys = dict_data.keys()
        if 'sorting' in kwargs:
            sorting = kwargs['sorting']
            keys = np.array(sorting)
            dict_data = dict_data[keys]
        else:
            keys = keys.sort_values()
            sorting_index = keys.argsort()
            dict_data = dict_data[keys[sorting_index]]
        data = {
            'x': keys,
            'y': dict_data,
            'text': round(dict_data*100/sum(dict_data),1).apply(lambda c: "{} %".format(c)),
            'marker': {}
        }
        if 'colors' in kwargs and kwargs['colors']:
            data['marker'] = {'color': kwargs['colors']}
        # labeling x axis
        if 'x_labels_map' in kwargs:
            keys = [kwargs['x_labels_map'][key] for key in list(keys)]
            data['x'] = keys
        # customizing bin widths
        if 'width' in kwargs:
            data['width'] = kwargs['width']
        # customizing hover text
        if 'text_hover' in kwargs:
            data['text'] = kwargs['text_hover']
        return iplot(go.Figure(data=[go.Bar(data)], layout=layout))
    else:
        # numerical
        # - check n_bins and bin_width simultaneously
        if 'n_bins' in kwargs and 'bin_width' in kwargs:
            raise Exception("Arguments 'n_bins' and 'bin_width' cannot be used simultaneously.")
        # - check proper start_bin and end_bin
        if ('start_bin' in kwargs) and ('end_bin' in kwargs) and (kwargs['start_bin'] >= kwargs['end_bin']):
            raise Exception("Argument 'start_bin' has to be lower than 'end_bin'.")
        # - data
        data = {
            'x': df[field],
            'xbins': {}}
        # - start_bin
        min_value = min(df[field])
        if 'start_bin' in kwargs:
            min_value = int(kwargs['start_bin'])
        data['xbins']['start'] = min_value
        # - end_bin
        max_value = max(df[field])
        if 'end_bin' in kwargs:
            max_value = int(kwargs['end_bin'])
        data['xbins']['end'] = max_value + 0.01
        # - width_bins
        range_value = max_value - min_value
        # - n_bins
        n_bins = df[field].unique().size - 1
        width_value = range_value/n_bins
        if 'n_bins' in kwargs:
            n_bins = int(kwargs['n_bins'])
            width_value = range_value/n_bins
#             data['xbins']['size']  = width_value
#         width_value = range_value/min([n_bins,df[field].unique().size])
        # - size
        if 'width_bins' in kwargs:
            width_value = kwargs['width_bins']
        data['xbins']['size'] = width_value
        # - verbosity
        if 'verbose' in kwargs and kwargs['verbose']:
            print('min_value: ', min_value)
            print('max_value: ', max_value)
            print('range_value: ', range_value)
            print('width_value: ', width_value)
            print('n_value: ', range_value/width_value)
        # - iplot
        return iplot(go.Figure(data=[go.Histogram(data)], layout=layout))


def plot_attribute_vs_attribute(dataframe, group_model, bar_model, **kwargs):
    # input examples:
    # - categorical:
        # - barmode  : ['group','stack','relative']
            # bar_model={
            #     'field': 'Survived',
            #     'bars':{
            #         0: {
            #             'name' :"Victims",
            #             'color':"#b82214"},
            #         1: {
            #             'name' :"Survivors",
            #             'color':"#75890c"}
            #     }
            # }
            # group_model={
            #     'field': 'Pclass',
            #     'groups': {
            #         1:'1st class',
            #         2:'2nd class',
            #         3:'3rd class'
            #     }
            # }
        # - barmode  : 'stacked_percentage'
            # group_model={
            #     'field':'country',
            #     'groups':{
            #         'chile'    :{'name':'CHI  '},
            #         'mexico'   :{'name':'MEX  '},
            #         'paraguay' :{'name':'PAR  '},
            #         'venezuela':{'name':'VEN  '}
            #     }
            # }
            # bar_model={
            #     'field':'app_disponibility',
            #     'bars':{
            #         1.0:{
            #             'color':'#c90a00',
            #             'name':'Muy Mal'},
            #         2.0:{
            #             'color':'#ff8400',
            #             'name':'Mal'},
            #         3.0:{
            #             'color':'#fcc90f',
            #             'name':'Normal'},
            #         4.0:{
            #             'color':'#rgb(125,204,61)',
            #             'name':'Bien'},
            #         5.0:{
            #             'color':'rgb(26,152,80)',
            #             'name':'Muy Bien'},
            #     }
            # }
    # - numerical:
        # bar_model={
        #     'field': 'Age',
        #     'barmode': 'bars'
        # }
        # group_model={
        #     'field': 'Survived',
        #     'groups': {
        #         0: {
        #             'name' :"Victims",
        #             'color':"#b82214",
        #             'start_bin':0,
        #             'end_bin'  :100,
        #             'n_bins' :5,
        # #             'verbose':True
        #         },
        #         1: {
        #             'name' :"Survivors",
        #             'color':"#75890c",
        #             'start_bin':0,
        #             'end_bin'  :100,
        #             'width_bins' :1,
        #             'opacity': 0.75,
        # #             'verbose':True
        #         },
        #     }
        # }
    valid_kwargs = [
        # common
        'mode', 'plot_title', 'x_title', 'y_title', 'help', 'barmode', 'help', 'rotate_xaxis_ticktext', 'height', 'width',
        # categorical
        'orientation', 'sorting', 'legend_traceorder', 'legend_orientation'
        # numerical
    ]
    if not kwargs:
        kwargs = {}
    # - help
    if ('help' in kwargs) and (kwargs['help'] == 'args'):
        return valid_kwargs
    # - check valid kwargs
    for kwarg in kwargs:
        if kwarg not in valid_kwargs:
            raise Exception("'{}' is not a valid kwarg for this function. Did you spell something wrong?".format(kwarg))
    # - layout
    layout_json = {
        'title': '',
        'legend': {
            'traceorder': 'normal',  # reversed
            'orientation': 'v'  # h
        },
        'xaxis': {},
        'yaxis': {}}
    if 'height' in kwargs:
        layout_json['height'] = kwargs['height']
    if 'width' in kwargs:
        layout_json['width'] = kwargs['width']
    if 'plot_title' in kwargs:
        layout_json['title'] = kwargs['plot_title']
    if 'x_title' in kwargs:
        layout_json['xaxis']['title'] = kwargs['x_title']
    if 'y_title' in kwargs:
        layout_json['yaxis']['title'] = kwargs['y_title']
    if 'legend_traceorder' in kwargs:
        layout_json['legend']['traceorder'] = kwargs['legend_traceorder']
    if 'legend_orientation' in kwargs:
        layout_json['legend']['orientation'] = kwargs['legend_orientation']
    # - orientation
    orientation = 'v'
    if 'orientation' in kwargs:
        orientation = kwargs['orientation']
    # - mode : ['categorical,'numerical','distribution']
    #   if not given, it will be guessed from the column data type
    # bars from data
    df_bars = sorted(dataframe[bar_model['field']].unique())
    # groups from data
    df_groups = sorted(dataframe[group_model['field']].unique())
    if 'mode' not in kwargs:
        if isinstance(df_bars[0], str):
            kwargs['mode'] = 'categorical'
        else:
            kwargs['mode'] = 'numerical'
    # CATEGORICAL
    if kwargs['mode'] == 'categorical':
        # - barmode  : ['group','stack','relative']
        barmode = 'group'
        if 'barmode' in kwargs and kwargs['barmode']:
            barmode = kwargs['barmode']
        layout_json['barmode'] = barmode
        if barmode in ['group', 'stack', 'relative']:
            # - y
            matrix_y = pd.DataFrame(0, index=df_groups, columns=list(set(df_bars + list(bar_model['bars'].keys()))))
            for group in df_groups:
                for bar in df_bars:
                    matrix_y.loc[group, bar] = \
                        len(dataframe[(dataframe[bar_model['field']] == bar) &
                                      (dataframe[group_model['field']] == group)])
            # - customizing hover text
            matrix_text = matrix_y.apply(lambda col: np.around(col/sum(col)*100, decimals=1), axis=1)
            matrix_text = matrix_text.applymap("{:.1f} %".format)
            data = []
            for bar in df_bars:
                # - x,y depending on orientation
                if orientation == 'h':
                    x = matrix_y[bar]
                    y = df_groups
                elif orientation == 'v':
                    x = df_groups
                    y = matrix_y[bar]
                # - bar_json
                bar_json = {
                    'x': x,
                    'y': y,
                    'text': matrix_text[bar],
                    'name': bar,
                    'marker': {}
                }
                if 'groups' in group_model and group_model['groups']:
                    bar_json['x'] = np.vectorize(lambda grp: group_model['groups'][grp]['name'])(df_groups)
                if 'bars' in bar_model and bar_model['bars']:
                    if 'name' in bar_model['bars'][bar]:
                        bar_json['name'] = bar_model['bars'][bar]['name']
                    if 'color' in bar_model['bars'][bar]:
                        bar_json['marker']['color'] = bar_model['bars'][bar]['color']
                data.append(go.Bar(bar_json))
        elif barmode in ['stacked_percentage']:
            # - preparing metrics and groups
            metrics = ['mode', 'median', 'mean']
            metrics_and_groups = list(df_groups)
            group_metrics = []
            if 'groups' in group_model and group_model['groups']:
                group_metrics = list(set(metrics).intersection(group_model['groups'].keys()))[::-1]
                if group_metrics:
                    metrics_and_groups = group_metrics + list(df_groups)
            df_data = pd.DataFrame(0, index=metrics_and_groups, columns=list(set(df_bars + list(bar_model['bars'].keys()))))
            # preparing data for plot
            for group in df_groups:
                df_data.loc[group, '#_total'] = len(dataframe[(dataframe[group_model['field']] == group)])
                df_data.loc[group, '%_total'] = \
                    round(len(dataframe[(dataframe[group_model['field']] == group)])*100/len(dataframe), 1)
            for group in df_groups:
                for bar in df_bars:
                    df_data.loc[group, bar] = \
                        round(len(dataframe[(dataframe[group_model['field']] == group) &
                                            (dataframe[bar_model['field']] == bar)])/df_data.loc[group, '#_total']*100,
                              1)
            # - adding metrics
            # mode
            if 'mode' in metrics_and_groups:
                df_data.loc['mode'] = \
                    round(df_data.iloc[len(group_metrics):].
                          apply(lambda col: scipy.stats.mode(col).mode[0], axis='index'), 1)
            # median
            if 'median' in metrics_and_groups:
                df_data.loc['median'] = round(df_data.iloc[len(group_metrics):, :].apply(np.median, axis='index'), 1)
            # mean
            if 'mean' in metrics_and_groups:
                df_data.loc['mean'] = round(df_data.iloc[len(group_metrics):, :].apply(np.mean, axis='index'), 1)
            # - sorting metrics and groups
            if 'sorting' in group_model and group_model['sorting']:
                if sorted(group_model['sorting']) != sorted(metrics_and_groups):
                    raise Exception("The columns groups in group_model['sorting']: \n {group_model_sorting} \n \
does not match with the columns in the dataframe: \n {metrics_and_groups}".format(
                        group_model_sorting=sorted(group_model['sorting']),
                        metrics_and_groups=sorted(metrics_and_groups)))
                df_data = df_data.reindex(group_model['sorting'][::-1])
                metrics_and_groups = group_model['sorting'][::-1]
            # - sorting bars
            if 'sorting' in bar_model and bar_model['sorting']:
                if 'value' in bar_model['sorting']:
                    sort_value = bar_model['sorting']['value']
                sort_order = True
                if 'ascending' in bar_model['sorting']:
                    sort_order = bar_model['sorting']['ascending']
                value_ascending_order = df_data.iloc[:][sort_value].sort_values(ascending=sort_order)
                df_data = df_data.reindex(list(value_ascending_order.index))
                metrics_and_groups = list(value_ascending_order.index)
            # - data
            data = []
            # - groups from model
            if 'groups' in group_model and group_model['groups']:
                for index, group in enumerate(metrics_and_groups):
                    # - groups name
                    if group in group_model['groups'] and 'name' in group_model['groups'][group]:
                        metrics_and_groups[index] = group_model['groups'][group]['name']
            # bars from model
            if 'bars' in bar_model and bar_model['bars']:
                bars = list(bar_model['bars'].keys())
            else:
                bars = df_bars
            for index, bar in enumerate(bars):
                # - x,y depending on orientation
                if orientation == 'h':
                    x = df_data[bar]
                    y = metrics_and_groups
                elif orientation == 'v':
                    y = metrics_and_groups
                    x = df_data[bar]
                # - data_json
                data_json = {
                    'x':x,
                    'y':y,
                    'text': x.apply("{} %".format),
                    'name': bar,
                    'hoverinfo': 'y+z+text+name',
                    'orientation': orientation,
                    'marker': {}}
                # - bars options
                if 'bars' in bar_model and bar_model['bars'] and bar in bar_model['bars']:
                    # - bars name
                    if 'name' in bar_model['bars'][bar]:
                        data_json['name'] = bar_model['bars'][bar]['name']
                    # - bars color
                    if 'color' in bar_model['bars'][bar]:
                        data_json['marker']['color'] = bar_model['bars'][bar]['color']
                data.append(go.Bar(data_json))
            layout_json['barmode'] = 'stack'
            layout_json['xaxis']['tickvals'] = \
                [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
            layout_json['xaxis']['ticktext'] = \
                ["0 %", "'", '|', "'", '|', "'", '|', "'", '|', "'",
                 '50 %', "'", '|', "'", '|', "'", '|', "'", '|', "'", '100 %']
            if 'rotate_xaxis_ticktext' in kwargs and kwargs['rotate_xaxis_ticktext'] is True:
                    layout_json['xaxis']['ticktext'] = \
                        ["0 %", ".", '-', ".", '-', ".", '-', ".", '-', ".", '50 %',
                         ".", '-', ".", '-', ".", '-', ".", '-', ".", '100 %']
#             layout_json['xaxis']['showticklabels']=False
        layout = go.Layout(layout_json)
        # - iplot
        return iplot(go.Figure(data=data, layout=layout))
    ### NUMERICAL ###
    elif kwargs['mode'] == 'numerical':
        # - check n_bins and bin_width simultaneously
        if 'n_bins' in kwargs and 'bin_width' in kwargs:
            raise Exception("Arguments 'n_bins' and 'bin_width' cannot be used simultaneously.")
        # - check proper start_bin and end_bin
        if ('start_bin' in kwargs) and ('end_bin' in kwargs) and (kwargs['start_bin'] >= kwargs['end_bin']):
            raise Exception("Argument 'start_bin' has to be lower than 'end_bin'.")
        # - data
        data = []
        for group in df_groups:
            histogram_json = {
                'name': group,
                'xbins': {},
                'marker': {},
                'opacity': 0.5}
            histogram_data = dataframe[dataframe[group_model['field']] == group][bar_model['field']]
            # - groups options
            if 'groups' in group_model and group_model['groups']:
                # - name
                if 'name' in group_model['groups'][group]:
                    histogram_json['name'] = group_model['groups'][group]['name']
                # - color
                if 'color' in group_model['groups'][group]:
                    histogram_json['marker']['color'] = group_model['groups'][group]['color']
                # - start_bin
                min_value = min(histogram_data)
                if 'start_bin' in group_model['groups'][group]:
                    min_value = int(group_model['groups'][group]['start_bin'])
                histogram_json['xbins']['start'] = min_value
                # - end_bin
                max_value = max(histogram_data)
                if 'end_bin' in group_model['groups'][group]:
                    max_value = int(group_model['groups'][group]['end_bin'])
                histogram_json['xbins']['end'] = max_value
                range_value = max_value - min_value
                width_value = range_value/min([10, histogram_data.unique().size])
                # - n_bins
                if 'n_bins' in group_model['groups'][group]:
                    width_value = range_value/int(group_model['groups'][group]['n_bins'])
                # - size
                if 'width_bins' in group_model['groups'][group]:
                    width_value = int(group_model['groups'][group]['width_bins'])
                histogram_json['xbins']['size'] = width_value
                # - opacity
                if 'opacity' in group_model['groups'][group]:
                    histogram_json['opacity'] = group_model['groups'][group]['opacity']
                # - verbosity
                if 'verbose' in group_model['groups'][group]:
                    print(' -- group --> : ', histogram_json['name'])
                    print('min_value: ', min_value)
                    print('max_value: ', max_value)
                    print('range_value: ', range_value)
                    print('width_value: ', width_value)
                    print('n_value: ', range_value/width_value)
            # - x,y depending on orientation
            if orientation == 'h':
                histogram_json['y'] = histogram_data
            elif orientation == 'v':
                histogram_json['x'] = histogram_data
            data.append(go.Histogram(histogram_json))
        # - barmode  : ['overlay','stack','bars']
        barmode = 'overlay'
        if 'barmode' in bar_model:
            if bar_model['barmode'] == 'bars':
                barmode = None
            else:
                barmode = bar_model['barmode']
        layout_json['barmode'] = barmode
        layout = go.Layout(layout_json)
        # - iplot
        return iplot(go.Figure(data=data, layout=layout))
    # DISTRIBUTION
    elif kwargs['mode'] == 'distribution':
        # :param (list[str]) group_labels: Names for each data set.
        # :param (list[float]|float) bin_size: Size of histogram bins. Default = 1.
        # :param (str) curve_type: 'kde' or 'normal'. Default = 'kde'
        # :param (str) histnorm: 'probability density' or 'probability'. Default = 'probability density'
        # :param (bool) show_hist: Add histogram to distplot? Default = True
        # :param (bool) show_curve: Add curve to distplot? Default = True
        # :param (bool) show_rug: Add rug to distplot? Default = True
        # :param (list[str]) colors: Colors for traces.
        # :param (list[list]) rug_text: Hovertext values for rug_plot,

        histogram_data = []  # sin relaccionar

        bin_size = 1
        if 'bin_size' in kwargs:
            bin_size = kwargs['bin_size']
        curve_type = 'kde'
        if 'curve_type' in kwargs:
            curve_type = kwargs['curve_type']
        colors = None
        if 'colors' in kwargs:
            colors = kwargs['colors']
        rug_text = None
        if 'rug_text' in kwargs:
            rug_text = kwargs['rug_text']
        histnorm = 'probability density'
        if 'histnorm' in kwargs:
            histnorm = kwargs['histnorm']
        show_hist = True
        if 'show_hist' in kwargs:
            show_hist = kwargs['show_hist']
        show_curve = True
        if 'show_curve' in kwargs:
            show_curve = kwargs['show_curve']
        show_rug = True
        if 'show_rug' in kwargs:
            show_rug = kwargs['show_rug']
        fig = ff.create_distplot(
            histogram_data,
            df_groups,
            bin_size=bin_size,
            curve_type=curve_type,
            colors=colors,
            rug_text=rug_text,
            histnorm=histnorm,
            show_hist=show_hist,
            show_curve=show_curve,
            show_rug=show_rug)
        fig['layout'] = layout_json

        data = []  # sin relaccionar
        layout = []  # sin relaccionar

        ff.create_distplot(histogram_data, df_groups, colors=colors, bin_size=.25, show_curve=False)
        return iplot(go.Figure(data=data, layout=layout))
        # bar_model={
        #     'field': 'app_disponibility'
        # }
        # group_model={
        #     'field': 'country',
        #     'groups': {
        #         'chile': {
        # #             'name' :"Victims",
        #             'color':"#ff0000",
        #             'opacity': 0.5},
        #         'mexico': {
        # #             'name' :"Survivors",
        #             'color':"#74890c",
        #             'opacity': 0.5},
        #         'paraguay': {
        # #             'name' :"Victims",
        #             'color':"#40415c",
        #             'opacity': 0.5},
        #         'venezuela': {
        # #             'name' :"Survivors",
        #             'color':"#a9032a",
        #             'opacity': 0.5}
        #     }
        # }
        # group_model['groups']
        # plot_attribute_vs_attribute(
        #     df[(df['app_disponibility_part']==True)],
        #     group_model, bar_model,
        # #     plot_title= 'Fare distribution for survivors and victims - Range [0,100]',
        # #     y_title= '# passengers'
        # )

        # :param (list[str]) group_labels: Names for each data set.
        # :param (list[float]|float) bin_size: Size of histogram bins. Default = 1.
        # :param (str) curve_type: 'kde' or 'normal'. Default = 'kde'
        # :param (str) histnorm: 'probability density' or 'probability'. Default = 'probability density'
        # :param (bool) show_hist: Add histogram to distplot? Default = True
        # :param (bool) show_curve: Add curve to distplot? Default = True
        # :param (bool) show_rug: Add rug to distplot? Default = True
        # :param (list[str]) colors: Colors for traces.
        # :param (list[list]) rug_text: Hovertext values for rug_plot,

        # x = np.random.randn(1000)
        # hist_data = [x]
        # group_labels = ['distplot']

        # fig = ff.create_distplot(hist_data, group_labels)
        # iplot(fig, filename='Basic Distplot')

        # # Add histogram data
        # x1 = np.random.randn(200)-2
        # x2 = np.random.randn(200)
        # x3 = np.random.randn(200)+2
        # x4 = np.random.randn(200)+4

        # # Group data together
        # hist_data = [x1, x2, x3, x4]

        # group_labels = ['Group 1', 'Group 2', 'Group 3', 'Group 4']

        # # Create distplot with custom bin_size
        # fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)

        # # Plot!
        # iplot(fig, filename='Distplot with Multiple Datasets')

        # df = df[(df['app_disponibility_part']==True)]

        # bin_size=1
        # curve_type='kde'
        # colors=None
        # rug_text=None
        # histnorm='probability density'
        # show_hist=True
        # show_curve=True
        # show_rug=True

        # bar_model={
        #     'field': 'app_disponibility'
        # }
        # group_model={
        #     'field': 'country',
        #     'groups': {
        #         'chile': {
        #             'name' :"chi",
        #             'color':"#ff0000",
        #             'opacity': 0.9},
        #         'mexico': {
        #             'name' :"mex",
        #             'color':"#74890c",
        #             'opacity': 0.2},
        #         'paraguay': {
        #             'name' :"para",
        #             'color':"#40415c",
        #             'opacity': 0.5},
        #         'venezuela': {
        #             'name' :"ven",
        #             'color':"#a9032a",
        #             'opacity': 0.7}
        #     }
        # }

        # fig = ff.create_distplot(
        #     hist_data,
        #     group_labels,
        #     bin_size  = bin_size,
        #     curve_type= curve_type,
        #     colors    = colors,
        #     rug_text  = rug_text,
        #     histnorm  = histnorm,
        #     show_hist = show_hist,
        #     show_curve= show_curve,
        #     show_rug  = show_rug)

        # # group_model
        # groups = list(group_model['groups'].keys())
        # # bar_model
        # histogram_data = []
        # for group in groups:
        #     histogram_data.append(list(df[df[group_model['field']]==group][bar_model['field']]))

        # # bin_size= 1
        # # if 'bin_size' in kwargs:   bin_size= kwargs['bin_size']
        # # curve_type= 'kde'
        # # if 'curve_type' in kwargs: curve_type= kwargs['curve_type']
        # # colors= None
        # # if 'colors' in kwargs:     colors= kwargs['colors']
        # # rug_text= None
        # # if 'rug_text' in kwargs:   rug_text= kwargs['rug_text']
        # # histnorm= 'probability density'
        # # if 'histnorm' in kwargs:   histnorm= kwargs['histnorm']
        # # show_hist= True
        # # if 'show_hist' in kwargs:  show_hist= kwargs['show_hist']
        # # show_curve= True
        # # if 'show_curve' in kwargs: show_curve= kwargs['show_curve']
        # # show_rug= True
        # # if 'show_rug' in kwargs:   show_rug= kwargs['show_rug']

        # fig = ff.create_distplot(
        #     histogram_data,
        #     groups,
        #     bin_size  = bin_size,
        #     curve_type= curve_type,
        #     colors    = colors,
        #     rug_text  = rug_text,
        #     histnorm  = histnorm,
        #     show_hist = show_hist,
        #     show_curve= show_curve,
        #     show_rug  = show_rug)

        # # layout_json = {
        # #     'title': '',
        # #     'xaxis': {},
        # #     'yaxis': {}}

        # # if 'plot_title' in kwargs: layout_json['title']          = kwargs['plot_title']
        # # if 'x_title'    in kwargs: layout_json['xaxis']['title'] = kwargs['x_title']
        # # if 'y_title'    in kwargs: layout_json['yaxis']['title'] = kwargs['y_title']

        # # fig['layout']= layout_json

        # iplot(fig)

