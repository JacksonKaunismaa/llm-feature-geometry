import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
import einops
from functools import reduce
import scipy

from .. import load_results

def find_good_ratio(size):
    """Given a size, find the closest (height,width) ratio of rows to columns that is as close to a square as possible."""
    if size == 1:
        return 1, 1
    start_width = int(np.sqrt(size))
    width = start_width
    # for width in range(start_width+1, 0, -1):
    #     if size % width == 0:
    #         break
    for height in range(size//width, size, 1):
        if height*width >= size:
            break
    return min(height, width), max(height, width)  # height, width


def tsne_activations(results_dict, split_by_layer=False):
    """Compute tSNE of activations, and return the results and feature names.
    
    Args:
    results_dict: dict of dicts, {dset_name: {feat_name: (layer, hidden_dim)}}
    split_by_layer: bool, whether to return a tSNE result for each layer, or a single tSNE result for all layers.
    
    Returns:
    tsne_results: np.ndarray, (layers, feats, 2)
    feature_names: dict, {dset_name: [feat_names]}"""
    layer_coefs = load_results.extract_feats_and_concatenate(results_dict) # (layer, feat, coef)
    feature_names = {dset_name: [feat for feat in dset_feats.keys()] for dset_name, dset_feats in results_dict.items()}
    if split_by_layer:
        tsne_results = np.stack([TSNE(n_components=2).fit_transform(coefs) for coefs in layer_coefs], axis=0)
    else:
        tsne_results = TSNE(n_components=2).fit_transform(layer_coefs.reshape(-1, layer_coefs.shape[-1])).reshape(
                                                                                        layer_coefs.shape[0], layer_coefs.shape[1], 2)

    return tsne_results, feature_names


def plot_tsne_split_layers(tsne_results, feature_names, title_text=None):
    """Plot tSNE results (computed with split_by_layer=True), with a different color for each dataset, 
    and a different subplot for each layer.
    
    Args:
    tsne_results: np.ndarray, (layers, feats, 2)
    feature_names: dict, {dset_name: [feat_names]}
    title_text: str, title of the plot."""
    num_layers = len(tsne_results)
    nrows, ncols = find_good_ratio(num_layers)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[f'Layer {i}' for i in range(num_layers)])
    colors = px.colors.qualitative.Plotly
    for i, embeds in enumerate(tsne_results):
        idx = 0
        for j, (dset_name, feat_names) in enumerate(feature_names.items()):
            num_feats = len(feat_names)
            fig.add_trace(go.Scatter(x=embeds[idx:idx+num_feats,0], 
                                    y=embeds[idx:idx+num_feats,1], 
                                    mode='markers', 
                                    marker=dict(color=colors[j]),
                                    name=dset_name, 
                                    text=feat_names,
                                    legendgroup=dset_name,
                                    showlegend=i==0), 
                                    row=i//4+1, col=i%4+1)
            idx += num_feats
    fig.update_layout(height=1400, width=1400, title_text=title_text)
    fig.show()


def plot_tsne_agg_layers(tsne_results, feature_names, title_text=None):
    """Plot tSNE results (computed with split_by_layer=False), with a different color for each dataset, 
    and every layer in the same plot.
    
    Args:
    tsne_results: np.ndarray, (layers, feats, 2)
    feature_names: dict, {dset_name: [feat_names]}
    title_text: str, title of the plot."""
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    num_layers = tsne_results.shape[0]
    idx = 0
    for j, (dset_name, feat_names) in enumerate(feature_names.items()):
        num_feats = len(feat_names)
        embeds = tsne_results[:, idx:idx+num_feats, :].reshape(-1, 2)  # (layers, feats, 2) -> (layers*feats, 2)
        feat_layer_names = [f'{feat_name}.{i}' for i in range(num_layers) for feat_name in feat_names]
        fig.add_trace(go.Scatter(x=embeds[:,0], 
                                y=embeds[:,1], 
                                mode='markers', 
                                marker=dict(color=colors[j]),
                                name=dset_name, 
                                text=feat_layer_names))
        idx += num_feats
    fig.update_layout(height=500, width=1000, title_text=title_text)
    fig.show()


def get_grams(results_dict, feature_set=None):
    """Compute gram matrices of the normalized coefficients/feature vectors and gram matrices that result
      from random coefficient vectors with the same global mean and standard devitaion as the actual coefficients.
      
      Args:
      results_dict: dict of dicts, {dset_name: {feat_name: (layer, hidden_dim)}}
      feature_set: str, the feature subset to use. If None, we use all feature sets.
      
      Returns:
      grams: np.ndarray, (feats, feats, layers, layers)
      rand_grams: np.ndarray, (feats, feats, layers, layers)
      norm_coefs: np.ndarray, (layers, feats, hidden_dim)
      rand_coefs: np.ndarray, (layers, feats, hidden_dim)
      feature_names: list of str, the feature names."""
    
    if feature_set is None:
        coefs_only = load_results.extract_feats_and_concatenate(results_dict)  # (layers, feats, hidden_dim)
        feature_names = [feat for dset_name, dset_feats in results_dict.items() for feat in dset_feats.keys()]
    else:
        coefs_only = load_results.extract_feats_and_concatenate(results_dict[feature_set])
        feature_names = list(results_dict[feature_set].keys())
    norm_coefs = coefs_only / np.linalg.norm(coefs_only, axis=-1, keepdims=True)
    grams = einops.einsum(norm_coefs, norm_coefs, 'l1 f1 k, l2 f2 k -> f1 f2 l1 l2')
    rand_coefs = np.random.normal(loc=norm_coefs.mean(), scale=norm_coefs.std(), size=norm_coefs.shape)
    rand_coefs = rand_coefs / np.linalg.norm(rand_coefs, axis=-1, keepdims=True)
    rand_grams = einops.einsum(rand_coefs, rand_coefs, 'l1 f1 k, l2 f2 k -> f1 f2 l1 l2')
    return grams, rand_grams, norm_coefs, rand_coefs, feature_names


def fast_plotly_hist(fig, x, hist_kwargs=None, bar_kwargs=None, trace_kwargs=None):
    """Plot a histogram with plotly, and add it to the figure. Computing the histogram is done in numpy since plotly
    would do it in Javascript, which is immensely slow, especially over slow connections.
    
    Args:
    fig: plotly.graph_objects.Figure, the figure to add the histogram to.
    x: np.ndarray, the data to plot.
    hist_kwargs: dict, kwargs to pass to np.histogram.
    bar_kwargs: dict, kwargs to pass to go.Bar.
    trace_kwargs: dict, kwargs to pass to fig.add_trace."""
    if hist_kwargs is None:
        hist_kwargs = {}
    if bar_kwargs is None:
        bar_kwargs = {}
    if trace_kwargs is None:
        trace_kwargs = {}
    hist, bin_edges = np.histogram(x, **hist_kwargs)
    fig.add_trace(go.Bar(x=(bin_edges[:-1]+bin_edges[1:])/2, y=hist, 
                         width=(bin_edges[1]-bin_edges[0]), **bar_kwargs), **trace_kwargs)
    fig.update_traces(marker_line_width=0, selector=dict(type='bar'))
    fig.update_layout(bargap=0, bargroupgap=0)


def plot_grams(grams, rand_grams, norm_coefs, rand_coefs, feature_names, title_text, layers=None):
    """Plot the gram matrices of the normalized coefficients `grams`, the gram matrices of random coefficients `rand_grams`,
    with the same global mean and standard deviation as the actual coefficients, and the histograms of the 
    gram matrices `norm_coefs` and the random coefficients `rand_coefs`, ie. the outputs of get_grams. The histograms are probability 
    density histograms, and the gram matrices are plotted as heatmaps.
    """

    if layers is None:
        layers = np.arange(grams.shape[-1])
    if isinstance(layers, int):
        layers = [layers]
    # select the layers we want
    grams = grams[..., layers[:,None], layers[None]]  # (feat, feat, layer, layer)
    rand_grams = rand_grams[..., layers[:,None], layers[None]]  # weird indexing to get 2d slice
    norm_coefs = norm_coefs[..., layers]
    rand_coefs = rand_coefs[..., layers]
    print(grams.shape, rand_grams.shape)
    num_layers = grams.shape[-1]
    layer_layer_grams = grams[..., np.arange(num_layers), np.arange(num_layers)]
    # print(layer_layer_grams.shape)
    # plot 4x4 grid of each layer_layer gram matrix
    nrows, ncols = find_good_ratio(num_layers)
    print(nrows, ncols)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[f'Layer {i}' for i in layers], 
                        shared_xaxes='all', shared_yaxes='all')
    for i in range(num_layers):
        fig.add_trace(go.Heatmap(z=layer_layer_grams[...,i], coloraxis='coloraxis'), row=i//nrows+1, col=i%ncols+1)
    # blue red colorscale from -1 to 1
    fig.update_layout(coloraxis=dict(colorscale='RdBu', cmin=-1, cmax=1))
    # fig.update_xaxes(title=dict(text='Feature 1'))
    # fig.update_yaxes(title=dict(text='Feature 2'))
    invisible_font = dict(color='rgba(0,0,0,0)', size=1)
    fig.update_xaxes(tickmode='array', ticktext=feature_names, tickvals=np.arange(len(feature_names)))#, tickfont=invisible_font,)
                    #  title=dict(text='Feature 1'))
    fig.update_yaxes(tickmode='array', ticktext=feature_names, tickvals=np.arange(len(feature_names)))#, tickfont=invisible_font)
    height_per = 400 if nrows > 2 else 500
    width_per = 400 if ncols > 2 else 500
    fig.update_layout(height=height_per*nrows, width=width_per*ncols, title_text=title_text[0])#, xaxis_title='Feature 1', yaxis_title='Feature 2')


    """Code to show a histogram of coefficient values and gram matrix values. 
    Should move to another function if we want to use this again"""
    # histogram of layer_layer_grams, rand_grams, and grams, opacity 0.5, probability density
    # fig = go.Figure()
    # names = ['Within same layer grams', 'Random Grams with same mean,std', 'Between layer grams']
    # gram_types = [layer_layer_grams, rand_grams, grams]
    # for gram_type, name in zip(gram_types, names):
    #     num_grams = reduce(lambda x,y: x*y, gram_type.shape)
    #     fast_plotly_hist(fig, gram_type.flatten()[~np.isclose(gram_type.flatten(), 1)],
    #                      bar_kwargs=dict(name=name, opacity=0.5), 
    #                      hist_kwargs=dict(density=True, bins='fd'))
    # fig.update_layout(barmode='overlay', title_text=title_text[1])
    # fig.update_traces(opacity=0.75)
    # fig.show()

    # # histogram of norm_coefs and rand_coefs, opacity 0.5, probability density
    # fig = go.Figure()
    # names = ['Actual coefficients', 'Random coefficients']
    # coef_types = [norm_coefs, rand_coefs]
    # for coef_type, name in zip(coef_types, names):
    #     fast_plotly_hist(fig, coef_type.flatten(), 
    #                      bar_kwargs=dict(name=name, opacity=0.5), 
    #                      hist_kwargs=dict(density=True, bins='fd'))
    # fig.update_layout(barmode='overlay', title_text=title_text[2])
    # fig.update_traces(opacity=0.75)
    # fig.show()


def plot_multi_grams(coefs_dict, good_keys, overall_title, sv_name, layer=8):
    """Plot the gram matrices of the normalized coefficients, for a specific list of feature sets `good_keys`, and 
    a specific `layer`. The gram matrices are plotted as heatmaps. Save the plot to `sv_name`."""
    # select the layers we want
    gram_dict = {}
    for key in good_keys:
        gram_dict[key] = get_grams(coefs_dict, key)
    num_plots = len(gram_dict)
    nrows, ncols = find_good_ratio(num_plots)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[x.split('.')[0] for x in good_keys])
    invisible_font = dict(color='rgba(0,0,0,0)', size=1)

    for i,(k,v) in enumerate(gram_dict.items()):
        grams, rand_grams, norm_coefs, rand_coefs, feat_names = v
        layer_layer_grams = grams[..., np.arange(grams.shape[-1]), np.arange(grams.shape[-1])]
        fig.add_trace(go.Heatmap(z=layer_layer_grams[...,layer], coloraxis='coloraxis'), row=i//ncols+1, col=i%ncols+1)
        next(fig.select_xaxes(row=i//ncols+1, col=i%ncols+1)).update(tickmode='array', 
                                                               ticktext=feat_names, 
                                                               tickvals=np.arange(len(feat_names)))#, tickfont=invisible_font,)
        next(fig.select_yaxes(row=i//ncols+1, col=i%ncols+1)).update(tickmode='array',
                                                                ticktext=feat_names, 
                                                                tickvals=np.arange(len(feat_names)))#, tickfont=invisible_font)

    # blue red colorscale from -1 to 1
    fig.update_layout(coloraxis=dict(colorscale='RdBu', cmin=-1, cmax=1))
    # fig.update_yaxes(tickmode='array', ticktext=feature_names, tickvals=np.arange(len(feature_names)))#, tickfont=invisible_font)
    height_per = 400 if nrows > 2 else 500
    width_per = 400 if ncols > 2 else 500
    fig.update_layout(height=height_per*nrows, width=width_per*ncols, title_text=overall_title)#, xaxis_title='Feature 1', yaxis_title='Feature 2')
    fig.write_image(sv_name)
    fig.show()



def plot_gram_grid(grams, titles, axes=None, sv_name=None):
    """Plot a grid of gram matrices, where grams is just a list of gram matrices."""

    nrows, ncols = find_good_ratio(len(grams))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles)
    for i, gram in enumerate(grams):
        fig.add_trace(go.Heatmap(z=gram, coloraxis='coloraxis'), row=i//ncols+1, col=i%ncols+1)
        if axes is not None and axes[i] is not None:
            next(fig.select_xaxes(row=i//ncols+1, col=i%ncols+1)).update(tickmode='array', 
                                                               ticktext=axes[i], 
                                                               tickvals=np.arange(len(axes[i])),
                                                               showticklabels=True)
            next(fig.select_yaxes(row=i//ncols+1, col=i%ncols+1)).update(tickmode='array',
                                                                ticktext=axes[i], 
                                                                tickvals=np.arange(len(axes[i])),
                                                                showticklabels=False)
    fig.update_layout(coloraxis=dict(colorscale='RdBu', cmin=-1, cmax=1))
    fig.update_layout(height=500*nrows, width=500*ncols, title='Gram Matrices')

    if sv_name is not None:
        fig.write_image(sv_name)
    fig.show()


def minimal_hyperspherical_energy(dims, features, visualize=False):
    """Find a configuration of `features` points on the unit hypersphere in `dims` dimensions, such that the energy function, 
    defined as the sum of pairwise dot products, is minimized."""
    x0 = np.random.standard_normal(size=(features, dims))
    x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
    # minimize sum of pairwise dot products while points are on the unit sphere
    def dots(x):
        x = x.reshape(features, dims)
        return x @ x.T

    def energy(x):
        return np.exp(dots(x)).sum()
    
    def on_sphere(x):
        x = x.reshape(features, dims)
        return np.linalg.norm(x, axis=1) - 1
    
    constraint = {'type': 'eq', 'fun': on_sphere}
    
    result = scipy.optimize.minimize(energy, x0.flatten(), method='trust-constr', constraints=constraint)
    x0 = result.x.reshape(features, dims)
    return x0, dots(x0), result.fun


def visualize_many(dims, features, sv_name):
    """Visualize the gram matrices of the optimal configurations of `features` points on the unit hypersphere in `dims` dimensions,
    for each number of features and each number of dimensions. The gram matrices are plotted as heatmaps."""
    total_num = len(dims)*len(features)
    nrows, ncols = find_good_ratio(total_num)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[f'Dims: {d}, Features: {f}' for d in dims for f in features])
    for i,d in enumerate(dims):
        for j,f in enumerate(features):
            _, grams, _ = minimal_hyperspherical_energy(d, f, visualize=False)
            print(i+1, j+1, nrows, ncols)
            fig.add_trace(go.Heatmap(z=grams, coloraxis='coloraxis'), row=j+1, col=i+1)
    fig.update_layout(coloraxis=dict(colorscale='RdBu', cmin=-1, cmax=1))
    fig.update_layout(height=200*nrows, width=200*ncols, title='Gram Matrices of Optimal Configurations')
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.write_image(sv_name)
    fig.show()
