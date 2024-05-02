import numpy as np
import scipy.optimize
import einops
from functools import partial
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.metrics import pairwise_distances
from scipy import special
import time
import pandas as pd
import warnings
import itertools
from tqdm import tqdm

from .geometry import find_good_ratio, get_grams
from .. import load_results
from .find_permutation import combined_sorting, poly_order_space


def plot_joint_energy_results(final_opts, feature_names=None, sv_name=None, colorscale='Hot'):
    """Plot the results of the joint optimization of the energy function. We plot the actual dist matrix, and the
    optimal dist matrix for each feature dimensionality."""
    if feature_names is None:
        feature_names = list(range(len(final_opts[0][0])))

    height,width = find_good_ratio(len(final_opts))
    max_z = max([np.abs(opt[0]).max() for opt in final_opts])
    fig = make_subplots(rows=height, cols=width, 
                        subplot_titles=['Actual similarity matrix'] + [f"{ndims=} ({cost:.2f})" 
                                                                 for _,ndims,cost,t in final_opts[1:]])
    for i, (final_opt,ndim,cost,time) in enumerate(final_opts):
        fig.add_trace(go.Heatmap(z=final_opt, coloraxis='coloraxis', zmin=0, zmax=max_z), row=i//width+1, col=i%width+1)
    cmin = 0 if colorscale == 'Hot' else -max_z
    fig.update_layout(coloraxis=dict(colorscale=colorscale, cmin=cmin, cmax=max_z))
    fig.update_layout(height=300*height, width=300*width, title='PSM comparison between optimal and actual feature dists')
    fig.update_xaxes(ticktext=feature_names, tickvals=np.arange(len(feature_names)), showticklabels=False)
    fig.update_yaxes(ticktext=feature_names, tickvals=np.arange(len(feature_names)), showticklabels=False)

    if sv_name is not None:
        fig.write_image(sv_name)
    fig.show()


def show_targets(coefs_dict, feature_set, feature_set_name, layer_set, ref_metrics, permute=False, sv_name=None):
    """Extract geometries for each layer and metric, and plot them in a heatmap."""
    all_targets = {}
    for layer in layer_set:
        for ref_metric in ref_metrics:
            all_targets[layer,ref_metric] = extract_targets(coefs_dict[feature_set], ref_metric, layer, permute)
        
    feat_names = list(coefs_dict[feature_set].keys())

    num_cols = len(layer_set)
    num_rows = len(ref_metrics)
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=[f'Layer {i+1}' for i in layer_set], shared_yaxes=True)
    # update subtitel font size
    for i in range(1, (num_rows*num_cols)//2+1):
        fig.layout.annotations[i-1].update(font=dict(size=20))
    for i, ((layer, ref_metric), pdm) in enumerate(all_targets.items()):
        row,col = i%num_rows+1, i//num_rows+1
        fig.add_trace(go.Heatmap(z=pdm, coloraxis=f'coloraxis{row}'), row=row, col=col)
        invisible_font = {'color': 'rgba(0,0,0,0)'}
        fig.update_xaxes(row=row, col=col, tickfont=invisible_font)
        if col == 1:
            fig.update_yaxes(title_text=ref_metric, row=row, col=col)#, tickfont=invisible_font)
            coloraxis_opts = dict(colorscale='Hot', 
                                cmin=min([pdm.min() for (_,metric), pdm in all_targets.items() if metric == ref_metric]), 
                                cmax=max([pdm.max() for (_,metric), pdm in all_targets.items() if metric == ref_metric]),
                                colorbar_len=5/(num_rows*6), colorbar_y=5*(num_rows-row+0.3)/(num_rows*4))
            fig.update_yaxes(row=row, col=col, tickmode='array', tickfont=dict(size=12),
                             ticktext=feat_names, tickvals=np.arange(len(feat_names)))
            # fig.update_xaxes(row=row, col=col, tickmode='array', tickangle=-45,
            #                  ticktext=feat_names, tickvals=np.arange(len(feat_names)))
            if ref_metric == 'RBF':
                coloraxis_opts.update(dict(colorscale='Agsunset'))
            if ref_metric == 'cosine':
                coloraxis_opts.update(dict(colorscale='RdBu', cmin=-1, cmax=1))
            fig.update_layout(**{f'coloraxis{row}': coloraxis_opts})
        fig.update_layout(title_text=f"Target Geometry for each Layer and Metric on {feature_set_name} Dataset", font=dict(size=20),
                          showlegend=False)
    fig.update_layout(height=270*num_rows, width=250*(num_cols-1.5))
    if sv_name:
        fig.write_image(sv_name)
    fig.show()


def similarity_and_energy(points, similarity_metric, energy_metric, **kwargs):
    """Given a set of points, compute the similarity matrix and the energy matrix.
    :returns: (similarity matrix, energy matrix)"""
    if similarity_metric == "RBF" or energy_metric in ['-log', 'riesz']:
        # norm_points = points/np.linalg.norm(points, axis=1, keepdims=True)
        dists = pairwise_distances(points, metric='euclidean')
    
    if similarity_metric == "cosine" or energy_metric == "exp":
        norm_points = points/np.linalg.norm(points, axis=1, keepdims=True)
        grams = norm_points @ norm_points.T

    if energy_metric is None:
        energy = None
    elif energy_metric == "exp":
        energy = np.exp(grams)
    elif energy_metric == "-log":
        energy = -np.ma.log(dists)
    elif energy_metric == "riesz":
        energy = np.ma.divide(np.sign(kwargs['s']), dists**kwargs['s'])
    else:
        raise ValueError(f"Unknown energy metric {energy_metric}")

    if similarity_metric is None:
        similarity = None
    elif similarity_metric == "cosine":
        similarity = grams
    elif similarity_metric == "RBF":
        similarity = np.exp(-dists**2 / 200)
    else:
        raise ValueError(f"Unknown similarity metric {similarity_metric}")
    
    if energy is not None:
        return similarity, energy
    return similarity


def energy_objective(energy, weights):
    """Given a dist matrix or a set of points, and a set of weights, compute the energy function that is a weighted
    sum of the exp of the dot products of the points."""
    weight_probs = special.softmax(weights)
    weights_outer = np.outer(weight_probs, weight_probs)
    return np.sum(energy*weights_outer)


def objectives(x, targets, lamda_xi, lamda_w, lamda_diff, energy_metric, similarity_metric, weights, 
             energy_options, force_all=False, constraint=None):
    """Given a set of weights and points, compute the energy function, the weight regularization, and the dist difference.
    Return a dictionary of the costs in each category {cost_name: cost_value}."""
    n_feats = targets.shape[0]
    if weights is None:
        weights = x[:n_feats]
        points = x[n_feats:].reshape(n_feats, -1)
    else:
        points = x.reshape(n_feats, -1)
    psm, energy = similarity_and_energy(points, similarity_metric, energy_metric, **energy_options)
    
    costs = {}
    if lamda_diff:   # difference with target dists
        dist_difference = ((psm-targets)**2).sum()
        costs['similarity_diff'] = lamda_diff*dist_difference
    if lamda_xi:   # limit the norm of our x_i (for riesz energy, larger norm => lower energy)
        if constraint:
            xi_regularization = constraint['fun'](points)
        else:
            xi_regularization = np.linalg.norm(points)
        costs['xi_norm'] = lamda_xi*xi_regularization
    if lamda_w:   # regularize the weights to be close to 0
        weight_regularization = np.linalg.norm(weights)
        costs['weight_norm'] = lamda_w*weight_regularization
    if lamda_diff is None or force_all:  # only optimize the energy if we are not doing dist differences
        costs['energy'] = energy_objective(energy, weights)
    return costs


def on_sphere(x, norms, n_feats, n_dims):
    """Constraint function that ensures that the points are each on a sphere of the given norm."""
    if x.shape[0] == n_feats*n_dims:  # assume it has no "weights" concatenated to it
        points = x.reshape(n_feats, n_dims)
    elif x.ndim == 2 and x.shape[0] == n_feats and x.shape[1] == n_dims:  # already reshaped
        points = x
    else:
        points = x[n_feats:].reshape(n_feats, n_dims)
    return abs(np.linalg.norm(points, axis=1) - norms).sum()


def _find_energy_function(targets, n_dims, lamda_xi, lamda_w, lamda_diff, 
                          similarity_metric, energy_metric, energy_options, method,
                          weights=None, points=None, reset_points=False, perturb_pct=0.1):
    """Given a particular number of dimensions to put points in, jointly optimize the weights and points to minimize
    the objective that involves regularization on the weights and a weighted energy function"""
    # we will consider energy functions of the form sum exp(x_i.dot(x_j)*weight_j*weight_i)
    n_feats = targets.shape[0]
    fixed_weights = weights is not None  # then assume we are in the first optimization step
    if points is None or reset_points:
        points = np.random.rand(n_feats, n_dims)
        points = points/np.linalg.norm(points, axis=1, keepdims=True)
    else:
        points += np.random.randn(*points.shape)*np.linalg.norm(points,axis=1,keepdims=True)*perturb_pct  # randomly perturb the points a bit

    # if constrain:    
    #     constraint = {'type': 'eq', 'fun': partial(on_sphere, norms=1, n_feats=n_feats, n_dims=n_dims)}
    #     # method = 'trust-constr'
    #     lamda_xi = None
    # else:
    #     constraint = {}

    if not fixed_weights:
        weights = np.zeros(n_feats)
        x0 = np.concatenate([weights, points.flatten()])
        constraint = None
    else:
        lamda_diff = None  # we dont have targets to compare against, so disable the dist difference
        lamda_w = None   # weights are fixed so dont regularize them
        x0 = points.flatten()
        constraint = {'type': 'eq', 'fun': partial(on_sphere, norms=np.linalg.norm(points,axis=1), 
                                                   n_feats=n_feats, n_dims=n_dims)}

    # print("shapes are targets", targets, "x0", x0.shape, "n_dims", n_dims, "weights", weights.shape, "points", points.shape)
    # we will regularize the weights to be close to 1
        # method = "CG"
    print("\tBEFORE: All objective are:", objectives(x0, targets=targets, lamda_xi=1, lamda_w=1, lamda_diff=1, 
                                                similarity_metric=similarity_metric, energy_metric=energy_metric, 
                                 weights=(weights if fixed_weights else None), energy_options=energy_options, force_all=True))
    
    _objectives = partial(objectives, targets=targets, lamda_xi=lamda_xi, lamda_w=lamda_w, lamda_diff=lamda_diff, 
                         similarity_metric=similarity_metric, energy_metric=energy_metric, 
                         weights=(weights if fixed_weights else None), energy_options=energy_options, constraint=constraint)
    _objective = lambda x: sum(_objectives(x).values())
    result = scipy.optimize.minimize(_objective, x0.flatten(),
                                     method=method,#  constraints=constraint, 
                                     options={'disp': fixed_weights})
    print(f"\tFinal results of {n_dims}:", _objectives(result.x))
    if constraint:
        print(f"Constraint: {constraint['fun'](result.x)}")
    # print(f"\tFinal results of 4*{n_dims}:", _objectives(4*result.x))

    print("\tAFTER: All objective are:", objectives(result.x, targets=targets, lamda_xi=1, lamda_w=1, lamda_diff=1, 
                                 similarity_metric=similarity_metric, energy_metric=energy_metric, 
                                 weights=(weights if fixed_weights else None), energy_options=energy_options, force_all=True))

    if not fixed_weights:
        print("\tWeights were:", result.x[:n_feats].tolist())
        return result.x[:n_feats], result.x[n_feats:].reshape(n_feats, n_dims)
    return result.x.reshape(n_feats, n_dims)


def extract_targets(coefs_or_dists, metric, layer, permute_similarity):
    # if its a dictionary, assume we have to extract the coefs ourselves and compute the dists
    # kwargs gets fed to distance_metric
    if isinstance(coefs_or_dists, dict):  # {feat_name: coefs ndarray(layers, hidden_size)}
        similarity_dists = load_results.extract_feats_and_concatenate(coefs_or_dists)[layer] # (n_feats, hidden_size)
        if permute_similarity:
            # print(similarity_dists.shape, similarity_dists)
            similarity_dists = np.apply_along_axis(np.random.permutation, 1, arr=similarity_dists)
            # print(similarity_dists.shape, similarity_dists)

        similarity_dists = similarity_and_energy(similarity_dists, metric, None)
    else:  # if its already an ndarry, assume that it is the dists
        similarity_dists = coefs_or_dists
    return similarity_dists


def _mask_targets(targets, thresh):
    n = targets.shape[0]

    # get off diagonal entries that are very similar
    similar_entries = np.nonzero(np.triu((targets > thresh), k=1))
    # eg. if {1, 5, 10, 12}, {2, 8}, and {4, 9} are the redundant sets, we should remove {5, 10, 12, 8, 9},
    #        nonzero will be ([1, 1, 1, 2, 4, 5, 5, 10], [5, 10, 12, 8, 9, 10, 12, 12]) => we can take set(nonzero[1])
    redundant_features = set(similar_entries[1])
    # take away the redundant features
    good_features = list(sorted(set(range(n)) - redundant_features))
    # take the subset of the dists that are not redundant
    masked_dists = targets[good_features, :][:, good_features]
    return masked_dists, good_features


def mask_targets(targets, thresh, num_unique):
    # mask out the target dists if they are very similar. Our energy function assumes that they should not be copies of each other
    # if a set of n features have a dot product greater than thresh, then n-1 one of them can be removed, since they are redundant
    # we compute this by seeing where in the upper triangle is dot product greater than thresh. If inverted, we instead
    # assume that large off-diagonal entries indicate similarity, and low off-diagonal entries indicate dissimilarity

    if thresh is not None and num_unique is not None:
        raise ValueError("thresh and num_unique are mutually exclusive")

    if thresh is None and num_unique is None:
        print("Disabling duplicate feature dropping, since both thresh and num_unique are None")
        return targets, list(range(targets.shape[0]))
    
    if thresh is not None:
        return _mask_targets(targets, thresh)
    
    # if we are here, then num_unique is not None, and we do a binary search to find the thresh
    max_thresh = targets.max()
    min_thresh = targets.min()
    while max_thresh - min_thresh > 1e-6:
        thresh = (max_thresh + min_thresh)/2
        masked_dists, good_features = _mask_targets(targets, thresh)
        # print("with thresh=", thresh, "we have", len(good_features), "features")
        # print("min, max", min_thresh, max_thresh)
        # print()
        if len(good_features) > num_unique:
            max_thresh = thresh  # if too many, then thresh needs to be lower
        elif len(good_features) < num_unique:
            min_thresh = thresh
        else:
            break
    return masked_dists, good_features
    

def find_energy_function(coefs_or_dists, layer=8, thresh=None, num_unique=None, min_dims=2, max_dims=None, lamda_diff=0.1, 
                         similarity_metric="cosine", permute_similarity=False,
                         energy_metric="cosine", energy_options=None,
                         reset_points=False, perturb_pct=0.1, lamda_xi=0.5, lamda_w=0.2,
                         mapper=poly_order_space):
    """Given a set of coefficients or dists, find the optimal set of features that will minimize the energy function. 
    We do a joint optimization of the weights and points to minimize the energy for each feature dimensionality, which 
    ranges from min_dims to the number of features.

    :param layer: the layer to consider
    :param thresh: threshold for similarity between features for redundancy deduplication. Mutually
                     exclusive with num_unique. If None, then `num_unique` is used instead.
    :param num_unique: mutually exclusive with thresh. Restricts the number of features to the num_unique features that 
                       are most different from each other. If None, then thresh is used. If both are None, then all features 
                       are used.
    :param min_dims: the minimum number of dimensions to consider when minimizing energy
    :param max_dims: the maximum number of dimensions to consider when minimizing energy. If None, then it will be (# features - 1)
    
    :param similarity_metric: the function that is applied to the x_i to generate a pairwise matrix to 
                    compare to the actual feature dists. Must be supported by sklearn.metrics.distance_metric
    :param permute_similarity: whether to permute the features before computing the similarity dists. Used to make sure we aren't
                                just getting false positives (in that even random configurations would be "energy minimizing")

    :param energy_metric: the pairwise metric to use for the energy function. Must be in ['exp', '-log', 'riesz']
    :param energy_options: extra arguments to be fed to the energy_metric (eg. s in riesz-s kernel)

    :param reset_points: if False, then x_i in reoptimization is initialized to the output of the first optimization step. 
                            If True, then the initial value of x_i is randomly initialized
    :param perturb_pct: if reset_points is True, then on reoptimization, initialization is randomly perturbed by an isotropic 
                        gaussian with standard deviation equal to perturb_pct times the norm of the points
    :param lamda_diff: weighting for the dists to be close to the similarity dists
    :param lamda_xi: regularization to push norm of each x_i in reoptimization step towards norm from joint optimization step.
    :param lamda_w: regularization parameter for the weights to be close to 1

    :param mapper: the function that is used in the sorting based permutation finder

    :returns: a list of tuples, `(opt_dist, n_dims, final_cost, time)`. `opt_dist` is the reoptimized distance matrix, 
             `n_dims` is the number of dimensions used, and `final_cost` is the final difference between the target 
             after re-optimizing. 'time' is the number of seconds it took to run the optimization."""
    # 1. -log(dist) replace this with the exp(dot)  (or maybe Riesz kernel)
    # 2. use smaller feature subset
    # 3. use average activation features
    # 4. check if re-running optimization changes things
    # 5. investigate how changing lamda_diff and lambda_w changes things
    # 6. double check that layer 8 is the same as everything assumption is good
    # old params:
    # :param constrain: whether to use a hard constraint for a sphere, or a regularization parameter `lamda`
    # 
    if energy_options is None:
        energy_options = {}
    
    unmasked_targets = extract_targets(coefs_or_dists, similarity_metric, layer, permute_similarity)
    masked_targets, good_features = mask_targets(unmasked_targets, thresh, num_unique) # (n_feats, n_feats)
    # target_units = np.linalg.norm(masked_targets)   # to convert the units between different metrics
    # cvt = (target_units/8.5)**2  # all numbers were based on cosine. Cosine has a norm of about 8.5
    cvt = 1

    n_feats = masked_targets.shape[0]
    final_opts = [(masked_targets, -1, np.inf, np.inf)]
    # plot_joint_energy_results(final_opts, feature_names=good_features)
    # return
    if max_dims is None:
        max_dims = n_feats-1
    for n_dims in range(min_dims, max_dims+1):   # everything past n_feats+1 is just a simplex anyway
        start = time.time()
        # lamda_w_now = lamda_w*(1+(n_dims - min_dims)/(max_dims - min_dims)) / cvt
        # lamda_xi_now = lamda_xi/(n_dims**(3/2)) / cvt
        # lamda_diff_now = lamda_diff / cvt
        weights, points = _find_energy_function(masked_targets, n_dims, 0, lamda_w, lamda_diff, 
                                                similarity_metric, energy_metric, energy_options, method="CG")
        # print(points)
        opt_dist = similarity_and_energy(points, similarity_metric, None)
        final_opts.append((opt_dist, -n_dims, np.linalg.norm(opt_dist-masked_targets), time.time() - start))
        # now reoptimize just the energy objective, with weights fixed
        reopt_points = _find_energy_function(masked_targets, n_dims, lamda_xi, lamda_w, lamda_diff, 
                                             similarity_metric, energy_metric, energy_options, method="Powell",
                                             points=points, weights=weights, reset_points=reset_points, perturb_pct=perturb_pct)
        # print(reopt_points)
        # undo any permutations that were learned
        opt_dist = similarity_and_energy(reopt_points, similarity_metric, None)
        permutation = combined_sorting(opt_dist, masked_targets, mapper)
        opt_dist = opt_dist[permutation,:][:,permutation]
        # save the distance matrix and the number of dimensions
        final_cost = np.linalg.norm(opt_dist-masked_targets)
        print("\tFinal cost was:", final_cost/cvt)
        final_opts.append((opt_dist, n_dims, final_cost, time.time() - start))#, reopt_points))
        print()

    return final_opts, good_features


def df_from_hyperresults(hyperresult_dict, column_order, filter_joint=True):
    """Convert the hyperresult_dict into a pandas dataframe, where each row is a different setting of the hyperparameters.
    The columns are the hyperparameters, the number of dimensions, the cost, and the PSM. If ndims is -1 then we assume
    it is associated with the target PSM. If it's less than -1, then we assume it was the output of a joint optimization step,
    and so we drop it."""
    value_cols = ['ndim', 'cost', 'time', 'PSM']
    df_rows = []
    for k,v in hyperresult_dict.items():
        for entry in v[0]:
            if filter_joint and entry[1] <= -2: # indicates that its an output of the joint optimization step, not the re-optimization step
                continue
            df_rows.append([*k, entry[1], entry[2], entry[3], entry[0]])
    return pd.DataFrame(df_rows, columns=column_order+value_cols)


def best_hyperparams_table(df):
    heat_map = {}
    for sim_metric in df['sim_metric'].unique():
        for energy_func in df['energy_metric'].unique():
            if not sim_metric in heat_map:
                heat_map[sim_metric] = []
            heat_map[sim_metric].append(df[(df['sim_metric'] == sim_metric) & 
                                        (df['energy_metric'] == energy_func) & 
                                        (np.isfinite(df['cost']))]['cost'].min())
    # print out as latex table
    # energy metric as headings
    print('Similarity Function & ' + ' & '.join([f"{x}" for x in df['energy_metric'].unique()]), '\\\\')
    for sim_metric, costs in heat_map.items():
        print(sim_metric, ' & '.join([f"{x:.2f}" for x in costs]), '\\\\')


def show_hyper_results(df, dset_name, sv_name=None, cost_col='cost'):
    """df is the ouptut of df_from_hyperresults. We plot the PSM for each energy function and similarity metric for each layer."""
    # 2d heatmap of cost_col where x-axis is energy_metric, y-axis is sim_metric. Do this for each energy_func
    min_cost_rows = df.loc[df.groupby(['layer', 'energy_metric', 'sim_metric'])[cost_col].idxmin()]
    # then, take the best from min_cost_rows grouped by layer and sim_metric
    min_min_cost = min_cost_rows.loc[min_cost_rows.groupby(['layer', 'sim_metric'])[cost_col].idxmin()]
    min_min_cost.drop(columns=['PSM'], inplace=True)
    print(min_min_cost)
    # find the row associated with the actual targets for each grouping of  (layer, energy_func, and ref_metric) (ndim will be -1)
    target_rows = df.loc[df.groupby(['layer', 'sim_metric'])['ndim'].idxmin()]

    # for each layer (subplots) and ref_metric, plot the PSM of each energy_func
    num_cols = len(df['energy_metric'].unique()) + 1
    num_rows = len(df['layer'].unique()) * len(df['sim_metric'].unique())
    # extract the reference PSMs by finding
    row_idxs = {(ref_metric,layer): i+1 
            for i,(ref_metric,layer) in enumerate(itertools.product(df['sim_metric'].unique(), df['layer'].unique()))}
    titles = [None]*(num_rows*num_cols)
    for layer in df['layer'].unique():
        for ref_metric in df['sim_metric'].unique():
            title_idx = (row_idxs[(ref_metric,layer)]-1)*num_cols
            titles[title_idx] = f"Target" 
            for k, energy_func in enumerate(df['energy_metric'].unique()):
                row = min_cost_rows[(min_cost_rows['layer'] == layer) &
                                    (min_cost_rows['sim_metric'] == ref_metric) & 
                                    (min_cost_rows['energy_metric'] == energy_func)]
                title_idx = (row_idxs[(ref_metric,layer)]-1)*num_cols + k + 1
                titles[title_idx] = f"{energy_func} | {row['cost'].values[0]:.2e}"

    # print(titles)
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=titles)
    # increase font size of subplot titles
    for i in range(1, num_rows*num_cols+1):
        fig.layout.annotations[i-1].update(font=dict(size=19))
    # print(num_rows, num_cols)

    invisible_font = {'color': 'rgba(0,0,0,0)'}
    for layer in df['layer'].unique():
        for ref_metric in df['sim_metric'].unique():
            # plot the target PDM
            row_idx = row_idxs[(ref_metric,layer)]
            ref_row = target_rows[(target_rows['layer'] == layer) & (target_rows['sim_metric'] == ref_metric)]
            fig.add_trace(go.Heatmap(z=ref_row['PSM'].values[0], coloraxis=f"coloraxis{row_idx}"), row=row_idx, col=1)
            fig.update_yaxes(title_text=f"Layer {layer} | {ref_metric}", row=row_idx, col=1, tickfont=invisible_font,
                            title_font=dict(size=16))
            fig.update_xaxes(tickfont=invisible_font, row=row_idx, col=1)
            # fig.update_layout(title_text='Target', row=row_idx, col=1)
            for k, energy_func in enumerate(df['energy_metric'].unique(), start=1):
                row = min_cost_rows[(min_cost_rows['layer'] == layer) & 
                                    (min_cost_rows['sim_metric'] == ref_metric) & 
                                    (min_cost_rows['energy_metric'] == energy_func)]
                fig.add_trace(go.Heatmap(z=row['PSM'].values[0], coloraxis=f"coloraxis{row_idx}"), row=row_idx, col=k+1)
                fig.update_xaxes(tickfont=invisible_font, row=row_idx, col=k+1)
                fig.update_yaxes(tickfont=invisible_font, row=row_idx, col=k+1)
                # fig.update_xaxes(title_text=energy_func, row=row_idx, col=k+1)
                # fig.update_layout(title_text=f"{energy_func} - {row[cost_col]}",  row=row_idx, col=k+1)
            # fig.update_yaxes(title_text=ref_metric, row=row_idx, col=k+1)
            full_row = min_cost_rows[(min_cost_rows['layer'] == layer) & (min_cost_rows['sim_metric'] == ref_metric)]
            cmin = full_row['PSM'].values[0].min()
            cmax = full_row['PSM'].values[0].max()
            abs_max = max(abs(cmin), abs(cmax))
            coloraxis_pos = dict(colorbar_len=5.2/(num_rows*7), 
                                  colorbar_y=6.75*(num_rows-row_idx+0.28)/(num_rows*6))
            if ref_metric == "RBF":
                coloraxis = dict(colorscale='Agsunset', cmin=cmin, cmax=cmax)
            else:
                coloraxis = dict(colorscale='RdBu', cmin=-abs_max, cmax=abs_max)
            coloraxis.update(coloraxis_pos)
            fig.update_layout(**{f"coloraxis{row_idx}": coloraxis})
    fig.update_layout(title_text=f"Target PSM vs. Abstract PSM on {dset_name}", font=dict(size=20),
                    showlegend=False)
    plot_size = 200
    fig.update_layout(height=plot_size*num_rows, width=plot_size*num_cols)
    # fig.show()
    if sv_name is not None:
        fig.write_image(sv_name)
    fig.show()


def run_hyper_experiment(coefs_dict, feature_set, params, min_dims, num_unique, fixed_params=None):
    """Given a set of coefficients, run a hyperparameter search over the parameters in `params`. For each entry in 
    fixed params, we will run a full hyperparameter search with the fixed parameters set over the range of values in
    `params`.
    :param coefs_dict: the dictionary of coefficients for each feature set
    :param feature_set: the feature set to use
    :param params: the hyperparameters to search over. It's a dictionary of lists of values
    :param min_dims: the minimum number of dimensions to consider when minimizing energy
    :param num_unique: the number of features that we should set as unique
    
    :param fixed_params: a list of dictionaries that are fixed parameters to use in the hyperparameter search."""
    if fixed_params is None:
        fixed_params = [{}]
    all_results = {}

    keys,values = zip(*params.items())
    settings = list(itertools.product(*values))
    np.random.shuffle(settings)
    for fixed in fixed_params:
        for bundle in tqdm(settings):
            d = dict(zip(keys, bundle)) 
            d.update(fixed)
            if 'energy' in d:
                energy_metric, energy_options = d.pop('energy')
            else:
                energy_metric, energy_options = d.pop('energy_metric'), d.pop('energy_options', None)

            # skip this settings since constrain => lamda_xi does nothing
            # if (d['constrain'] == True) and ('lamda_xi' in d) and ('lamda_xi' in params) and (d['lamda_xi'] != params['lamda_xi'][0]):
            #     continue
                
            d_key = d.copy()
            d_key['energy_func'] = energy_metric
            if energy_metric == 'riesz':
                d_key['energy_func'] = f"{energy_metric}_{energy_options['s']}"
            result_key = tuple(d_key.values())

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="delta_grad == 0.0")
                print("Running with options set to:", d_key)
                all_results[result_key] = find_energy_function(coefs_dict[feature_set], energy_metric=energy_metric, 
                                                            energy_options=energy_options, min_dims=min_dims,
                                                                num_unique=num_unique, **d)
    return all_results


def get_perturb_line(df, layer, sim_metric):
    """Given a dataframe of hyperparameter search results, extract the perturbation line for the given layer and sim_metric."""
    df = df[(df['layer'] == layer) & (df['sim_metric'] == sim_metric) & (np.isfinite(df['cost']))]
    perturb_line = df.loc[df.groupby(['perturb_pct'])['cost'].idxmin()]
    order = np.argsort(perturb_line['perturb_pct'])
    return perturb_line.iloc[order]


def show_perturb_graph(perturb_df, permuted_perturb_df, dset_name, sv_name=None):
    fig = make_subplots(rows=1, cols=len(perturb_df['sim_metric'].unique()), subplot_titles=perturb_df['sim_metric'].unique())
    # update font size
    for i in range(1, len(perturb_df['sim_metric'].unique())+1):
        fig.layout.annotations[i-1].update(font=dict(size=25))
    col_map = {sim_metric: i+1 for i,sim_metric in enumerate(perturb_df['sim_metric'].unique())}
    colors = ['red', 'blue', 'red', 'blue']
    color_map = {(sim_metric,layer): color 
                 for color, (sim_metric,layer) in
                   zip(colors, itertools.product(perturb_df['sim_metric'].unique(), perturb_df['layer'].unique()))}
    for sim_metric in perturb_df['sim_metric'].unique():
        for layer in perturb_df['layer'].unique():
            df = get_perturb_line(perturb_df, layer, sim_metric)
            perm_df = get_perturb_line(permuted_perturb_df, layer, sim_metric)
            fig.add_trace(go.Scatter(x=df['perturb_pct'], y=df['cost'], mode='lines', 
                                    name=f'Layer {layer}', marker_size=4,
                                    showlegend=(sim_metric=='RBF'),
                                    line=dict(color=color_map[(sim_metric,layer)])), row=1, col=col_map[sim_metric])
            fig.add_trace(go.Scatter(x=perm_df['perturb_pct'], y=perm_df['cost'], mode='lines', 
                                    name=f'{sim_metric} layer {layer}, permuted',
                                     showlegend=False,
                                    marker_size=4,
                                    line=dict(dash='dash', color=color_map[(sim_metric,layer)])), row=1, col=col_map[sim_metric])
            fig.update_xaxes(title_text='Perturbation Scale', row=1, col=col_map[sim_metric])

    fig.update_layout(title=f'{dset_name} Perturbation', yaxis_title='Distance to Target per Feature Pair', font=dict(size=18))
    fig.update_layout(legend=dict(font=dict(size=12)))
    if sv_name is not None:
        fig.write_image(sv_name)
    fig.show()