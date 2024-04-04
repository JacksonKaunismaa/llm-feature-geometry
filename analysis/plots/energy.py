import numpy as np
import scipy.optimize
import einops
from functools import partial
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.metrics import pairwise_distances

from .geometry import find_good_ratio, get_grams
from .. import load_results
from .find_permutation import combined_sorting, poly_order_space




def plot_joint_energy_results(final_opts, feature_names=None, sv_name=None):
    """Plot the results of the joint optimization of the energy function. We plot the actual dist matrix, and the
    optimal dist matrix for each feature dimensionality."""
    if feature_names is None:
        feature_names = list(range(len(final_opts[0][0])))

    height,width = find_good_ratio(len(final_opts))
    fig = make_subplots(rows=height, cols=width, 
                        subplot_titles=['Actual dist matrix'] + [f"Optimal {ndims=}" for (_,ndims) in final_opts[1:]])
    for i, (final_opt,ndim) in enumerate(final_opts):
        fig.add_trace(go.Heatmap(z=final_opt, coloraxis='coloraxis', zmin=-1, zmax=1), row=i//width+1, col=i%width+1)

    fig.update_layout(coloraxis=dict(colorscale='RdBu', cmin=-1, cmax=1))
    fig.update_layout(height=300*height, width=300*width, title='dist Matrix comparison between optimal and actual feature dists')
    fig.update_xaxes(ticktext=feature_names, tickvals=np.arange(len(feature_names)), showticklabels=False)
    fig.update_yaxes(ticktext=feature_names, tickvals=np.arange(len(feature_names)), showticklabels=False)

    if sv_name is not None:
        fig.write_image(sv_name)
    fig.show()



def energy_objective(dists, weights, energy_func):
    """Given a dist matrix or a set of points, and a set of weights, compute the energy function that is a weighted
    sum of the exp of the dot products of the points."""
    weight_probs = weights/weights.sum()
    weights_outer = np.outer(weight_probs, weight_probs)
    return np.sum(energy_func(dists*weights_outer))


def objectives(x, targets, lamda_xi, lamda_w, lamda_diff, energy_func, energy_metric, reference_metric, weights, **kwargs):
    """Given a set of weights and points, compute the energy function, the weight regularization, and the dist difference.
    Return a dictionary of the costs in each category {cost_name: cost_value}."""
    n_feats = targets.shape[0]
    if weights is None:
        weights = x[:n_feats]
        points = x[n_feats:].reshape(n_feats, -1)
    else:
        points = x.reshape(n_feats, -1)
    dists = pairwise_distances(points, metric=reference_metric, **kwargs)
    if energy_metric == reference_metric:
        energy_dists = dists
    else:
        energy_dists = pairwise_distances(points, metric=energy_metric, **kwargs)
    
    # Objective:
    # 1. make the dist matrix of the points look like the dist matrix of the actual features
    # 2. minimize a weighted energy function of the points
    # 3. make the energy function look like a reasonable one by having the weights be close to 1
    costs = {}
    if lamda_diff is not None:   # difference with target dists
        dist_difference = ((dists-targets)**2).sum()
        costs['dist_diff'] = lamda_diff*dist_difference
    if lamda_xi is not None:   # regularize the norm of our xi
        xi_regularization = np.linalg.norm(points, axis=1)
        costs['xi_norm'] = lamda_xi*np.linalg.norm(xi_regularization)
    if lamda_w is not None:   # regularize the weights to be close to 1
        weight_regularization = np.linalg.norm(weights-1)
        costs['weight_norm'] = lamda_w*weight_regularization
    costs['energy'] = energy_objective(energy_dists, weights, energy_func)
    return costs


def objective(*args, **kwargs):
    """Given a set of weights and points, compute the sum of the objectives. This is the function that is minimized."""
    return sum(objectives(*args, **kwargs).values())


def on_sphere(x, n_feats, n_dims):
    """Constraint function that ensures that the points are on the sphere. Returns the norm of the points minus 1."""
    if x.shape[0] == n_feats*n_dims:  # assume it has no "weights" concatenated to it
        points = x.reshape(n_feats, n_dims)
    else:
        points = x[n_feats:].reshape(n_feats, n_dims)
    return np.linalg.norm(points, axis=1) - 1


def _find_energy_function(targets, n_dims, lamda_xi, lamda_w, lamda_diff, energy_func, reference_metric, energy_metric, constrain, 
                          weights=None, points=None, **kwargs):
    """Given a particular number of dimensions to put points in, jointly optimize the weights and points to minimize
    the objective that involves regularization on the weights and a weighted energy function"""
    # we will consider energy functions of the form sum exp(x_i.dot(x_j)*weight_j*weight_i)
    n_feats = targets.shape[0]
    fixed_weights = weights is None  # then assume we are in the first optimization step
    if not fixed_weights:
        weights = np.ones(n_feats)

        points = np.random.rand(n_feats, n_dims)
        points = points/np.linalg.norm(points, axis=1, keepdims=True)
        x0 = np.concatenate([weights, points.flatten()])
    else:
        lamda_diff = None  # we dont have targets to compare against, so disable the dist difference
        lamda_w = None   # weights are fixed so dont regularize them
        x0 = points.flatten()

    # print("shapes are targets", targets, "x0", x0.shape, "n_dims", n_dims, "weights", weights.shape, "points", points.shape)
    # we will regularize the weights to be close to 1

    if constrain:    
        constraint = {'type': 'eq', 'fun': partial(on_sphere, n_feats=n_feats, n_dims=n_dims)}
        method = 'trust-constr'
        lamda_xi = None
    else:
        constraint = {}
        method = 'L-BFGS-B'

    _objective = partial(objective, targets=targets, lamda_xi=lamda_xi, lamda_w=lamda_w, lamda_diff=lamda_diff, 
                         energy_func=energy_func, reference_metric=reference_metric, energy_metric=energy_metric, 
                         weights=(weights if fixed_weights else None), **kwargs)
    result = scipy.optimize.minimize(_objective, x0.flatten(),
                                     method=method, constraints=constraint, options={'disp': False})
    # print(f"Final results of {n_dims} opt (energy, weight, dist_diff)", objectives(result.x, dist_matrix, lamda, lamda_diff))
    # print("Weights were:", result.x[:dist_matrix.shape[0]])
    # weights = result.x[:masked_dists.shape[0]]
    # points = result.x[masked_dists.shape[0]:].reshape(masked_dists.shape[0], n_dims)
    # print("For n_dims", n_dims, "objective value was", result.fun)
    return result


def extract_targets(coefs_or_dists, dist_func, dist_funcs, layer, **kwargs):
    # if its a dictionary, assume we have to extract the coefs ourselves and compute the dists
    # kwargs gets fed to pairwise_distances
    if isinstance(coefs_or_dists, dict):  # {feat_name: coefs ndarray(layers, hidden_size)}
        reference_dists = load_results.extract_feats_and_concatenate(coefs_or_dists)[layer] # (n_feats, hidden_size)
        reference_dists = pairwise_distances(reference_dists, metric=dist_func, **kwargs)
    else:  # if its already an ndarry, assume that it is the dists
        reference_dists = coefs_or_dists
    return reference_dists


def mask_targets(targets, thresh):
    # mask out the target dists if they are very similar. Our energy function assumes that they should not be copies of each other
    # if a set of n features have a dot product greater than thresh, then n-1 one of them can be removed, since they are redundant
    # we compute this by seeing where in the upper triangle is dot product greater than thresh

    n = targets.shape[0]

    if thresh >= 1:
        print("Disabling duplicate feature dropping, since thresh is >= 1")
        return targets, list(range(n))
    
    # get off diagonal entries that are very similar
    similar_entries = np.nonzero(np.triu((targets < thresh), k=1))
    # eg. if {1, 5, 10, 12}, {2, 8}, and {4, 9} are the redundant sets, we should remove {5, 10, 12, 8, 9},
    #        nonzero will be ([1, 1, 1, 2, 4, 5, 5, 10], [5, 10, 12, 8, 9, 10, 12, 12]) => we can take set(nonzero[1])
    redundant_features = set(similar_entries[1])
    # take away the redundant features
    good_features = list(sorted(set(range(n)) - redundant_features))
    # take the subset of the dists that are not redundant
    masked_dists = targets[good_features, :][:, good_features]
    return masked_dists, good_features


def find_energy_function(coefs_or_dists, layer=8, thresh=0.8, min_dims=2, lamda_diff=0.1, 
                         reference_metric="cosine", energy_func=np.exp, energy_metric="cosine",
                         constrain=False, lamda_xi=0.5, lamda_w=0.2,
                         mapper='poly_order_space', **kwargs):
    """Given a set of coefficients or dists, find the optimal set of features that will minimize the energy function. 
    We do a joint optimization of the weights and points to minimize the energy for each feature dimensionality, which 
    ranges from min_dims to the number of features.

    :param layer: the layer to consider
    :param thresh: the threshold for similarity between features
    :param min_dims: the minimum number of dimensions to consider when minimizing energy
    :param lamda_diff: regularization for the dists to be close to the reference dists
    :param reference_metric: the function that is applied to the x_i to generate a pairwise matrix to 
                    compare to the actual feature dists. Must be supported by sklearn.metrics.pairwise_distances
    :param energy_metric: the pairwise metric to use for the energy function. Must be supported by 
                          sklearn.metrics.pairwise_distances
    :param energy_func: the function that is applied to the dists and then summed to minimize interference
    :param constrain: whether to use a hard constraint for a sphere, or a regularization parameter `lamda`
    :param lamda_xi: regularization paramater to impose bounded, non-zero norm on x_i. Has no effect if constrain=True.
    :param lamda_w: regularization parameter for the weights to be close to 1
    :param mapper: the function that is used in the sorting based permutation finder
    :param **kwargs: extra arguments to be fed to pairwise_distances.
    :returns: a list of tuples, where each tuple is the optimal distance matrix and the number of dimensions used to
              generate that distance matrix. The first element is the actual distance matrix."""

    # 1. -log(dist) replace this with the exp(dot)  (or maybe Riesz kernel)
    # 2. use smaller feature subset
    # 3. use average activation features
    # 4. check if re-running optimization changes things
    # 5. investigate how changing lamda_diff and lambda changes things
    # 6. double check that layer 8 is the same as everything assumption is good
    
    unmasked_targets = extract_targets(coefs_or_dists, reference_metric, layer, **kwargs)
    masked_targets, good_features = mask_targets(unmasked_targets, thresh, reference_metric, **kwargs) # (n_feats, n_feats)

    n_feats = masked_targets.shape[0]
    final_opts = [(masked_targets, -1)]
    for n_dims in range(min_dims, n_feats+1):   # everything past n_feats+1 is just a simplex anyway
        weights, points = _find_energy_function(masked_targets, n_dims, lamda_xi/n_dims, lamda_w, lamda_diff, 
                                                energy_func, reference_metric, energy_metric, constrain, 
                                                **kwargs)
        # now reoptimize just the energy objective, with weights fixed
        _, reopt_points = _find_energy_function(masked_targets, n_dims, lamda_xi/n_dims, lamda_w, lamda_diff, 
                                                energy_func, reference_metric, energy_metric, constrain, 
                                             points=points, weights=weights, **kwargs)
        # undo any permutations that were learned
        opt_dist = pairwise_distances(reopt_points, metric=reference_metric, **kwargs)
        permutation = combined_sorting(opt_dist, masked_targets, mapper)
        opt_dist = opt_dist[permutation,:][:,permutation]
        # save the distance matrix and the number of dimensions
        final_opts.append((opt_dist, n_dims))

    return final_opts, good_features