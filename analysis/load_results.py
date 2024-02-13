import os
import json
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
import re


def load_probing_experiment_results(results_dir, experiment_name, model_name, dataset_name, inner_loop, uncollapse_features=False):
    result_dir = os.path.join(
        results_dir, experiment_name, model_name, dataset_name, inner_loop)
    results = {}
    config = None
    for result_file in os.listdir(result_dir):
        if result_file == 'config.json':
            config_file = os.path.join(result_dir, result_file)
            config = json.load(open(config_file, 'r'))
            continue
        # example: heuristic_sparsity_sweep.arxiv.pythia-125m.mlp,hook_post.max.0.p
        _, feature, _, hook_loc, aggregation, layer, _ = result_file.split('.')
        layer = int(layer)
        hook_loc = hook_loc.replace(',', '.')
        results_dict = pickle.load(
            open(os.path.join(result_dir, result_file), 'rb'))
        if uncollapse_features:  # --save_features_together enabled
            for k, v in results_dict.items():
                results[(f'{k}', layer, aggregation, hook_loc)] = v
        else:
            results[feature, layer, aggregation, hook_loc] = results_dict
    return results, config


def load_probing_experiment_results_old(results_dir, experiment_name, inner_loop, model_name):
    # old version
    result_dir = os.path.join(
        results_dir, experiment_name, inner_loop, model_name)
    results = {}
    for result_file in os.listdir(result_dir):
        if len(result_file.split('.')) == 5:
            _, feature, _, layer, file_type = result_file.split('.')
        else:
            continue
            print(result_file)
            _, feature, probe_loc,  _, layer, file_type = result_file.split(
                '.')
        layer = int(layer[1:])
        if feature not in results:
            results[feature] = {}
        results[feature][layer] = pickle.load(
            open(os.path.join(result_dir, result_file), 'rb'))
    return results


def make_heuristic_probing_results_df(results_dict):
    flattened_results = {}
    for feature in results_dict:
        for layer in results_dict[feature]:
            for sparsity in results_dict[feature][layer]:
                flattened_results[(feature, layer, sparsity)
                                  ] = results_dict[feature][layer][sparsity]
    rdf = pd.DataFrame(flattened_results).T.sort_index().rename_axis(
        index=['feature', 'layer', 'k'])
    return rdf


def collect_monosemantic_results(probing_results):
    dfs = {}
    for k, result in probing_results.items():
        dfs[k] = pd.DataFrame(result).T
    rdf = pd.concat(dfs)  # .reset_index()
    rdf.rename_axis(
        index=['feature', 'layer', 'aggregation', 'hook_loc', 'neuron'],
        inplace=True
    )
    return rdf.sort_index()




# our version of loading stuff

def get_probe_result(results_dir, model_name, experiment_name, probe_type, probe_loc='mlp,hook_post.none',
                     subtract_global_mean=False, num_layers=16):
    """Load the results of a probing experiment. Returns {dset: {feat_name: ndarray of coefs (num_layers, hidden_size)}}"""
    data_path = os.path.join(results_dir, experiment_name, model_name)
    pattern = f"{probe_type}\.(.+?)\.{model_name}\.{probe_loc}\.(\d+)\.p"
    dset_name_pattern = f"{model_name}\/(.+?)\/{probe_type}"
    results = {} # dset -> feat_type -> (layer, hidden_dim)
    all_coefs = defaultdict(list)  # layer -> list of coefs
    for dset, _, files in os.walk(data_path):
        for f in files:
            m = re.match(pattern, f)
            if m:
                dset_name = re.search(dset_name_pattern, dset).group(1)
                feat_name, layer = m.groups()
                layer = int(layer)
                if dset_name not in results:
                    results[dset_name] = {}
                with open(os.path.join(dset, f), 'rb') as p:
                    coef = pickle.load(p)['coef']
                    if feat_name not in results[dset_name]:
                        results[dset_name][feat_name] = np.zeros((num_layers, coef.shape[0]))  # so that the order will always be the same
                    # print('loading ', osp.join(dset, f), 'to ', dset_name, feat_name, layer)
                    results[dset_name][feat_name][layer] = coef
                    all_coefs[layer].append(coef)
                    
    if subtract_global_mean:
        mean_coefs = {layer: np.mean(all_coefs[layer]) for layer in all_coefs}
        for dset in results:
            for feat_name in results[dset]:
                for layer in range(num_layers):
                    results[dset][feat_name][layer] -= mean_coefs[layer]
    return results

def coefs_to_numpy(results_dict):
    """given {dset: {feat_name: coefs ndarray(layers, hidden)}} return ndarray(layers, feats, hidden)
    OR, given {feat_name: ndarray of coefs (layers, hidden)} return ndarray(layers, feats, hidden)"""
    first_elem = next(iter(results_dict.values()))
    if isinstance(first_elem, dict):
        return np.stack([coefs for dset in results_dict.values() for coefs in dset.values() ], axis=1)
    elif isinstance(first_elem, np.ndarray):  # given {feat_name: ndarray of coefs (num_layers, hidden_size)}
        return np.stack([coefs for coefs in results_dict.values()], axis=1)  # return ndarray of (num_layers, num_feats, hidden_size)
