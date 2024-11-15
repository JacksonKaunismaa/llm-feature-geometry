# Energy Minization in LLM MLP features
Code repository for my undergraduate thesis "An Investigation into Energy Minimization Properties of MLP Features in LLMs", working with Professor Vardan Papyan.

Infrastructure for feature exctraction based upon [Finding Neurons in a Haystack: Case Studies with Sparse Probing](https://arxiv.org/abs/2305.01610)

## Organization
The code for computing energy minimization plots for actual features can be found at `analysis/plots/energy.py`. The optimal feature test case, as well as some helper functions, can be found at `analysis/plots/geometry.py`. The code for permutating the Gram matrices can be found at `analysis/plots/find_permutation.py`.

Apart from some minor changes to infrastructure and a helper function or two, those are the primary changes introduced in this repository compared to the original paper.

To browse over already computed examples (likely confusingly ordered), you can also look at `notebooks/explore.ipynb`.



### Getting started
Create virtual environment and install required packages
```
git clone https://github.com/JacksonKaunismaa/sparse-probing-paper.git
cd sparse-probing
pip install virtualenv
python -m venv sparprob
source sparprob/bin/activate
pip install -r requirements.txt
```

Acquire Gurobi [license](https://www.gurobi.com/features/academic-named-user-license/). Free for academics. Make sure you are on campus wifi (you may also need to seperately install [grbgetkey](https://support.gurobi.com/hc/en-us/articles/360059842732)).

### Environment variables
To enable running our code in many different environments we use environemnt variables to specify the paths for all data input and output. For examples
```
export RESULTS_DIR=/Users/wesgurnee/Documents/mechint/sparse_probing/sparse-probing/results
export FEATURE_DATASET_DIR=/Users/wesgurnee/Documents/mechint/sparse_probing/sparse-probing/feature_datasets
export TRANSFORMERS_CACHE=/Users/wesgurnee/Documents/mechint/sparse_probing/sparse-probing/downloads
export HF_DATASETS_CACHE=/Users/wesgurnee/Documents/mechint/sparse_probing/sparse-probing/downloads
export HF_HOME=/Users/wesgurnee/Documents/mechint/sparse_probing/sparse-probing/downloads
```


## Citation
Citation for original paper:
```
@article{gurnee2023finding,
  title={Finding Neurons in a Haystack: Case Studies with Sparse Probing},
  author={Gurnee, Wes and Nanda, Neel and Pauly, Matthew and Harvey, Katherine and Troitskii, Dmitrii and Bertsimas, Dimitris},
  journal={arXiv preprint arXiv:2305.01610},
  year={2023}
}
```
