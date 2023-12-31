#!/bin/bash
#SBATCH -o log/activation_all.log-%j
#SBATCH -c 4
#SBATCH --gres=gpu:1

source env_vars.sh
cd /home/jackk/sparse-probing-paper
# Assuming the datasets are in the ../superposition/sparse_datasets/ directory
datasets_dir="../superposition/sparse_datasets"

# Iterate over each dataset in the directory
for dataset_path in "$datasets_dir"/*; do
	# Extract dataset name from the path
	dataset_name=$(basename "$dataset_path")
	# Run the Python command for each dataset
	python3 get_activations.py --model pythia-1b-deduped --experiment_type activation_probe_dataset --experiment_name activation_dataset --feature_dataset "$dataset_name" --skip_computing_token_summary_df
done

