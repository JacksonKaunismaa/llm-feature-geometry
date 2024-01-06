#!/bin/bash
#SBATCH -o log/activation_all.log-%j
#SBATCH -c 4

module load gcc/9.3.0 arrow/13.0.0
source /home/jackk/.bashrc
source env_vars.sh
source /home/jackk/interp/bin/activate
module load gcc/9.3.0 arrow/13.0.0
# module list
# echo "hi"
# pip3 list
# which python3
# echo "hihlcroe"
module list
which python3
echo "INSTALLING"
pip3 install transformer_lens
# echo "HOME"
# ls /home/jackk
# echo "DEF-PAPYAN"
# ls /home/jackk/projects/def-papyan/jackk
# echo "SCRATCH"
# ls /home/jackk/scratch
# echo "PWD"
# pwd
# ls 
pip3 list
cd /home/jackk/sparse-probing-paper
# Assuming the datasets are in the ../superposition/sparse_datasets/ directory
datasets_dir="/home/jackk/projects/def-papyan/jackk/sparse_datasets"

# Iterate over each dataset in the directory
for dataset_path in $datasets_dir/*; do
	# Extract dataset name from the path
	dataset_name=$(basename $dataset_path)

	echo "DSET_PATH: $dataset_path"
	echo "DSET_name: $dataset_name"
	# check if it already exists in RESULTS_DIR, if so, skip
	if [ -d "$RESULTS_DIR/activation_datasets/pythia-1b-deduped/$dataset_name" ]; then
		echo "Skipping $dataset_name"
		continue
	fi
	# Run the Python command for each dataset
	python3 get_activations.py --model pythia-1b-deduped --experiment_type activation_probe_dataset --experiment_name activation_dataset --feature_dataset "$dataset_name" --skip_computing_token_summary_df
done

