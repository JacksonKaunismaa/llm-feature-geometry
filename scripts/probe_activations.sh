#!/bin/bash
#SBATCH -o log/activation_all.log.%j-%a
#SBATCH -c 24
#SBATCH -a 1-16
#SBATCH --mem-per-cpu=4G
#SBATCH -N 1


source /home/jackk/.bashrc
source /home/jackk/.venv/interp/bin/activate
# source env_vars.sh

module list
# echo "hi"
# pip3 list
which python3
# echo "hihlcroe"
module load gcc/9.3.0 arrow/13.0.0
module load gurobi/10.0.3
module list
source env_vars.sh
# which python3
# echo "HOME"
# ls /home/jackk
# echo "DEF-PAPYAN"
# ls /home/jackk/projects/def-papyan/jackk
# echo "SCRATCH"
# ls /home/jackk/scratch
# echo "PWD"
# pwd
# ls
which python3
pip3 install --no-index scikit_learn
cd /home/jackk/gurobi_install
python3 setup.py build install
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
        if [ -d "$RESULTS_DIR/initial_probes/pythia-1b-deduped/$dataset_name" ]; then
                echo "Skipping $dataset_name"
                continue
        fi
        # Run the Python command for each dataset
        echo "about to call python"
        python3 probing_experiment.py --feature_dataset "$dataset_name" --experiment_type optimal_sparse_probing --experiment_name initial_probes --model pythia-1b-deduped
done

