#!/bin/bash

version=$1
LOCAL=${2:-0}

if [[ $LOCAL -eq "0" ]]; then
    echo '#DISBATCH PERENGINE START { cp ~/miniforge3.tar /dev/shm/ ; cp ~/nix-portable.tar /dev/shm/ ; cd /dev/shm ; tar -xf miniforge3.tar ; tar -xf nix-portable.tar ; } &> logs/engine_start_${SLURM_JOBID}_${DISBATCH_JOBID}_${DISBATCH_ENGINE_RANK}.log'
fi


# Working codes:
# - pysr
# - eql
# - gpzgd
# - uDSR
# - E2ET
# - QLattice
# Broken codes:
# - operon
#   - Installs nix-portable in home
# - Bingo
#   - Issues with MPI.
#   - (Can run locally though; without srun.)
# - nsga-dcgp
#   - Outputs too many equations.


codes=( E2ET PS-Tree )
datasets=( hubble kepler newton tully leavitt schechter bode ideal_gas planck rydberg )

for seed in {1..5}; do
    for dataset in ${datasets[@]}; do
        for code in ${codes[@]}; do
            echo "./launch_task.sh $code $dataset $seed $version 2>&1 &> logs/trial_${code}_${dataset}_${seed}_\${SLURM_JOBID}_\${DISBATCH_JOBID}.log"
        done
    done
done

# For debugging only:
# echo "sleep infinity"