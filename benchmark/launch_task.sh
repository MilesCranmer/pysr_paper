#!/bin/zsh
LMOD_CMD="/mnt/sw/nix/store/x240zci3c4y5kbr6yw2qcfpfcydda11q-lmod-8.6.5/lmod/lmod/libexec/lmod"
module () {
    eval $($LMOD_CMD zsh "$@") && eval $(${LMOD_SETTARG_CMD:-:} -s sh)
}
module purge && module load modules/2.0-20220630 cuda/11.4.4 cudnn/8.2.4.15-11.4 gcc/10.3.0 python/3.10.4 ffmpeg/4.4.1-nix imagemagick git/2.35.1 texlive singularity zsh/5.8 disBatch/beta

# If $1 is `operon`, we run differently:
if [[ "$1" = "operon" ]]; then
    # Operon is a special case, because it has a different launch strategy.
    export key="build_operon$RANDOM$RANDOM$RANDOM"
    export NP_LOCATION="/dev/shm/"
    export build_dir="/dev/shm/$key/"
    mkdir $build_dir
    cd $build_dir
    cp ~/pysr_paper_syw/benchmark/official_competitors/operon/flake.nix ./
    /mnt/home/mcranmer/bin/nix-portable nix develop -i --no-write-lock-file -c /bin/sh -c 'export HOME=/mnt/home/mcranmer && cd ~/pysr_paper_syw/benchmark && python evaluate_method.py '"--method operon --dataset $2 --seed $3 --version $4"
else

    if [[ -d /dev/shm/miniforge3 ]]; then
        echo "miniforge3 exists; will load."
    else
        echo "miniforge3 does not exist!"
        exit 1
    fi

    . "/dev/shm/miniforge3/etc/profile.d/conda.sh"
    conda activate
    conda activate $1
    cd /mnt/home/mcranmer/pysr_paper_syw/benchmark
    PYTHON=/dev/shm/miniforge3/envs/$1/bin/python
    $PYTHON evaluate_method.py --method $1 --dataset $2 --seed $3 --version $4
fi