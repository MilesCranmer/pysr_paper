# How to run a benchmark

First, you can load the conda environments with:

```bash
cp miniforge3.tar /dev/shm/ && \
    cp nix-portable.tar /dev/shm/ && \
    cd /dev/shm && \
    tar -xf miniforge3.tar && \
    tar -xf nix-portable.tar
```

1. `conda_activate && conda activate {method}`
2. `python evaluate_method.py --method {method} --dataset {dataset}`

This works for most methods. 

However, `operon` needs special treatment. To run `operon`, use the following instead:

1. Initialize with `mkdir -p /tmp/operon && cp official_competitors/operon/flake.nix /tmp/operon/ && export OLDDIR=$(pwd) && cd /tmp/operon`.
2. Run with: `nix-portable nix develop -i -c /bin/bash -c 'export HOME=/mnt/home/mcranmer && cd ~/pysr_paper_syw/srbench-comp && python evaluate_method.py --method operon --dataset {dataset}'`
3. Back to the directory with `cd $OLDDIR`.

Finally, you can clean up with (if they have changed)

```bash
tar -cvf miniforge3.tar miniforge3 && mv -i miniforge3.tar ~/
tar -cvf nix-portable.tar .nix-portable && mv -i nix-portable.tar ~/
```
