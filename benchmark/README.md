To test a particular method on the benchmark, run with:

```bash
python evaluate_method.py [--method <method>] [--dataset <dataset>] [--seed <seed>]
                          [--version <version>] [--test]
```

Where:
- `method` is the folder name of a particular method in `official_competitors`,
- `dataset` is one of `hubble`, `kepler`, `newton`, `tully`, `leavitt`, `schechter`,
`bode`, `ideal_gas`, `planck`, `rydberg`,
- `seed` is an integer,
- and `version` is any string or integer to indicate the run's version.

The number of cores is set with `OMP_NUM_THREADS`. However, some methods do not use
this to select their compute allocation, so you should execute this command with, for example,
Slurm's `srun` command, so that the method is limited to 8 cores at most.

Methods should automatically exit themselves before 60 minutes, but, just in case, the method's search
duration is recorded as an output.
