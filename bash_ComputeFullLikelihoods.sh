#!/bin/bash

for IGMF in 0 1 2 3 4 5 6 7 8; do
#for IGMF in  1 5; do
    srun sbatch batch_ComputeFullLikelihoods.job $IGMF
done

#done
