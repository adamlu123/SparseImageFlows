#!/usr/bin/env bash
act=${act}
latent=${latent}
result_fldr=${result_fldr}
source activate pytorch


cd /extra/yadongl10/BIG_sandbox/SparseImageFlows/SMAF

echo $PATH

python main.py --act ${act} --latent ${latent} --result_dir ${result_fldr}

