The model checkpoints, example outputs, and preprocessed data are available at https://zenodo.org/record/8012171.
This repository contains only [the COVID-Fact dataset](https://github.com/asaakyan/covidfact) converted to the jsonline format.
You can download the other datasets from the Zenodo's URL above.

## Getting started with toy examples

You can try input reduction with the provided model checkpoints and toy examples.
For example, first download a model checkpoint and uncompress it:
```shell
conda activate pathologies
cd pathologies/experiments/
wget https://zenodo.org/record/8012171/files/covidfact_cross_entropy.tar.gz
tar xvfz covidfact_cross_entropy.tar.gz
```

Then, run:
```shell
python ../reduce.py --checkpoint_file covidfact_cross_entropy/roberta-base-128-mod/checkpoints/epoch\=9-step\=2040-val_acc\=0.8329.ckpt --in_file data/toy/covidfact1.jsonl --gpus 1
```

If everything works properly, you should see something like:
```shell
num_labels: 2
Create features from: data/toy/covidfact1.jsonl
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 318.23it/s]
Reduce: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.07s/it]
Input reduction took '0:00:06.724095'
Saved reduced inputs to data/toy/covidfact1.reduced.jsonl
```

The output file is in the jsonline format:
```shell
cat data/toy/covidfact1.reduced.jsonl
{"evidence": "CONCLUS IONS : In our cohort of CO VID - 19 patients , immun os upp ression was associated with a lower risk of moderate - severe AR DS .", "evidence_tokens": "CONCLUS IONS Ġ: ĠIn Ġour Ġcohort Ġof ĠCO VID - 19 Ġpatients , Ġimmun os upp ression Ġwas Ġassociated Ġwith Ġa Ġlower Ġrisk Ġof Ġmoderate - severe ĠAR DS .", "orig_claim": "Imm un os upp ression is associated with a lower risk of moderate to severe acute respiratory distress syndrome in cov id - 19 .", "orig_claim_tokens": "Imm un os upp ression Ġis Ġassociated Ġwith Ġa Ġlower Ġrisk Ġof Ġmoderate Ġto Ġsevere Ġacute Ġrespiratory Ġdistress Ġsyndrome Ġin Ġcov id - 19 Ġ.", "orig_conf": "1.000", "claim": "upp moderate respiratory .", "claim_tokens": "upp Ġmoderate Ġrespiratory Ġ.", "pred_label": "S", "conf": "0.999"}
{"evidence": "CONCLUS IONS : In our cohort of CO VID - 19 patients , immun os upp ression was associated with a lower risk of moderate - severe AR DS .", "evidence_tokens": "CONCLUS IONS Ġ: ĠIn Ġour Ġcohort Ġof ĠCO VID - 19 Ġpatients , Ġimmun os upp ression Ġwas Ġassociated Ġwith Ġa Ġlower Ġrisk Ġof Ġmoderate - severe ĠAR DS .", "orig_claim": "Imm un os upp ression is associated with a higher risk of moderate to severe acute respiratory distress syndrome in cov id - 19 .", "orig_claim_tokens": "Imm un os upp ression Ġis Ġassociated Ġwith Ġa Ġhigher Ġrisk Ġof Ġmoderate Ġto Ġsevere Ġacute Ġrespiratory Ġdistress Ġsyndrome Ġin Ġcov id - 19 .", "orig_conf": "0.999", "claim": "is associated", "claim_tokens": "Ġis Ġassociated", "pred_label": "R", "conf": "0.904"}
```

The above examples correspond to Figure 1 in [our paper](https://aclanthology.org/2023.findings-acl.730.pdf), where we only apply input reduction to the claim. 

In addition, we can check the reduction path (as shown in Figure 2 in our paper) by adding `--print_reduction_path --batch_size 1`:
```shell
python ../reduce.py --checkpoint_file covidfact_cross_entropy/roberta-base-128-mod/checkpoints/epoch\=9-step\=2040-val_acc\=0.8329.ckpt --in_file data/toy/covidfact2.jsonl --print_reduction_path --batch_size 1 --gpus 1
```

You should see:
```shell
num_labels: 2
Create features from: data/toy/covidfact2.jsonl
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 188.99it/s]
Reduce:   0%|                                                                                                                      | 0/1 [00:00<?, ?it/s]
(0.999) R aspberry pi about to avoid vent il ators for coron av irus victims
(0.999) aspberry pi about to avoid vent il ators for coron av irus victims
(0.999) pi about to avoid vent il ators for coron av irus victims
(0.999) pi about to avoid vent ators for coron av irus victims
(0.997) pi about to avoid vent ators for av irus victims
(0.995) about to avoid vent ators for av irus victims
(0.997) to avoid vent ators for av irus victims
(0.989) avoid vent ators for av irus victims
(0.986) vent ators for av irus victims
(0.988) ators for av irus victims
(0.989) ators for av irus
Reduce: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.34s/it]
Input reduction took '0:00:03.330434'
Saved reduced inputs to data/toy/covidfact2.reduced.jsonl
```

:warning: In the current implementation, the `--print_reduction_path` option only works when setting `--batch_size 1`.


## Training, prediction, and evaluation

Each subdirectory name in `experiments/` is roughly in the format of `[dataset]_[method]`, which will be parsed by the scripts inside it.
Each subdirectory contains ordered scripts like `01_train.sh`, `02_xxx.sh`, etc. 
If you run them sequentially, you should be able to reproduce the results in the paper.

Some experiments require outputs from other experiments.
For example, `covidfact_input_reduction+confidence_penalty` (i.e., $L_\widetilde{cp}$ on COVID-Fact) requires the reduced data created by `covidfact_input_reduction`, which in turn needs the model from `covidfact_cross_entropy` (i.e., $L_{ce}$ on COVID-Fact).
The same flow applies to other datasets.

You can skip data creation by downloading the preprocessed data from [here](https://zenodo.org/record/8012171/files/data.tar.gz) and the baseline model from [here](https://zenodo.org/record/8012171/files/covidfact_cross_entropy.tar.gz).
Then, uncompress and place them in `experiments/`.
You should see `{train,dev,test}.jsonl` as well as `train.reduced.jsonl`:
```shell
wc -l data/covidfact/*.jsonl
    419 data/covidfact/dev.jsonl
    404 data/covidfact/test.jsonl
   3263 data/covidfact/train.jsonl
   3263 data/covidfact/train.reduced.jsonl
   7349 total
```


### Step 1: Train

```shell
cd covidfact_input_reduction+confidence_penalty
sbatch [option] 01_train.sh
```

The script `01_train.sh` aims to work with the `sbatch` command on Slurm.
It can also work with the normal `sh` command with a slight modification (or no modification in some cases).


This script submits four jobs to Slurm to train four models for $\beta$ = {0.05, 0.1, 0.3, 0.5}.
We can use `sh` instead of `sbatch` by manually adding `SLURM_ARRAY_TASK_ID=1` (or `2`, `3`, `4`).
Each corresponds to the line number in `experiments/beta.txt`.

```bash
SLURM_ARRAY_TASK_ID=1
beta=$(sed -n "$SLURM_ARRAY_TASK_ID"p '../beta.txt')
```


### Step 2: Predict on the dev set

```shell
sbatch [option] 02_dev.sh
```

This script submits four jobs to Slurm to run predictions with the four models on the dev set.
Like `01_train.sh`, you can modify it to work with `sh`.


### Step 3: Find an optimal $\beta$

```shell
sh 03_best_beta.sh
```

This script uses `sh` and needs no GPU.
It calls `find_best_beta.py` that attempts to parse the evaluation files and saves an optimal $\beta$ value to `best_beta.txt`.
The current heuristic is to choose $\beta$ that yields a lower ECE score than the baseline model (`covidfact_cross_entropy`) while maintaining a high accuracy score.


### Step 4: Predict on the test set

```shell
sbatch [option] 04_test.sh
```

This script submits only one job to Slurm, which runs a prediction on the test with the model having the best $\beta$.


### Step 5: Run input reduction on the test set

```shell
sbatch [option] 05_reduce_on_test.sh
```

This script runs input reduction on the test set.

### Step 6: Evaluate

```shell
sh 06_eval.sh
```

Finally, we evaluate the results using several metrics. If everything works properly, we should see something like:
```
Saved reliability diagram to roberta-base-128-0.05-out/eval.test.ece.pdf
Saved confidence histogram to roberta-base-128-0.05-out/eval.test.conf.pdf
Saved results to roberta-base-128-0.05-out/eval.test.txt
Avg. claim length after reduction: 6.2
    S    R
S  95   35
R  37  237
          S     R
Prec:  72.0  87.1
Rec:   73.1  86.5
F1:    72.5  86.8
ACC: 82.2
ECE: 13.5
```
