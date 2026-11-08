# EchoHate: Targeted LLM Supervision and Deferral in Encoder–LLM Collaborative Hate Speech Detection

Submitted to the ACL Rolling Review (ARR) October cycle.

Run commands from this repository root. Set `ECHOHATE_ROOT` if the process cwd is elsewhere.

```bash
python -m pipeline help
python -m pipeline.preprocessing --help
```

## Encoder

The encoder is RoBERTa-large. Every training example uses the ground-truth binary
label for cross-entropy and, when enabled, the pairwise cosine-margin contrastive
loss on the binary logits. A subset of training rows additionally supplies one
mechanism label to an auxiliary classification head.

Pass mechanism labels in a CSV (`--accepted_selected_k_csv`).

## Pipeline

```
raw data
  → preprocessing (fixed splits)
  → roberta_baseline (+ training_dynamics.csv)
  → select_hard_cases_for_mech (var25 or rand)
  → mechanism-label CSV for the selected rows
  → mech_distill
  → run_eval_test_from_checkpoint
```

| Stage | Module | Role |
|-------|--------|------|
| Preprocess | `pipeline.preprocessing` | Ingest 7 datasets; build `splitted_data_80_10_10` |
| Stage 2 | `roberta_baseline` | Binary fine-tune + learning dynamics |
| Stage 1 | `select_hard_cases_for_mech` | Top-variability or class-matched random rows |
| Stage 2 | `mech_distill` | CE + contrastive + mechanism auxiliary loss |
| Stage 2 | `run_eval_test_from_checkpoint` | Validation-tuned threshold, then one test eval |

## Mechanism-label CSV

`row_id` follows `train.csv` order: `train_0`, `train_1`, ...
`mechanism` must be one closed-list label for that dataset.

Hate datasets (DynaHate, HatEval, ETHOS, HateXplain, IHC, ToxiGen):

- Dehumanization
- Inferiority
- Group Vilification
- Threatening and Incitement
- In-group Victimhood Framing
- Support for Hateful Entities
- Not Offensive
- Offensive but No Protected-Group Attack Intent

SBIC (offensive vs non-offensive):

- Stereotyping and Implied Bias
- Lewdness
- Explicit Rudeness or Toxicity
- Neutral or Benign Speech
- Counter Speech

```bash
python -m pipeline stage2 mech_distill \
  --dataset IHC \
  --data_root data/<SEED> \
  --model_name roberta-large \
  --accepted_selected_k_csv path/to/mechanism_labels.csv \
  --lambda_mech 0.05 \
  --ce_contrastive_lambda 0.5 \
  --contrastive_margin 0.75
```

`--accepted_selected_k_csv` is the supported input. Rows in that file receive
the auxiliary loss. With `--lambda_mech > 0`, training stops if none of those
mechanisms match the inventory above.

## Configuration

[`config/echohate_project_config.json`](config/echohate_project_config.json) is loaded by
`pipeline.common.project_config`.

| Setting | Config key | Default |
|---------|------------|---------|
| Split seed | `create_split_random_state` | 42 |
| Training / selection seed | `random_state` | 42 |
| Default data bucket name | `default_data_seed_dir` | `seed_42_model` |

Keep one seed directory consistent across baseline, selection, and `mech_distill`.

Environment:

- `ECHOHATE_ROOT` — repository root
- `ECHOHATE_DATA_ROOT` / `ECHOHATE_DATA_SEED_DIR` — data layout overrides
- `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` — Hugging Face downloads, if the model is gated

## Install

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Or:

```bash
conda env create -f environment.yml && conda activate echohate
```

License: [`LICENSE`](LICENSE) (MIT). Citation: [`CITATION.cff`](CITATION.cff).

## CLI

```bash
python -m pipeline <stage> <module> [args…]
```

**Stage 1:** `select_hard_cases_for_mech`, `selection_variability_top25`,
`selection_balanced_random`  
**Stage 2:** `roberta_baseline`, `mech_distill`, `run_eval_test_from_checkpoint`

Preprocessing:

```bash
python -m pipeline.preprocessing <command>
```

Commands: `split`, `dynahate`, `ihc`, `ethos`, `sbic`, `hateval`,
`hatexplain-random-split-variants`, `toxigen`

**Split policy:** DynaHate and SBIC keep official splits where available; IHC merges
stages then random 80/10/10; other datasets follow each ingest script.

## Typical run

Replace `<SEED>` with the data bucket (for example `seed_0_mech`).

```bash
# 1. Preprocess
python -m pipeline.preprocessing ihc

# 2. RoBERTa-large baseline (writes training_dynamics.csv)
python -m pipeline stage2 roberta_baseline \
  --dataset IHC \
  --data_root data/<SEED> \
  --model_name roberta-large \
  --output_dir data/<SEED>/IHC/roberta_large_baseline_wCL \
  --ce_contrastive_lambda 0.5 \
  --track-dynamics

# 3. Highest-variability 25% within each class
python -m pipeline stage1 select_hard_cases_for_mech \
  --method var25 \
  --data-root data \
  --seed-dir <SEED> \
  --datasets IHC \
  --baselines roberta_large_baseline_wCL

# 4. Mechanism-supervised training
python -m pipeline stage2 mech_distill \
  --dataset IHC \
  --data_root data/<SEED> \
  --model_name roberta-large \
  --accepted_selected_k_csv path/to/mechanism_labels.csv \
  --output_dir data/<SEED>/IHC/roberta_large_baseline_wCL_encoder/encoder_fft_mech \
  --lambda_mech 0.05 \
  --ce_contrastive_lambda 0.5

# 5. Official test number: tune the threshold on validation, then evaluate test once
python -m pipeline stage2 run_eval_test_from_checkpoint --help
```

Tune checkpoints and the decision threshold on validation. Do not treat the
in-training `mech_distill` test metrics as the reported test number.

## Code layout

```
pipeline/
  common/           config, paths, hf auth, mechanism taxonomy
  preprocessing/    dataset ingest + split
  stage1/           variability and random selection
  stage2/           baseline, mech_distill, checkpoint eval
config/
  echohate_project_config.json
```

## Data layout

The caller provides these files. The repository does not ship them.

```
data/<SEED>/
  <Dataset>/
    splitted_data_80_10_10/
      train.csv  validation.csv  test.csv
    roberta_large_baseline_wCL/
      training_dynamics.csv
      selection_variability_top25/
        selected_hard_rows.csv
    roberta_large_baseline_wCL_encoder/
      encoder_fft_mech/
```

## Partial release

This repository is the encoder code: preprocessing, the RoBERTa baseline, variability-based example selection, and mechanism-supervised training. The LLM teacher (mechanism annotation) and uncertainty-guided deferral are withheld.

The withheld code will be released upon acceptance. This snapshot does not include datasets, mechanism labels, or cluster jobs. Those inputs are supplied by whoever runs the code.
