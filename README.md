# MO-SIMI Reproduction Package

This repository accompanies **Verified Self-Improving Learning of Large Language Models for Multi-Objective Point-Spec Circuit Design**. It contains the code, LoRA adapters, task definitions, simulator interfaces, training logs, and evaluation artifacts used for the MO-SIMI experiments on:

- DC--DC converters: buck, boost, SEPIC, and buck--boost;
- amplifier passive networks around predefined two-stage and RF active cores;
- oscillator passive networks around predefined LC, RC, ring, and Wien active cores.

The base Qwen2.5-7B model is not redistributed. Download it separately and pass its local path through `--base_model`. The amplifier and oscillator experiments do not synthesize complete active circuits from scratch.

## Repository layout

Each circuit-family directory follows the same layout:

```text
dcdc|amplifier|oscillator/
  1_eval_code/                 evaluation and SPICE-verification code
  2_train_code/                SFT, PVPO, Safe-PPO, and pipeline code
  3_eval_artifacts_latest/     archived evaluation outputs
  4_train_artifacts_latest/    adapters, logs, data, and run configurations
```

The files under `1_eval_code/` and `2_train_code/` intentionally include duplicate shared modules so that each directory can be used as a self-contained code snapshot. Archived `code_snapshot_*` directories preserve the implementation used by individual runs.

## Prerequisites

The reported training run used:

- Ubuntu 22.04.5 LTS;
- Python 3.12.3;
- PyTorch 2.7.0;
- PEFT 0.13.2;
- ngspice 36;
- Qwen2.5-7B as the base model;
- CUDA GPUs for SFT/PVPO/Safe-PPO and CPU workers for SPICE verification.

Install the Python dependencies in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
ngspice --version
```

NGSpice must be available on `PATH`. Training requires enough GPU memory for a 7B model with LoRA adapters. Evaluation also uses model inference, while the SPICE portion runs on CPUs.

## Quick-start evaluation

The repository includes executable wrappers under `scripts/`. To replay the final
DC--DC adapter, run:

```bash
bash scripts/quickstart_eval_dcdc.sh /path/to/Qwen2.5-7B
```

The following command evaluates the archived final DC--DC adapter with ten raw generations per task. Replace `/path/to/Qwen2.5-7B` with the local base-model directory.

```bash
cd dcdc/1_eval_code
python eval_dcdc_family.py \
  --base_model /path/to/Qwen2.5-7B \
  --adapter ../4_train_artifacts_latest/vpspi_tol001_20260111_005147_PURE_SFT_ANCHOR_LOOP_r40_8GPU176CPU/round_02/safe_ppo/ppo_best \
  --outdir ./replay_final_raw \
  --n_per_task 10 \
  --max_new_tokens 320 \
  --temperature 0.7 \
  --top_p 0.9 \
  --seed 2025 \
  --tol 0.01 \
  --min_elems 20 \
  --constrained \
  --no_fallback
```

Use `eval_amp_family.py` and `eval_osc_family.py` in the corresponding family directories for amplifier and oscillator evaluation. Run any evaluator with `--help` to list family-specific options.

## Quick-start training

The one-round wrapper takes the base model, anchor adapter, and the JSONL corpus
used for post-PVPO SFT regularization:

```bash
bash scripts/quickstart_train_dcdc.sh \
  /path/to/Qwen2.5-7B \
  /path/to/anchor_sft_adapter \
  /path/to/pvpo_sft.jsonl
```

The DC--DC orchestration entry point is `dcdc/2_train_code/run_vpspi_pipeline.py`. A one-round example is:

```bash
cd dcdc/2_train_code
python run_vpspi_pipeline.py \
  --base_model /path/to/Qwen2.5-7B \
  --anchor_adapter /path/to/anchor_sft_adapter \
  --run_root ./runs \
  --name mosimi_reproduction \
  --rounds 1 \
  --tol 0.01 \
  --min_elems 20 \
  --selfplay_n_gen 20 \
  --selfplay_max_rounds 4 \
  --selfplay_min_pass_cv 2 \
  --selfplay_eda_repair \
  --selfplay_repair_max_evals 36 \
  --dpo_epochs 3 \
  --ppo_steps 80 \
  --ppo_batch_size 8 \
  --ppo_group_size 4 \
  --ppo_lr 5e-6 \
  --ppo_target_kl 0.03 \
  --ppo_clip 0.2 \
  --ppo_max_new_tokens 320 \
  --ppo_temperature 0.7 \
  --ppo_top_p 0.9 \
  --train_gpus 1
```

The exact archived main-run command and all continuation commands are recorded in:

```text
dcdc/4_train_artifacts_latest/
  vpspi_tol001_20260111_005147_PURE_SFT_ANCHOR_LOOP_r40_8GPU176CPU/
    run_config.json
    run_config_resume_*.json
    logs/pipeline.log
```

Server-specific absolute paths in archived commands must be replaced with local paths. Set `--selfplay_sim_workers` and `--ppo_sim_workers` according to the effective CPU quota, not the host-wide logical CPU count.

## Training stages and artifacts

For each completed outer round, the pipeline records:

1. candidate netlists and simulator measurements under `round_*/selfplay_data/`;
2. preference pairs under `round_*/selfplay_data/dpo_pairs.jsonl`;
3. PVPO adapters under `round_*/pvpo_dpo/`;
4. Safe-PPO checkpoints and guard logs under `round_*/safe_ppo/`;
5. acceptance decisions and the selected model state in `round_*/round_state.json`.

The DC--DC main run completed three outer rounds (`round_00`--`round_02`). Grid77 is the development/self-improvement task grid used by those rounds; it is not a held-out test set.

## Evaluation conventions

- **OK**: the generated DSL is valid and the SPICE simulation completes.
- **AR**: all family-specific point specifications meet their tolerances.
- **CE**: all family-specific engineering constraints pass.
- **Acc**: `max(0, 1 - e_spec)`; invalid samples contribute zero in strict replay summaries.
- Canonical deduplication is performed within each task before duplicate-free metrics are reported.
- Raw/open-loop evaluation disables template fallback and post-generation search.

The numerical near-miss search used in strict replay edits only predeclared numeric passive values and, for DC--DC tasks, bounded duty variables. The historical training-time recovery pipeline also contains structural completion and passive-bank expansion; these operations should not be interpreted as unconstrained topology synthesis.

## Reproducing paper tables

Archived summary JSON files in `3_eval_artifacts_latest/` and per-round metric files in `4_train_artifacts_latest/` are the source records for the paper tables. Preserve task scope, tolerance, fallback setting, repair/search budget, and duplicate policy when comparing summaries. Results with different settings are not directly interchangeable.

## License and citation

Use the repository license for redistribution terms. Please cite the accompanying paper when using the code or artifacts.
