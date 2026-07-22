# MO-SIMI Reproduction Package

This repository accompanies **Verified Self-Improving Learning of Large Language Models for Multi-Objective Point-Spec Circuit Design**. It contains source snapshots, task definitions, simulator interfaces, training components, and runnable wrappers for MO-SIMI experiments on:

- DC--DC converters: buck, boost, SEPIC, and buck--boost;
- amplifier passive networks around predefined two-stage and RF active cores;
- oscillator passive networks around predefined LC, RC, ring, and Wien active cores.

The base Qwen2.5-7B model and the archived LoRA checkpoints are not redistributed. Download the base model separately and pass local model and adapter paths to the wrappers. The amplifier and oscillator experiments do not synthesize complete active circuits from scratch.

## Repository layout

The public source layout is:

```text
dcdc/code/                     DC--DC evaluation, recovery, and training code
amplifier/code/                amplifier-family verifier and training code
oscillator/code/               oscillator-family verifier and training code
scripts/                       quick-start wrappers
requirements.txt               Python dependency versions
```

Each family directory is a self-contained source snapshot and therefore contains some shared modules. Full training logs, generated candidates, and LoRA checkpoints are substantially larger than the source repository; the paper and Supplementary Information report the audited configurations and results.

## Prerequisites

The reported training run used:

- Ubuntu 22.04.5 LTS;
- Python 3.12.3;
- PyTorch 2.7.0;
- PEFT 0.13.2;
- ngspice 36;
- Qwen2.5-7B as the base model;
- CUDA GPUs for SFT/PVPO/Safe-PPO and CPU workers for SPICE verification.

The archived training pipeline used ngspice 36. The controlled CPU replay reported in the revised manuscript uses NGSpice 46; simulator version, verifier settings, task scope, and tolerance must be held fixed when reproducing or comparing those replay rows.

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

The repository includes executable wrappers under `scripts/`. To evaluate a
DC--DC adapter, run:

```bash
bash scripts/quickstart_eval_dcdc.sh \
  /path/to/Qwen2.5-7B \
  /path/to/final_adapter
```

The equivalent direct command generates ten raw candidates per task. Replace both model paths with local directories.

```bash
cd dcdc/code
python eval_dcdc_family.py \
  --base_model /path/to/Qwen2.5-7B \
  --adapter /path/to/final_adapter \
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

Use `amplifier/code/eval_amp_family.py` and `oscillator/code/eval_osc_family.py` for amplifier and oscillator evaluation. Run any evaluator with `--help` to list family-specific options.

## Quick-start training

The one-round wrapper takes the base model, anchor adapter, and the JSONL corpus
used for post-PVPO SFT regularization:

```bash
bash scripts/quickstart_train_dcdc.sh \
  /path/to/Qwen2.5-7B \
  /path/to/anchor_sft_adapter \
  /path/to/pvpo_sft.jsonl
```

The DC--DC orchestration entry point is `dcdc/code/run_vpspi_pipeline.py`. A one-round example is:

```bash
cd dcdc/code
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

Set `--selfplay_sim_workers` and `--ppo_sim_workers` according to the effective CPU quota, not the host-wide logical CPU count. The archived main-run values and continuation accounting are reported in the Supplementary Information.

## Training stages and artifacts

For each completed outer round, the pipeline writes:

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

The paper tables are accompanied by task definitions, aggregate values, tolerance sweeps, and controlled-replay details in the Supplementary Information. Preserve task scope, tolerance, fallback setting, repair/search budget, and duplicate policy when comparing new runs with those values. Results with different settings are not directly interchangeable.

## License and citation

Use the repository license for redistribution terms. Please cite the accompanying paper when using the code or artifacts.
