# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "pyyaml",
# ]
# ///

import argparse
import itertools
import stat
from pathlib import Path
from typing import Any

import yaml

BASE_DIR = Path("slurm_jobs")
TRAIN_STUB = "uv run --no-sync mt_htrain"
EVAL_STUB = "uv run --no-sync mt_hpredict"
METRIC_STUB = "uvx --from git+https://github.com/GuillaumeMougeot/mini_metrics.git mm_metrics"


def build_shared_args_string(args: dict[str, Any]) -> str:
    """Constructs the CLI string for shared, static arguments."""
    cmd = []
    for key, value in args.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        elif isinstance(value, list):
            val_str = " ".join(str(x) for x in value)
            cmd.extend([f"--{key}", val_str])
        else:
            cmd.extend([f"--{key}", str(value)])
    return " ".join(cmd)


def main():
    parser = argparse.ArgumentParser(description="Generate a Slurm Job Array from a YAML matrix.")
    parser.add_argument("config_path", type=Path, help="Path to the experiment YAML configuration file.")
    cli_args = parser.parse_args()

    config_path = cli_args.config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file {config_path} not found.")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    name = config.get("name", None)
    if name is None:
        raise KeyError(f"Missing `name` field in configuration file: {config_path}")

    slurm_cfg = config.get("slurm", {})
    exp_cfg = config.get("experiment", {})
    args_cfg = config.get("args", {})

    dataset_paths = config.get("datasets", {})

    variable_args_cfg = {}
    for k in list(args_cfg.keys()):
        v = args_cfg[k]
        if isinstance(v, dict):
            variable_args_cfg[k] = args_cfg.pop(k)
        else:
            continue

    # Extract the dataset mapping dictionary from eval configuration
    eval_cfg = config.get("eval", {})
    eval_dataset_map = eval_cfg.get("dataset", {})

    if not exp_cfg:
        raise ValueError("No 'experiment' parameters found in the configuration.")

    # Calculate the cartesian product across all dynamic axes
    param_names = list(exp_cfg.keys())
    param_value_lists = list(exp_cfg.values())
    combinations = list(itertools.product(*param_value_lists))

    num_tasks = len(combinations)
    if num_tasks == 0:
        print("No combinations generated. Exiting.")
        return

    # Set up output directory
    out_dir = BASE_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir = out_dir / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    run_dir = res_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    metric_dir = res_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)

    train_tasks_file = out_dir / "train_tasks.txt"
    eval_tasks_file = out_dir / "eval_tasks.txt"
    metric_tasks_file = out_dir / "metric_tasks.txt"
    submit_file = out_dir / "array.sh"

    flag_mapping = {"dataset": "-i"}
    train_lines = []
    eval_lines = []
    metric_lines = []

    # Pre-calculate shared arguments string for the training commands
    shared_args_str = build_shared_args_string(args_cfg)

    for combo in combinations:
        experiment_params = dict(zip(param_names, combo))

        # Assign variable arguments
        extra_args = {}
        for train_param, exp_param in variable_args_cfg.items():
            for vk, vvks in exp_param.items():
                if vk not in experiment_params:
                    raise KeyError(f"Variable training argument {train_param} assigned to non-existing experiment parameter {vk}")
                for vvk, vv in vvks.items():
                    if experiment_params[vk] == vvk:
                        extra_args[train_param] = vv

        # 1. Prepare Training Commands
        train_args = []
        safe_values = [str(v).replace("/", "-") for v in experiment_params.values()]
        combo_name = "_".join(safe_values)

        for key, value in itertools.chain(experiment_params.items(), extra_args.items()):
            if key == "dataset":
                value = dataset_paths[value]
            cli_flag = flag_mapping.get(key, f"--{key}")
            train_args.extend([cli_flag, str(value)])

        train_args.extend(["-o", str(run_dir.absolute()), "--name", combo_name])

        # Construct the full training command directly in python
        train_cmd = f"{TRAIN_STUB} {' '.join(train_args)} {shared_args_str}"
        train_lines.append(train_cmd)

        # 2. Prepare Evaluation and Metric Commands
        current_train_dataset = experiment_params.get("dataset")
        eval_datasets = eval_dataset_map.get(current_train_dataset, [])

        model_out_dir = run_dir / combo_name
        weights_path = model_out_dir / "weights" / "last.pt"

        e_cmds = []
        m_cmds = []

        for eval_ds in eval_datasets:
            eval_out_dir = model_out_dir / "predict"
            eval_out_dir.mkdir(parents=True, exist_ok=True)
            result_csv = eval_out_dir / eval_ds / "mini_metric.csv"
            metric_output = metric_dir / eval_ds / combo_name
            metric_output.parent.mkdir(parents=True, exist_ok=True)

            e_cmds.append(
                f"{EVAL_STUB} -i {dataset_paths[eval_ds]} -o {eval_out_dir.absolute()} "
                f"--name {eval_ds} --weights {weights_path.absolute()} --verbose"
            )
            m_cmds.append(f"{METRIC_STUB} --file {result_csv.absolute()} -av --output {metric_output.absolute()}")

        if e_cmds:
            eval_lines.append(" && ".join(e_cmds))
        else:
            eval_lines.append('echo "No evaluation datasets mapped."')

        if m_cmds:
            metric_lines.append(" && ".join(m_cmds))
        else:
            metric_lines.append('echo "No metrics to calculate."')

    train_tasks_file.write_text("\n".join(train_lines) + "\n")
    eval_tasks_file.write_text("\n".join(eval_lines) + "\n")
    metric_tasks_file.write_text("\n".join(metric_lines) + "\n")

    # Update slurm_cfg to handle array formatting for output logs
    output_log = Path(slurm_cfg.get("output", f"{out_dir}/logs/train_%A_%a.log"))
    print(output_log, output_log.parent)
    output_log.parent.mkdir(parents=True, exist_ok=True)
    if "%j" in output_log.name:
        output_log = output_log.with_name(output_log.name.replace("%j", "%A_%a"))
    slurm_cfg["output"] = str(output_log.absolute())

    # Clean up the base job name
    slurm_cfg["job-name"] = name

    script_lines = ["#!/bin/bash"]
    for key, value in slurm_cfg.items():
        script_lines.append(f"#SBATCH --{key}={value}")

    # Append the array directive
    script_lines.append(f"#SBATCH --array=1-{num_tasks}")

    script_lines.extend(
        [
            "",
            'echo "Job ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"',
            'echo "Running on node: $SLURMD_NODENAME"',
            "",
            "# Extract the Nth line from task files",
            f'TRAIN_CMD=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {train_tasks_file.absolute()})',
            f'EVAL_CMDS=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {eval_tasks_file.absolute()})',
            f'METRIC_CMDS=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {metric_tasks_file.absolute()})',
            "",
            "# Train",
            'echo "=== Start training ==="',
            'eval "$TRAIN_CMD"',
            'echo "=== End training ==="',
            "",
            "# Eval",
            'echo "=== Start eval ==="',
            'eval "$EVAL_CMDS"',
            'echo "=== End eval ==="',
            "",
            "# Metrics",
            'echo "=== Start metrics ==="',
            'eval "$METRIC_CMDS"',
            'echo "=== End metrics ==="',
            "",
            'echo "Finished"',
        ]
    )

    submit_file.write_text("\n".join(script_lines) + "\n")
    submit_file.chmod(submit_file.stat().st_mode | stat.S_IEXEC)

    print(f"Generated {num_tasks} task pipelines in '{out_dir}/'.")
    print(f"Generated submission script at '{submit_file}'.")
    print("\nRun the array from your root directory using:")
    print(f"sbatch {submit_file}")


if __name__ == "__main__":
    main()
