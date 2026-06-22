# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "pyyaml",
# ]
# ///

import argparse
import itertools
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

BASE_DIR = Path("slurm_jobs")


@dataclass(kw_only=True)
class Argument:
    key: str
    value: Any

    def __str__(self):
        if isinstance(self.value, bool):
            if not self.value:
                return ""
            return f"--{self.key}"
        values = [self.value] if not isinstance(self.value, (list, tuple)) else self.value
        value_str = " ".join(str(x) for x in values)
        return f"--{self.key} {value_str}"

    def __repr__(self):
        return f"{self.__class__.__name__}[key={self.key}, value={self.value}] => ({self})"


class ArgumentSet:
    def __init__(self, arguments: list[Argument] | None = None):
        if arguments is None:
            arguments = []
        self.arguments = arguments
        self.argument_dict = {arg.key: arg for arg in self.arguments}
        if len(self.arguments) != len(self.argument_dict):
            raise ValueError(f"Duplicate arguments found: {[arg.key for arg in self.arguments]}")

    def __len__(self):
        return len(self.arguments)

    def __contains__(self, arg: Argument | str):
        return (arg.key if isinstance(arg, Argument) else arg) in self.argument_dict

    def __iter__(self):
        yield from self.arguments

    def keys(self):
        for arg in self:
            yield arg.key

    def values(self):
        for arg in self:
            yield arg.value

    def __getitem__(self, i: str | int | slice):
        if isinstance(i, str):
            return self.get(i)
        return self.arguments[i]

    def index(self, arg: Argument | str):
        for i, iarg in enumerate(self):
            if arg is iarg or isinstance(arg, str) and arg == iarg.key:
                return i
        raise KeyError(f'Argument "{arg}" does not exist.')

    def get(self, k: str):
        arg = self.argument_dict.get(k)
        if arg is None:
            raise KeyError(f'Argument "{k}" does not exist.')
        return arg

    def pop(self, k: str):
        arg = self.get(k)
        self.arguments.pop(self.index(k))
        self.argument_dict.pop(k)
        return arg

    def add(self, arg: Argument):
        if arg in self:
            raise KeyError(f'Argument "{arg}" already exists.')
        self.arguments.append(arg)
        self.argument_dict[arg.key] = arg
        return self

    def __add__(self, other: "ArgumentSet"):
        return ArgumentSet(self.arguments + other.arguments)

    def copy(self):
        """Return a shallow copy of the argument set."""
        return ArgumentSet(self.arguments.copy())

    def build_args_string(self) -> str:
        """Constructs the CLI string for shared, static arguments."""
        return " ".join(map(str, self))


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
    if not isinstance(name, str):
        raise TypeError(f"Expected name to be a string, but found {name=} ({type(name)})")

    stubs = config.get("stubs")
    TRAIN_STUB = stubs["train"]
    EVAL_STUB = stubs["eval"]
    METRIC_STUB = stubs["metric"]
    slurm_cfg = config.get("slurm", {})
    exp_cfg = config.get("experiment", {})
    _args_cfg = ArgumentSet([Argument(key=k, value=v) for k, v in (config.get("args", {}) or {}).items()])

    dataset_cfg = config.get("datasets", {})

    args_cfg = ArgumentSet()
    variable_args_cfg = ArgumentSet()
    for arg in _args_cfg:
        if isinstance(arg.value, dict):
            variable_args_cfg.add(arg)
        else:
            args_cfg.add(arg)

    # Extract the dataset mapping dictionary from eval configuration
    eval_dataset_map = config.get("eval", {})

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

    train_lines = []
    eval_lines = []
    metric_lines = []

    for combo in combinations:
        experiment_params = ArgumentSet([Argument(key=k, value=v) for k, v in zip(param_names, combo)])
        combo_name = "_".join(str(v).replace("/", "-") for v in experiment_params.values())

        # Assign variable arguments
        extra_args = ArgumentSet()
        for var_arg in variable_args_cfg:
            var_arg_key = var_arg.key
            for exp_param_key, override in var_arg.value.items():
                if exp_param_key not in experiment_params:
                    raise KeyError(
                        f"Variable training argument {var_arg.key} assigned to non-existing experiment parameter {exp_param_key}"
                    )
                for exp_param_value, arg_value in override.items():
                    if experiment_params.get(exp_param_key).value == exp_param_value:
                        extra_args.add(Argument(key=var_arg_key, value=arg_value))

        # 1. Prepare Training Commands
        train_args = args_cfg.copy() + experiment_params + extra_args
        dataset = train_args.pop("dataset")
        data_index = dataset_cfg[dataset.value]["data_index"]
        train_args.add(Argument(key="input", value=dataset_cfg[dataset.value]["path"]))
        if data_index:
            train_args.add(Argument(key="data_index", value=data_index))
        train_args.add(Argument(key="output", value=run_dir.absolute()))
        train_args.add(Argument(key="name", value=combo_name))

        train_cmd = f"{TRAIN_STUB} {train_args.build_args_string()}"
        train_lines.append(train_cmd)

        # 2. Prepare Evaluation and Metric Commands
        eval_dataset_names = eval_dataset_map.get(dataset.value, [])

        model_out_dir = run_dir / combo_name
        weights_path = model_out_dir / "weights" / "last.pt"

        base_eval_args = ArgumentSet([Argument(key="weights", value=weights_path.absolute()), Argument(key="verbose", value=True)])
        base_metric_args = ArgumentSet([Argument(key="all", value=True), Argument(key="verbose", value=True)])

        eval_cmds, metric_cmds = [], []
        for eval_name in eval_dataset_names:
            if not isinstance(eval_name, str):
                raise TypeError(f"Expected name to be a string, but found {eval_name=} ({type(eval_name)})")
            eval_data_index = dataset_cfg[eval_name]["data_index"]
            if not eval_data_index:
                eval_data_index = data_index or (model_out_dir / "data_index.json").absolute()
            eval_dataset = dataset_cfg[eval_name]
            eval_out_dir = model_out_dir / "predict"
            eval_out_dir.mkdir(parents=True, exist_ok=True)
            result_csv = eval_out_dir / eval_name / "mini_metric.csv"
            metric_output = metric_dir / eval_name / combo_name
            metric_output.parent.mkdir(parents=True, exist_ok=True)

            eval_args = base_eval_args.copy() + ArgumentSet(
                [
                    Argument(key="input", value=eval_dataset["path"]),
                    Argument(key="output", value=eval_out_dir.absolute()),
                    Argument(key="name", value=eval_name),
                    Argument(key="data_index", value=eval_data_index),
                ]
            )

            metric_args = base_metric_args.copy() + ArgumentSet(
                [Argument(key="file", value=result_csv.absolute()), Argument(key="output", value=metric_output.absolute())]
            )

            eval_cmds.append(f"{EVAL_STUB} {eval_args.build_args_string()}")
            metric_cmds.append(f"{METRIC_STUB} {metric_args.build_args_string()}")

        if eval_cmds:
            eval_lines.append(" && ".join(eval_cmds))
        else:
            eval_lines.append('echo "No evaluation datasets mapped."')

        if metric_cmds:
            metric_lines.append(" && ".join(metric_cmds))
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
        script_lines.append(f"#SBATCH --{key}={str(value)}")

    # Append the array directive
    script_lines.append(f"#SBATCH --array=1-{num_tasks}")

    script_lines.extend(
        [
            "",
            "set -e",
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
