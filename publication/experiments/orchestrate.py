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
        value_str = " ".join(str(x.absolute().resolve() if isinstance(x, Path) else x) for x in values)
        return f"--{self.key} {value_str}"

    def __repr__(self):
        return f"{self.__class__.__name__}[key={self.key}, value={self.value}] => ({self})"


class ArgumentSet:
    def __init__(self, arguments: list[Argument] | None = None, **kwargs):
        if arguments is None:
            arguments = []
        self.arguments = arguments
        self.argument_dict = {arg.key: arg for arg in self.arguments}
        if len(self.arguments) != len(self.argument_dict):
            raise ValueError(f"Duplicate arguments found: {[arg.key for arg in self.arguments]}")
        self.add(**kwargs)

    def __len__(self):
        return len(self.arguments)

    def __bool__(self):
        return bool(self.arguments)

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

    def add(self, *args: Argument, **kwargs):
        new_args = list(args)
        for k, v in kwargs.items():
            new_args.append(Argument(key=k, value=v))
        for arg in new_args:
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
        return " ".join(map(str, sorted(self, key=lambda arg: isinstance(arg.value, Path))))


def assign_variable_args(match: ArgumentSet, options: ArgumentSet):
    new_args = ArgumentSet()
    for opt_cfg in options:
        opt_key = opt_cfg.key
        for match_key, opt_case in opt_cfg.value.items():
            if match_key not in match:
                raise KeyError(f"Variable training argument {opt_key} assigned to non-existing match parameter {match_key}")
            for match_value, opt_value in opt_case.items():
                if match.get(match_key).value == match_value:
                    new_args.add(Argument(key=opt_key, value=opt_value))
    return new_args


def split_var_args(orig: ArgumentSet):
    fixed_args = ArgumentSet()
    var_args = ArgumentSet()
    for arg in orig:
        if isinstance(arg.value, dict):
            var_args.add(arg)
        else:
            fixed_args.add(arg)
    return fixed_args, var_args


def configure_input_output(args: ArgumentSet, datasets: dict[str, dict[str, str | bool]]):
    dataset = args.pop("dataset")
    args.add(Argument(key="input", value=datasets[dataset.value]["path"]))

    if "data_index" not in args:
        args.add(Argument(key="data_index", value=datasets[dataset.value]["data_index"]))
    data_index = args.get("data_index").value

    if isinstance(data_index, Path):
        data_index = str(data_index.absolute().resolve())
    assert isinstance(data_index, (bool, str))

    return args, dataset, data_index


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
    args_data = config.get("args", {}) or {}

    # Split fixed and variable args
    (
        (shared_args, var_shared_args),
        (base_train_args, var_train_args),
        (base_eval_args, var_eval_args),
        (base_metric_args, var_metric_args),
    ) = [split_var_args(ArgumentSet(**(args_data.get(arg_set, {}) or {}))) for arg_set in ["shared", "train", "eval", "metrics"]]

    # Dataset config
    dataset_cfg = config.get("datasets", {})

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
    res_dir = out_dir / "results"
    run_dir = res_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    metric_dir = res_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)

    # Resolve config arguments
    output_log = Path(slurm_cfg.get("output", f"{out_dir}/logs/train_%A_%a.log"))
    print(output_log, output_log.parent)
    output_log.parent.mkdir(parents=True, exist_ok=True)
    if "%j" in output_log.name:
        output_log = output_log.with_name(output_log.name.replace("%j", "%A_%a"))
    slurm_cfg["output"] = str(output_log.absolute().resolve())
    slurm_cfg["job-name"] = name

    # Define SLURM task files
    train_tasks_file, eval_tasks_file, metric_tasks_file = [
        (out_dir / f"{task}_tasks.txt").absolute().resolve() for task in ["train", "eval", "metric"]
    ]
    train_lines, eval_lines, metric_lines = [], [], []

    submit_file = out_dir / "array.sh"

    for combo in combinations:
        experiment_params = ArgumentSet([Argument(key=k, value=v) for k, v in zip(param_names, combo)])
        combo_name = "_".join(str(v).replace("/", "-") for v in experiment_params.values())

        # 1. Prepare Training Command
        train_args = base_train_args + shared_args + experiment_params
        train_args.add(*assign_variable_args(train_args, var_train_args + var_shared_args))
        train_args, train_dataset, train_data_index = configure_input_output(
            args=train_args.add(output=run_dir, name=combo_name), datasets=dataset_cfg
        )
        train_lines.append(f"{TRAIN_STUB} {train_args.build_args_string()}")

        # 2. Prepare Evaluation and Metric Command(s)
        eval_dataset_names = eval_dataset_map.get(train_dataset.value, [])

        model_out_dir = run_dir / combo_name
        weights_path = model_out_dir / "weights" / "last.pt"

        eval_cmds, metric_cmds = [], []
        for eval_name in eval_dataset_names:
            if not isinstance(eval_name, str):
                raise TypeError(f"Expected name to be a string, but found {eval_name=} ({type(eval_name)})")

            eval_out_dir = model_out_dir / "predict"
            result_csv = eval_out_dir / eval_name / "mini_metric.csv"
            metric_output = metric_dir / eval_name / combo_name
            metric_output.parent.mkdir(parents=True, exist_ok=True)

            eval_args = base_eval_args.copy().add(
                experiment_params.get("head"), weights=weights_path, verbose=True, dataset=eval_name, output=eval_out_dir, name=eval_name
            )
            eval_args.add(*assign_variable_args(eval_args, var_eval_args + var_shared_args))
            eval_args, eval_dataset, eval_data_index = configure_input_output(args=eval_args, datasets=dataset_cfg)
            # If data_index is not specified in the `datasets` section of the
            # config YAML, we can use the constructed data_index.json from
            # the training run - if and only if, the evaluation dataset is
            # the same as the training dataset
            if not eval_data_index:
                assert eval_dataset.value == train_dataset.value
                eval_args.pop("data_index")
                eval_args.add(data_index=model_out_dir / "data_index.json")

            metric_args = base_metric_args.copy().add(file=result_csv, output_dir=metric_output)
            # Scaffolding for variable metric args is implemented, but the use-case
            # is not clear, so we'll alert the user
            _mva = assign_variable_args(metric_args, var_metric_args)
            if _mva:
                print(f"Found variable metric args `{_mva.build_args_string()}`!")
                metric_args.add(*_mva)

            eval_cmds.append(f"{EVAL_STUB} {eval_args.build_args_string()}")
            metric_cmds.append(f"{METRIC_STUB} {metric_args.build_args_string()}")

        eval_lines.append(" && ".join(eval_cmds) or 'echo "No evaluation datasets mapped."')
        metric_lines.append(" && ".join(metric_cmds) or 'echo "No metrics to calculate."')

    train_tasks_file.write_text("\n".join(train_lines) + "\n")
    eval_tasks_file.write_text("\n".join(eval_lines) + "\n")
    metric_tasks_file.write_text("\n".join(metric_lines) + "\n")

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
            f'TRAIN_CMD=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {train_tasks_file})',
            f'EVAL_CMDS=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {eval_tasks_file})',
            f'METRIC_CMDS=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {metric_tasks_file})',
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
