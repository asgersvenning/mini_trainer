"""Score an ``mt_predict`` collector CSV with lepinet's rescoring function (mini_metrics' current convention).

Test images whose species is outside the model's vocabulary are dropped first: the parity test set is set 0
restricted to the training species. The scorer is lepinet's ``dev/093_rescore_mini_metrics.py:score`` so that
lepinet and mini_trainer runs share one implementation. Run in an environment with pandas, scikit-learn and
mini_metrics, for example lepinet's (see README.md). Writes ``predictions.csv`` (lepinet's layout) and
``scores.csv`` next to the input.
"""

import argparse
import importlib.util
from pathlib import Path

import pandas as pd

COLUMNS = ["instance_id", "filename", "level", "label", "prediction", "confidence", "threshold", "known_label"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, required=True, help="mini_metric.csv written by mt_predict / mt_hpredict.")
    parser.add_argument("--lepinet", type=Path, required=True, help="lepinet checkout providing dev/093_rescore_mini_metrics.py.")
    parser.add_argument("--expect-instances", type=int, default=None, help="Fail unless this many images remain after the restriction.")
    parser.add_argument("--seeds", type=int, default=5, help="Calibration splits for the calibrated column (0 skips it).")
    args = parser.parse_args()

    rows = pd.read_csv(args.predictions)
    unknown = set(rows.loc[(rows.level == 0) & (rows.known_label == 0), "instance_id"])
    rows = rows[~rows.instance_id.isin(unknown)].copy()
    instances = rows.instance_id.nunique()
    print(f"{len(unknown)} out-of-vocabulary images dropped; {instances} images scored.")
    if args.expect_instances is not None and instances != args.expect_instances:
        raise SystemExit(f"Expected {args.expect_instances} images after the vocabulary restriction, found {instances}.")
    rows["known_label"] = rows.known_label.astype(bool)
    destination = args.predictions.with_name("predictions.csv")
    rows[COLUMNS].to_csv(destination, index=False)

    spec = importlib.util.spec_from_file_location("lepinet_rescore", args.lepinet / "dev" / "093_rescore_mini_metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scores = pd.DataFrame(module.score(destination, None, args.seeds, None))
    scores.to_csv(args.predictions.with_name("scores.csv"), index=False)
    print(scores[scores.metric == "f1"][["level", "n_rows", "zero", "cal_mean", "cal_sd"]].to_string(index=False))


if __name__ == "__main__":
    main()
