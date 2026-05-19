import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from leaderboard_utils import rebuild_leaderboards_from_summaries


def _write_summary(root, model_name, best_val, test_acc):
    model_dir = os.path.join(root, model_name)
    os.makedirs(model_dir, exist_ok=True)
    summary = {
        "model": model_name,
        "save_key": "accuracy",
        "best_val": best_val,
        "best_path": os.path.join(model_dir, f"best_{model_name}.pth"),
        "test_metrics": {"accuracy": test_acc},
        "output_dir": model_dir,
    }
    with open(os.path.join(model_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    with tempfile.TemporaryDirectory(prefix="lb_agg_") as tmp_root:
        _write_summary(tmp_root, "model_a", best_val=0.62, test_acc=0.55)
        _write_summary(tmp_root, "model_b", best_val=0.71, test_acc=0.60)
        _write_summary(tmp_root, "model_c", best_val=0.58, test_acc=0.53)

        rebuild_leaderboards_from_summaries(tmp_root)

        _write_summary(tmp_root, "model_d", best_val=0.69, test_acc=0.61)
        rebuild_leaderboards_from_summaries(tmp_root)

        lb_val = _load_json(os.path.join(tmp_root, "leaderboard_val.json"))
        lb_test = _load_json(os.path.join(tmp_root, "leaderboard_test.json"))

        expected_models = {"model_a", "model_b", "model_c", "model_d"}
        if {row["model"] for row in lb_val} != expected_models:
            raise AssertionError(f"Val leaderboard models mismatch: {lb_val}")
        if {row["model"] for row in lb_test} != expected_models:
            raise AssertionError(f"Test leaderboard models mismatch: {lb_test}")

        val_map = {row["model"]: row["val_accuracy"] for row in lb_val}
        test_map = {row["model"]: row["test_accuracy"] for row in lb_test}

        expected_val = {"model_a": 0.62, "model_b": 0.71, "model_c": 0.58, "model_d": 0.69}
        expected_test = {"model_a": 0.55, "model_b": 0.60, "model_c": 0.53, "model_d": 0.61}

        if val_map != expected_val:
            raise AssertionError(f"Val accuracies mismatch: {val_map} != {expected_val}")
        if test_map != expected_test:
            raise AssertionError(f"Test accuracies mismatch: {test_map} != {expected_test}")

        if [row["model"] for row in lb_val] != ["model_b", "model_d", "model_a", "model_c"]:
            raise AssertionError(f"Val sorting mismatch: {lb_val}")
        if [row["model"] for row in lb_test] != ["model_d", "model_b", "model_a", "model_c"]:
            raise AssertionError(f"Test sorting mismatch: {lb_test}")

        print("Leaderboard aggregation verification passed.")


if __name__ == "__main__":
    main()
