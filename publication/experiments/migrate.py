# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

import argparse
import os
import shutil
from pathlib import Path

TEXT_EXTENSIONS = {".yaml", ".yml", ".json", ".txt", ".sh", ".csv"}


def rewrite_paths_in_file(file_path: Path, old_str: str, new_str: str) -> bool:
    """Read a text file, replace instances of old_str with new_str, and write back if modified."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception:
        return False

    if old_str in content:
        updated = content.replace(old_str, new_str)
        file_path.write_text(updated, encoding="utf-8")
        return True
    return False


def migrate_results(src_dir: Path, dst_dir: Path, task_dir: Path | None = None, move: bool = False, dry_run: bool = False) -> None:
    """Migrate result directory and rewrite embedded paths in config/task files."""
    src_abs = src_dir.expanduser().resolve()
    dst_abs = dst_dir.expanduser().resolve()

    if not src_abs.exists():
        raise FileNotFoundError(f"Source directory '{src_abs}' does not exist.")

    print("Migrating experiment results:")
    print(f"  Source:      {src_abs}")
    print(f"  Destination: {dst_abs}")
    if move:
        print("  Mode:        Move")
    else:
        print("  Mode:        Copy")

    if not dry_run:
        dst_abs.parent.mkdir(parents=True, exist_ok=True)
        if move:
            if dst_abs.exists():
                shutil.copytree(src_abs, dst_abs, dirs_exist_ok=True)
                shutil.rmtree(src_abs)
            else:
                shutil.move(src_abs, dst_abs)
        else:
            shutil.copytree(src_abs, dst_abs, dirs_exist_ok=True)

    # Collect directories to scan for path replacements
    scan_dirs = [dst_abs]
    if task_dir is not None:
        task_abs = task_dir.expanduser().resolve()
        if task_abs.exists():
            scan_dirs.append(task_abs)

    old_path_str = str(src_abs)
    new_path_str = str(dst_abs)

    modified_count = 0
    for sdir in scan_dirs:
        for root, _, files in os.walk(sdir):
            for fname in files:
                fpath = Path(root) / fname
                if fpath.suffix.lower() in TEXT_EXTENSIONS or fpath.name.endswith("_tasks.txt") or fpath.name == "array.sh":
                    if dry_run:
                        try:
                            content = fpath.read_text(encoding="utf-8")
                            if old_path_str in content:
                                modified_count += 1
                                print(f"[DRY-RUN] Would update paths in: {fpath}")
                        except Exception:
                            pass
                    else:
                        if rewrite_paths_in_file(fpath, old_path_str, new_path_str):
                            modified_count += 1
                            print(f"Updated paths in: {fpath}")

    print(f"Migration complete. Updated path references in {modified_count} file(s).")


def main():
    parser = argparse.ArgumentParser(description="Migrate experiment results and rewrite embedded absolute file paths.")
    parser.add_argument("src_dir", type=Path, help="Source results directory (e.g., slurm_jobs/<NAME>/results).")
    parser.add_argument("dst_dir", type=Path, help="Destination results directory (e.g., /dcai/projects/.../results).")
    parser.add_argument(
        "--task_dir",
        type=Path,
        default=None,
        help="Optional SLURM task directory (e.g., slurm_jobs/<NAME>) to update task scripts.",
    )
    parser.add_argument("--move", action="store_true", help="Move instead of copying files.")
    parser.add_argument("--dry_run", action="store_true", help="Print actions without modifying files.")

    args = parser.parse_args()
    migrate_results(src_dir=args.src_dir, dst_dir=args.dst_dir, task_dir=args.task_dir, move=args.move, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
