import argparse
import pathlib
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Run metrics on all .pt files in a model folder")
    parser.add_argument("script", type=str, help="Path to the metric script (e.g., ~/cdgm_textgen/measure_collapse.py)")
    parser.add_argument("model_dir", type=pathlib.Path, help="Path to the model directory")
    args = parser.parse_args()

    script_path = "sergey/eacl/run_metrics.sh"

    # Check if directory exists
    if not args.model_dir.is_dir():
        raise NotADirectoryError(f"{args.model_dir} is not a valid directory")

    # Find and process .pt files
    pt_files = sorted(args.model_dir.glob("*.pt"))
    if not pt_files:
        print(f"No .pt files found in {args.model_dir}")
        return

    for pt_file in pt_files:
        print(f"Running metrics for {pt_file.name}...")
        subprocess.run(
            ["bash", script_path, str(args.script), str(args.model_dir), pt_file.name],
            check=True
        )

if __name__ == "__main__":
    main()
