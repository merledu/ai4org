import argparse
import yaml
from .pipeline_runner import run_pipeline

def build_parser():
    p = argparse.ArgumentParser("data-pipeline")
    p.add_argument("command", choices=["run"], help="Command to run")
    p.add_argument("--config", default="config/pipeline_config.yaml", help="Pipeline config YAML")
    p.add_argument("--input", help="Input PDF/TXT file (overrides config)")
    p.add_argument("--output", help="Output JSON file (overrides config)")
    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    input_path = args.input or cfg.get("file_path")
    out_file = args.output or cfg.get("output_file", "data/output/results.json")

    run_pipeline(input_path, out_file, cfg)

if __name__ == "__main__":
    main()

