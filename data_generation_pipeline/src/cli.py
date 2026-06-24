import argparse

from config_reader import load_config
from pipeline_runner import run_pipeline


def build_parser():
    p = argparse.ArgumentParser("data-pipeline")
    p.add_argument("command", choices=["run"], help="Command to run")
    p.add_argument(
        "--config", default="config/pipeline_config.yaml", help="Pipeline config YAML"
    )
    p.add_argument("--input", help="Input PDF/TXT file (overrides config)")
    p.add_argument("--output", help="Output JSON file (overrides config)")
    return p


def main():
    # parser = build_parser()
    # args = parser.parse_args()
    # with open(args.config, "r", encoding="utf-8") as f:
    #     cfg = yaml.safe_load(f)

    cfg1 = load_config("config/pipeline_config.yaml")
    cfg2 = load_config("config/model_config.yaml")
    cfg = cfg1 | cfg2

    input_path = ""
    out_file = ""
    if cfg:
        input_path = cfg["input_path"]
        out_file = cfg["output_file"]

    # input_path = args.input or cfg.get("file_path")
    # out_file = args.output or cfg.get("output_file", "data/output/results.json")

    run_pipeline(input_path, out_file, cfg)


if __name__ == "__main__":
    main()
