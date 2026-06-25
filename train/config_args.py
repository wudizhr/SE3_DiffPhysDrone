import argparse

from se3diff_config.io import config_to_namespace, load_experiment_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the CUDA batch drone policy with the pm_model network."
    )
    parser.add_argument("--config", required=True, help="Path to YAML experiment config")
    return parser


def parse_train_args(argv=None):
    parsed = build_parser().parse_args(argv)
    config = load_experiment_config(parsed.config)
    namespace = config_to_namespace(config)
    namespace.config = parsed.config
    namespace.structured_config = config
    return namespace
