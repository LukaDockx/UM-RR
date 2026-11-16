"""Command-line interface for the initial condition generator."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import InitialConditionConfig
from .sampler import InitialConditionGenerator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate quasiclassical initial conditions.")
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="YAML input file. If omitted, package defaults are used.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the YAML settings without running electronic calculations.",
    )
    args = parser.parse_args(argv)

    base_dir = Path.cwd().resolve()
    if args.input is None:
        config = InitialConditionConfig.from_default_yaml()
    else:
        input_path = Path(args.input).resolve()
        if input_path.suffix.lower() in {".yaml", ".yml"}:
            config = InitialConditionConfig.from_yaml(input_path)
            base_dir = input_path.parent
        else:
            raise ValueError(f"Input must be a YAML file: {input_path}")

    config.validate()
    if args.check:
        print(f"UM-RR configuration is valid: {args.input or 'package defaults'}")
        return 0

    print(
        f"[UMRR] Loaded configuration from {args.input or 'package defaults'}",
        flush=True,
    )
    print(f"[UMRR] Base directory: {base_dir}", flush=True)
    generator = InitialConditionGenerator(config, base_dir=base_dir, verbose=True)
    result = generator.run()
    print(
        f"Wrote {result.num_samples} initial condition(s) to "
        f"{generator.output_dir / config.output.initial_conditions_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
