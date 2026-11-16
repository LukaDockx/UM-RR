"""Run the MeOOH example from its XYZ geometry and YAML input."""

from __future__ import annotations

import argparse
from pathlib import Path

from UMRR import InitialConditionConfig, InitialConditionGenerator
from UMRR.reference import prepare_reference


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write the optimization input without running Q-Chem.",
    )
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = InitialConditionConfig.from_yaml(config_path)
    directory = config_path.parent
    reference = prepare_reference(config, directory, "seed.xyz", prepare_only=args.prepare_only)
    if args.prepare_only:
        print(f"Wrote optimization input: {reference}")
        return 0
    generator = InitialConditionGenerator(config, base_dir=directory, verbose=True)
    result = generator.run()
    print(f"Wrote {result.num_samples} initial conditions to {generator.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
