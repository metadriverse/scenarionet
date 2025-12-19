"""
Export ScenarioNet scenarios to OpenDRIVE (.xodr) and OpenSCENARIO (.xosc) formats.

This script reads scenarios from a ScenarioNet database and exports them to
ASAM OpenDRIVE and OpenSCENARIO formats, which are widely supported by
driving simulation platforms.

Usage:
    python -m scenarionet.export_openx --database_path /path/to/database --output_path /path/to/output

    # Export only OpenDRIVE (map)
    python -m scenarionet.export_openx -d /path/to/db -o /path/to/output --format opendrive

    # Export only OpenSCENARIO (scenarios)
    python -m scenarionet.export_openx -d /path/to/db -o /path/to/output --format openscenario

    # Export both (default)
    python -m scenarionet.export_openx -d /path/to/db -o /path/to/output --format both
"""

desc = "Export ScenarioNet scenarios to OpenDRIVE/OpenSCENARIO formats"

if __name__ == "__main__":
    import argparse
    import logging
    import os
    import sys
    from typing import List, Optional

    import tqdm

    from scenarionet.common_utils import read_dataset_summary, read_scenario
    from scenarionet.exporter.opendrive.exporter import OpenDriveExporter
    from scenarionet.exporter.openscenario.exporter import OpenScenarioExporter

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--database_path",
        "-d",
        required=True,
        help="Path to the ScenarioNet database directory containing .pkl files"
    )
    parser.add_argument(
        "--output_path",
        "-o",
        required=True,
        help="Directory to save the exported OpenDRIVE/OpenSCENARIO files"
    )
    parser.add_argument(
        "--format",
        "-f",
        default="both",
        choices=["opendrive", "openscenario", "both"],
        help="Export format: 'opendrive' (.xodr), 'openscenario' (.xosc), or 'both' (default)"
    )
    parser.add_argument(
        "--scenario_ids",
        "-s",
        nargs="+",
        default=None,
        help="Specific scenario IDs to export. If not specified, exports all scenarios"
    )
    parser.add_argument(
        "--max_scenarios",
        "-m",
        type=int,
        default=None,
        help="Maximum number of scenarios to export"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input
    if not os.path.exists(args.database_path):
        logger.error(f"Database path does not exist: {args.database_path}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Read dataset
    logger.info(f"Reading dataset from: {args.database_path}")
    try:
        summary, scenario_ids, mapping = read_dataset_summary(args.database_path)
    except Exception as e:
        logger.error(f"Failed to read dataset: {e}")
        sys.exit(1)

    logger.info(f"Found {len(scenario_ids)} scenarios in the database")

    # Filter scenarios if specific IDs provided
    if args.scenario_ids:
        scenario_ids = [sid for sid in scenario_ids if sid in args.scenario_ids]
        logger.info(f"Filtered to {len(scenario_ids)} specified scenarios")

    # Limit scenarios if max specified
    if args.max_scenarios and len(scenario_ids) > args.max_scenarios:
        scenario_ids = scenario_ids[:args.max_scenarios]
        logger.info(f"Limited to {len(scenario_ids)} scenarios")

    if not scenario_ids:
        logger.warning("No scenarios to export")
        sys.exit(0)

    # Export scenarios
    export_opendrive = args.format in ["opendrive", "both"]
    export_openscenario = args.format in ["openscenario", "both"]

    success_count = 0
    error_count = 0

    for scenario_id in tqdm.tqdm(scenario_ids, desc="Exporting scenarios"):
        try:
            # Read scenario
            scenario = read_scenario(args.database_path, mapping, scenario_id)

            # Generate output file names
            base_name = scenario_id.replace(".pkl", "").replace("/", "_")
            xodr_path = os.path.join(args.output_path, f"{base_name}.xodr")
            xosc_path = os.path.join(args.output_path, f"{base_name}.xosc")

            # Check if files exist
            if not args.overwrite:
                if export_opendrive and os.path.exists(xodr_path):
                    logger.debug(f"Skipping existing file: {xodr_path}")
                    continue
                if export_openscenario and os.path.exists(xosc_path):
                    logger.debug(f"Skipping existing file: {xosc_path}")
                    continue

            # Export OpenDRIVE
            if export_opendrive:
                try:
                    OpenDriveExporter.export_scenario(scenario, xodr_path)
                    logger.debug(f"Exported OpenDRIVE: {xodr_path}")
                except Exception as e:
                    logger.warning(f"Failed to export OpenDRIVE for {scenario_id}: {e}")

            # Export OpenSCENARIO
            if export_openscenario:
                try:
                    # Link to OpenDRIVE file if also exported
                    odr_ref = f"{base_name}.xodr" if export_opendrive else None
                    OpenScenarioExporter.export_scenario(scenario, xosc_path, opendrive_path=odr_ref)
                    logger.debug(f"Exported OpenSCENARIO: {xosc_path}")
                except Exception as e:
                    logger.warning(f"Failed to export OpenSCENARIO for {scenario_id}: {e}")

            success_count += 1

        except Exception as e:
            logger.warning(f"Failed to process scenario {scenario_id}: {e}")
            error_count += 1

    # Summary
    logger.info(f"\nExport complete!")
    logger.info(f"  Successful: {success_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info(f"  Output directory: {args.output_path}")

    if export_opendrive:
        logger.info(f"  OpenDRIVE files: {args.output_path}/*.xodr")
    if export_openscenario:
        logger.info(f"  OpenSCENARIO files: {args.output_path}/*.xosc")
