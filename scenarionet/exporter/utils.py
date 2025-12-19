"""
Utility functions for OpenDRIVE/OpenSCENARIO export.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from scenarionet.common_utils import read_dataset_summary, read_scenario
from scenarionet.exporter.opendrive.exporter import OpenDriveExporter
from scenarionet.exporter.openscenario.exporter import OpenScenarioExporter

logger = logging.getLogger(__name__)


def export_scenario_to_openx(
    scenario: Dict[str, Any],
    output_dir: str,
    scenario_name: Optional[str] = None,
    export_opendrive: bool = True,
    export_openscenario: bool = True,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Export a single scenario to OpenDRIVE and/or OpenSCENARIO formats.

    Args:
        scenario: ScenarioNet scenario dict
        output_dir: Directory to save exported files
        scenario_name: Base name for output files (without extension)
        export_opendrive: Whether to export OpenDRIVE format
        export_openscenario: Whether to export OpenSCENARIO format

    Returns:
        Tuple of (opendrive_path, openscenario_path), None for formats not exported
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate scenario name if not provided
    if scenario_name is None:
        metadata = scenario.get("metadata", {})
        scenario_name = metadata.get("scenario_id", metadata.get("id", "scenario"))
        scenario_name = str(scenario_name).replace("/", "_").replace("\\", "_")

    xodr_path = None
    xosc_path = None

    # Export OpenDRIVE
    if export_opendrive:
        xodr_path = os.path.join(output_dir, f"{scenario_name}.xodr")
        try:
            OpenDriveExporter.export_scenario(scenario, xodr_path)
            logger.info(f"Exported OpenDRIVE: {xodr_path}")
        except Exception as e:
            logger.error(f"Failed to export OpenDRIVE: {e}")
            xodr_path = None

    # Export OpenSCENARIO
    if export_openscenario:
        xosc_path = os.path.join(output_dir, f"{scenario_name}.xosc")
        try:
            odr_ref = f"{scenario_name}.xodr" if xodr_path else None
            OpenScenarioExporter.export_scenario(scenario, xosc_path, opendrive_path=odr_ref)
            logger.info(f"Exported OpenSCENARIO: {xosc_path}")
        except Exception as e:
            logger.error(f"Failed to export OpenSCENARIO: {e}")
            xosc_path = None

    return xodr_path, xosc_path


def export_database_to_openx(
    database_path: str,
    output_dir: str,
    scenario_ids: Optional[List[str]] = None,
    max_scenarios: Optional[int] = None,
    export_opendrive: bool = True,
    export_openscenario: bool = True,
    overwrite: bool = False,
) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """
    Export multiple scenarios from a ScenarioNet database to OpenDRIVE/OpenSCENARIO.

    Args:
        database_path: Path to ScenarioNet database
        output_dir: Directory to save exported files
        scenario_ids: Specific scenario IDs to export (None = all)
        max_scenarios: Maximum number of scenarios to export
        export_opendrive: Whether to export OpenDRIVE format
        export_openscenario: Whether to export OpenSCENARIO format
        overwrite: Whether to overwrite existing files

    Returns:
        Dict mapping scenario_id to (opendrive_path, openscenario_path)
    """
    # Read database
    summary, all_ids, mapping = read_dataset_summary(database_path)

    # Filter scenario IDs
    if scenario_ids:
        ids_to_export = [sid for sid in all_ids if sid in scenario_ids]
    else:
        ids_to_export = list(all_ids)

    if max_scenarios:
        ids_to_export = ids_to_export[:max_scenarios]

    logger.info(f"Exporting {len(ids_to_export)} scenarios")

    results = {}
    for scenario_id in ids_to_export:
        # Check if already exists
        base_name = scenario_id.replace(".pkl", "").replace("/", "_")
        xodr_path = os.path.join(output_dir, f"{base_name}.xodr")
        xosc_path = os.path.join(output_dir, f"{base_name}.xosc")

        if not overwrite:
            if os.path.exists(xodr_path) and os.path.exists(xosc_path):
                logger.debug(f"Skipping existing: {scenario_id}")
                results[scenario_id] = (xodr_path, xosc_path)
                continue

        try:
            scenario = read_scenario(database_path, mapping, scenario_id)
            result = export_scenario_to_openx(
                scenario,
                output_dir,
                scenario_name=base_name,
                export_opendrive=export_opendrive,
                export_openscenario=export_openscenario,
            )
            results[scenario_id] = result
        except Exception as e:
            logger.error(f"Failed to export {scenario_id}: {e}")
            results[scenario_id] = (None, None)

    return results


def validate_opendrive(xodr_path: str) -> bool:
    """
    Basic validation of an OpenDRIVE file.

    Args:
        xodr_path: Path to .xodr file

    Returns:
        True if file is valid XML with OpenDRIVE structure
    """
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(xodr_path)
        root = tree.getroot()
        return root.tag == "OpenDRIVE"
    except Exception as e:
        logger.error(f"OpenDRIVE validation failed: {e}")
        return False


def validate_openscenario(xosc_path: str) -> bool:
    """
    Basic validation of an OpenSCENARIO file.

    Args:
        xosc_path: Path to .xosc file

    Returns:
        True if file is valid XML with OpenSCENARIO structure
    """
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(xosc_path)
        root = tree.getroot()
        return root.tag == "OpenSCENARIO"
    except Exception as e:
        logger.error(f"OpenSCENARIO validation failed: {e}")
        return False
