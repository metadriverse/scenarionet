"""
Tests for OpenDRIVE and OpenSCENARIO export functionality.
"""

import os
import shutil
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from scenarionet.exporter.opendrive.exporter import OpenDriveExporter
from scenarionet.exporter.openscenario.exporter import OpenScenarioExporter
from scenarionet.exporter.utils import (
    export_scenario_to_openx,
    validate_opendrive,
    validate_openscenario,
)


def create_test_scenario():
    """Create a synthetic test scenario for testing export functionality."""
    # Create track length
    track_length = 100
    timesteps = np.linspace(0, 10, track_length)

    # Create ego vehicle track
    ego_track = {
        "type": "VEHICLE",
        "state": {
            "position": np.column_stack([
                np.linspace(0, 100, track_length),  # x
                np.linspace(0, 50, track_length),   # y
                np.zeros(track_length),              # z
            ]),
            "heading": np.full(track_length, 0.5),
            "velocity": np.column_stack([
                np.full(track_length, 10.0),  # vx
                np.full(track_length, 5.0),   # vy
            ]),
            "valid": np.ones(track_length, dtype=bool),
            "length": np.full(track_length, 4.5),
            "width": np.full(track_length, 1.8),
            "height": np.full(track_length, 1.5),
        },
        "metadata": {
            "track_length": track_length,
            "type": "VEHICLE",
            "object_id": "ego",
            "dataset": "test",
        },
    }

    # Create another vehicle
    vehicle_track = {
        "type": "VEHICLE",
        "state": {
            "position": np.column_stack([
                np.linspace(10, 110, track_length),  # x
                np.linspace(5, 55, track_length),    # y
                np.zeros(track_length),               # z
            ]),
            "heading": np.full(track_length, 0.5),
            "velocity": np.column_stack([
                np.full(track_length, 10.0),
                np.full(track_length, 5.0),
            ]),
            "valid": np.ones(track_length, dtype=bool),
            "length": np.full(track_length, 4.0),
            "width": np.full(track_length, 1.7),
            "height": np.full(track_length, 1.4),
        },
        "metadata": {
            "track_length": track_length,
            "type": "VEHICLE",
            "object_id": "vehicle_1",
            "dataset": "test",
        },
    }

    # Create a pedestrian
    pedestrian_track = {
        "type": "PEDESTRIAN",
        "state": {
            "position": np.column_stack([
                np.linspace(20, 30, track_length),
                np.linspace(10, 15, track_length),
                np.zeros(track_length),
            ]),
            "heading": np.full(track_length, 1.0),
            "velocity": np.column_stack([
                np.full(track_length, 1.0),
                np.full(track_length, 0.5),
            ]),
            "valid": np.ones(track_length, dtype=bool),
            "length": np.full(track_length, 0.5),
            "width": np.full(track_length, 0.5),
            "height": np.full(track_length, 1.7),
        },
        "metadata": {
            "track_length": track_length,
            "type": "PEDESTRIAN",
            "object_id": "pedestrian_1",
            "dataset": "test",
        },
    }

    # Create lane map features
    lane_1 = {
        "type": "LANE_SURFACE_STREET",
        "polyline": np.column_stack([
            np.linspace(0, 200, 50),
            np.zeros(50),
            np.zeros(50),
        ]),
        "speed_limit_kmh": 50.0,
        "entry_lanes": [],
        "exit_lanes": ["lane_2"],
        "width": np.column_stack([
            np.full(50, 1.8),
            np.full(50, 1.8),
        ]),
    }

    lane_2 = {
        "type": "LANE_SURFACE_STREET",
        "polyline": np.column_stack([
            np.linspace(200, 300, 30),
            np.linspace(0, 50, 30),
            np.zeros(30),
        ]),
        "speed_limit_kmh": 40.0,
        "entry_lanes": ["lane_1"],
        "exit_lanes": [],
        "width": np.column_stack([
            np.full(30, 1.8),
            np.full(30, 1.8),
        ]),
    }

    road_line = {
        "type": "ROAD_LINE_SOLID_SINGLE_WHITE",
        "polyline": np.column_stack([
            np.linspace(0, 200, 50),
            np.full(50, -3.5),
            np.zeros(50),
        ]),
    }

    stop_sign = {
        "type": "STOP_SIGN",
        "position": np.array([195, 0, 0]),
        "lane": ["lane_1"],
    }

    # Create traffic light states
    traffic_light = {
        "type": "TRAFFIC_LIGHT",
        "state": {
            "object_state": ["LANE_STATE_GO"] * 50 + ["LANE_STATE_CAUTION"] * 20 + ["LANE_STATE_STOP"] * 30,
        },
        "lane": "lane_1",
        "stop_point": np.array([195, 0, 0]),
        "metadata": {
            "track_length": track_length,
            "type": "TRAFFIC_LIGHT",
            "object_id": "tl_1",
            "dataset": "test",
        },
    }

    # Assemble scenario
    scenario = {
        "id": "test_scenario_001",
        "version": "test_v1.0",
        "length": track_length,
        "tracks": {
            "ego": ego_track,
            "vehicle_1": vehicle_track,
            "pedestrian_1": pedestrian_track,
        },
        "map_features": {
            "lane_1": lane_1,
            "lane_2": lane_2,
            "road_line_1": road_line,
            "stop_sign_1": stop_sign,
        },
        "dynamic_map_states": {
            "tl_1": traffic_light,
        },
        "metadata": {
            "id": "test_scenario_001",
            "scenario_id": "test_scenario_001",
            "dataset": "test",
            "coordinate": "right-handed",
            "timestep": timesteps,
            "sdc_id": "ego",
            "track_length": track_length,
        },
    }

    return scenario


class TestOpenDriveExporter:
    """Tests for OpenDRIVE export functionality."""

    def test_export_creates_file(self):
        """Test that export creates a valid .xodr file."""
        scenario = create_test_scenario()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.xodr")
            exporter = OpenDriveExporter(scenario)
            result = exporter.export(output_path)

            assert os.path.exists(result)
            assert result.endswith(".xodr")

    def test_export_valid_xml(self):
        """Test that exported file is valid XML."""
        scenario = create_test_scenario()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.xodr")
            OpenDriveExporter.export_scenario(scenario, output_path)

            # Parse XML
            tree = ET.parse(output_path)
            root = tree.getroot()

            assert root.tag == "OpenDRIVE"

    def test_export_contains_header(self):
        """Test that export contains header element."""
        scenario = create_test_scenario()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.xodr")
            OpenDriveExporter.export_scenario(scenario, output_path)

            tree = ET.parse(output_path)
            root = tree.getroot()

            header = root.find("header")
            assert header is not None
            assert header.get("revMajor") == "1"
            assert header.get("revMinor") == "6"

    def test_export_contains_roads(self):
        """Test that export contains road elements for lanes."""
        scenario = create_test_scenario()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.xodr")
            OpenDriveExporter.export_scenario(scenario, output_path)

            tree = ET.parse(output_path)
            root = tree.getroot()

            roads = root.findall("road")
            assert len(roads) >= 1  # At least one road from lanes

    def test_road_has_geometry(self):
        """Test that roads have planView with geometry."""
        scenario = create_test_scenario()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.xodr")
            OpenDriveExporter.export_scenario(scenario, output_path)

            tree = ET.parse(output_path)
            root = tree.getroot()

            road = root.find("road")
            if road is not None:
                plan_view = road.find("planView")
                assert plan_view is not None

                geometry = plan_view.find("geometry")
                assert geometry is not None
                assert geometry.find("line") is not None

    def test_road_has_lanes(self):
        """Test that roads have lane definitions."""
        scenario = create_test_scenario()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.xodr")
            OpenDriveExporter.export_scenario(scenario, output_path)

            tree = ET.parse(output_path)
            root = tree.getroot()

            road = root.find("road")
            if road is not None:
                lanes = road.find("lanes")
                assert lanes is not None

                lane_section = lanes.find("laneSection")
                assert lane_section is not None


class TestOpenScenarioExporter:
    """Tests for OpenSCENARIO export functionality."""

    def test_export_creates_file(self):
        """Test that export creates a valid .xosc file."""
        scenario = create_test_scenario()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.xosc")
            exporter = OpenScenarioExporter(scenario)
            result = exporter.export(output_path)

            assert os.path.exists(result)
            assert result.endswith(".xosc")

    def test_export_valid_xml(self):
        """Test that exported file is valid XML."""
        scenario = create_test_scenario()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.xosc")
            OpenScenarioExporter.export_scenario(scenario, output_path)

            tree = ET.parse(output_path)
            root = tree.getroot()

            assert root.tag == "OpenSCENARIO"

    def test_export_contains_file_header(self):
        """Test that export contains FileHeader."""
        scenario = create_test_scenario()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.xosc")
            OpenScenarioExporter.export_scenario(scenario, output_path)

            tree = ET.parse(output_path)
            root = tree.getroot()

            header = root.find("FileHeader")
            assert header is not None
            assert header.get("revMajor") == "1"

    def test_export_contains_entities(self):
        """Test that export contains entity definitions."""
        scenario = create_test_scenario()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.xosc")
            OpenScenarioExporter.export_scenario(scenario, output_path)

            tree = ET.parse(output_path)
            root = tree.getroot()

            entities = root.find("Entities")
            assert entities is not None

            scenario_objects = entities.findall("ScenarioObject")
            assert len(scenario_objects) >= 1

    def test_export_ego_entity(self):
        """Test that ego vehicle is exported correctly."""
        scenario = create_test_scenario()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.xosc")
            OpenScenarioExporter.export_scenario(scenario, output_path)

            tree = ET.parse(output_path)
            root = tree.getroot()

            entities = root.find("Entities")
            ego = None
            for obj in entities.findall("ScenarioObject"):
                if obj.get("name") == "Ego":
                    ego = obj
                    break

            assert ego is not None
            vehicle = ego.find("Vehicle")
            assert vehicle is not None

    def test_export_contains_storyboard(self):
        """Test that export contains Storyboard."""
        scenario = create_test_scenario()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.xosc")
            OpenScenarioExporter.export_scenario(scenario, output_path)

            tree = ET.parse(output_path)
            root = tree.getroot()

            storyboard = root.find("Storyboard")
            assert storyboard is not None

            init = storyboard.find("Init")
            assert init is not None

            story = storyboard.find("Story")
            assert story is not None

    def test_export_with_opendrive_reference(self):
        """Test export with OpenDRIVE file reference."""
        scenario = create_test_scenario()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.xosc")
            OpenScenarioExporter.export_scenario(
                scenario, output_path, opendrive_path="test.xodr"
            )

            tree = ET.parse(output_path)
            root = tree.getroot()

            road_network = root.find("RoadNetwork")
            assert road_network is not None

            logic_file = road_network.find("LogicFile")
            assert logic_file is not None
            assert logic_file.get("filepath") == "test.xodr"


class TestExportUtils:
    """Tests for export utility functions."""

    def test_export_scenario_to_openx_both(self):
        """Test exporting both formats."""
        scenario = create_test_scenario()

        with tempfile.TemporaryDirectory() as tmpdir:
            xodr, xosc = export_scenario_to_openx(
                scenario, tmpdir, "test_scenario",
                export_opendrive=True, export_openscenario=True
            )

            assert xodr is not None
            assert xosc is not None
            assert os.path.exists(xodr)
            assert os.path.exists(xosc)

    def test_export_scenario_opendrive_only(self):
        """Test exporting only OpenDRIVE."""
        scenario = create_test_scenario()

        with tempfile.TemporaryDirectory() as tmpdir:
            xodr, xosc = export_scenario_to_openx(
                scenario, tmpdir, "test_scenario",
                export_opendrive=True, export_openscenario=False
            )

            assert xodr is not None
            assert xosc is None

    def test_validate_opendrive(self):
        """Test OpenDRIVE validation."""
        scenario = create_test_scenario()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.xodr")
            OpenDriveExporter.export_scenario(scenario, output_path)

            assert validate_opendrive(output_path) is True

    def test_validate_openscenario(self):
        """Test OpenSCENARIO validation."""
        scenario = create_test_scenario()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.xosc")
            OpenScenarioExporter.export_scenario(scenario, output_path)

            assert validate_openscenario(output_path) is True

    def test_validate_invalid_file(self):
        """Test validation with invalid file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = os.path.join(tmpdir, "invalid.xodr")
            with open(invalid_path, "w") as f:
                f.write("not xml")

            assert validate_opendrive(invalid_path) is False


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_map_features(self):
        """Test export with empty map features."""
        scenario = create_test_scenario()
        scenario["map_features"] = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.xodr")
            exporter = OpenDriveExporter(scenario)
            result = exporter.export(output_path)

            assert os.path.exists(result)

    def test_empty_tracks(self):
        """Test export with empty tracks."""
        scenario = create_test_scenario()
        scenario["tracks"] = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.xosc")
            exporter = OpenScenarioExporter(scenario)
            result = exporter.export(output_path)

            assert os.path.exists(result)

    def test_missing_metadata(self):
        """Test export with minimal metadata."""
        scenario = {
            "id": "test",
            "version": "1.0",
            "length": 10,
            "tracks": {},
            "map_features": {},
            "dynamic_map_states": {},
            "metadata": {},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            xodr_path = os.path.join(tmpdir, "test.xodr")
            xosc_path = os.path.join(tmpdir, "test.xosc")

            OpenDriveExporter.export_scenario(scenario, xodr_path)
            OpenScenarioExporter.export_scenario(scenario, xosc_path)

            assert os.path.exists(xodr_path)
            assert os.path.exists(xosc_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
