#!/usr/bin/env python
"""
Generate example OpenDRIVE and OpenSCENARIO files from a synthetic scenario.

This script creates a synthetic driving scenario and exports it to both formats
to demonstrate the export functionality.
"""

import os
import sys

# Add parent directories to path for direct imports (avoiding metadrive dependency)
script_dir = os.path.dirname(os.path.abspath(__file__))
scenarionet_dir = os.path.dirname(script_dir)
repo_dir = os.path.dirname(scenarionet_dir)
sys.path.insert(0, repo_dir)

import numpy as np

# Direct imports to avoid __init__.py which requires metadrive
from scenarionet.exporter.opendrive.exporter import OpenDriveExporter
from scenarionet.exporter.openscenario.exporter import OpenScenarioExporter


def create_intersection_scenario():
    """Create a synthetic intersection scenario with multiple vehicles."""

    track_length = 100  # 10 seconds at 10Hz
    timesteps = np.linspace(0, 10, track_length)

    # Ego vehicle: driving straight through intersection
    ego_x = np.linspace(-50, 50, track_length)
    ego_y = np.zeros(track_length)

    ego_track = {
        "type": "VEHICLE",
        "state": {
            "position": np.column_stack([ego_x, ego_y, np.zeros(track_length)]),
            "heading": np.zeros(track_length),  # Heading east
            "velocity": np.column_stack([
                np.full(track_length, 10.0),  # 10 m/s = 36 km/h
                np.zeros(track_length),
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
            "dataset": "synthetic",
        },
    }

    # Vehicle 2: Approaching from the right, then stopping
    v2_y = np.linspace(50, 5, track_length)  # Approaching from north
    v2_x = np.zeros(track_length)
    # Slow down as approaching intersection
    v2_speed = np.linspace(12, 0, track_length)

    vehicle2_track = {
        "type": "VEHICLE",
        "state": {
            "position": np.column_stack([v2_x, v2_y, np.zeros(track_length)]),
            "heading": np.full(track_length, -np.pi/2),  # Heading south
            "velocity": np.column_stack([
                np.zeros(track_length),
                -v2_speed,
            ]),
            "valid": np.ones(track_length, dtype=bool),
            "length": np.full(track_length, 4.8),
            "width": np.full(track_length, 1.9),
            "height": np.full(track_length, 1.6),
        },
        "metadata": {
            "track_length": track_length,
            "type": "VEHICLE",
            "object_id": "vehicle_2",
            "dataset": "synthetic",
        },
    }

    # Vehicle 3: Turning left
    t = np.linspace(0, np.pi/2, track_length)
    v3_x = -20 * np.cos(t) - 30
    v3_y = 20 * np.sin(t) - 20
    v3_heading = t  # Rotating from east to north

    vehicle3_track = {
        "type": "VEHICLE",
        "state": {
            "position": np.column_stack([v3_x, v3_y, np.zeros(track_length)]),
            "heading": v3_heading,
            "velocity": np.column_stack([
                8 * np.cos(v3_heading),
                8 * np.sin(v3_heading),
            ]),
            "valid": np.ones(track_length, dtype=bool),
            "length": np.full(track_length, 4.2),
            "width": np.full(track_length, 1.7),
            "height": np.full(track_length, 1.4),
        },
        "metadata": {
            "track_length": track_length,
            "type": "VEHICLE",
            "object_id": "vehicle_3",
            "dataset": "synthetic",
        },
    }

    # Pedestrian crossing the street
    ped_x = np.linspace(20, 25, track_length)
    ped_y = np.linspace(-8, 8, track_length)

    pedestrian_track = {
        "type": "PEDESTRIAN",
        "state": {
            "position": np.column_stack([ped_x, ped_y, np.zeros(track_length)]),
            "heading": np.full(track_length, np.pi/2),  # Heading north
            "velocity": np.column_stack([
                np.full(track_length, 0.5),
                np.full(track_length, 1.6),
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
            "dataset": "synthetic",
        },
    }

    # Cyclist on the side of the road
    cyc_x = np.linspace(-40, 40, track_length)
    cyc_y = np.full(track_length, -5)

    cyclist_track = {
        "type": "CYCLIST",
        "state": {
            "position": np.column_stack([cyc_x, cyc_y, np.zeros(track_length)]),
            "heading": np.zeros(track_length),
            "velocity": np.column_stack([
                np.full(track_length, 8.0),
                np.zeros(track_length),
            ]),
            "valid": np.ones(track_length, dtype=bool),
            "length": np.full(track_length, 1.8),
            "width": np.full(track_length, 0.6),
            "height": np.full(track_length, 1.7),
        },
        "metadata": {
            "track_length": track_length,
            "type": "CYCLIST",
            "object_id": "cyclist_1",
            "dataset": "synthetic",
        },
    }

    # Map features - Create an intersection
    # East-West main road
    lane_ew_1 = {
        "type": "LANE_SURFACE_STREET",
        "polyline": np.column_stack([
            np.linspace(-100, -10, 50),
            np.full(50, -1.8),
            np.zeros(50),
        ]),
        "speed_limit_kmh": 50.0,
        "entry_lanes": [],
        "exit_lanes": ["lane_junction_1"],
        "width": np.column_stack([np.full(50, 1.8), np.full(50, 1.8)]),
    }

    lane_ew_2 = {
        "type": "LANE_SURFACE_STREET",
        "polyline": np.column_stack([
            np.linspace(10, 100, 50),
            np.full(50, -1.8),
            np.zeros(50),
        ]),
        "speed_limit_kmh": 50.0,
        "entry_lanes": ["lane_junction_1"],
        "exit_lanes": [],
        "width": np.column_stack([np.full(50, 1.8), np.full(50, 1.8)]),
    }

    # Junction connector lane
    lane_junction_1 = {
        "type": "LANE_SURFACE_STREET",
        "polyline": np.column_stack([
            np.linspace(-10, 10, 20),
            np.full(20, -1.8),
            np.zeros(20),
        ]),
        "speed_limit_kmh": 30.0,
        "entry_lanes": ["lane_ew_1"],
        "exit_lanes": ["lane_ew_2"],
        "width": np.column_stack([np.full(20, 1.8), np.full(20, 1.8)]),
    }

    # North-South road
    lane_ns_1 = {
        "type": "LANE_SURFACE_STREET",
        "polyline": np.column_stack([
            np.full(50, 1.8),
            np.linspace(100, 10, 50),
            np.zeros(50),
        ]),
        "speed_limit_kmh": 50.0,
        "entry_lanes": [],
        "exit_lanes": ["lane_junction_2"],
        "width": np.column_stack([np.full(50, 1.8), np.full(50, 1.8)]),
    }

    lane_ns_2 = {
        "type": "LANE_SURFACE_STREET",
        "polyline": np.column_stack([
            np.full(50, 1.8),
            np.linspace(-10, -100, 50),
            np.zeros(50),
        ]),
        "speed_limit_kmh": 50.0,
        "entry_lanes": ["lane_junction_2"],
        "exit_lanes": [],
        "width": np.column_stack([np.full(50, 1.8), np.full(50, 1.8)]),
    }

    lane_junction_2 = {
        "type": "LANE_SURFACE_STREET",
        "polyline": np.column_stack([
            np.full(20, 1.8),
            np.linspace(10, -10, 20),
            np.zeros(20),
        ]),
        "speed_limit_kmh": 30.0,
        "entry_lanes": ["lane_ns_1"],
        "exit_lanes": ["lane_ns_2"],
        "width": np.column_stack([np.full(20, 1.8), np.full(20, 1.8)]),
    }

    # Road markings
    road_line_center = {
        "type": "ROAD_LINE_SOLID_DOUBLE_YELLOW",
        "polyline": np.column_stack([
            np.linspace(-100, 100, 100),
            np.zeros(100),
            np.zeros(100),
        ]),
    }

    road_line_edge_south = {
        "type": "ROAD_LINE_SOLID_SINGLE_WHITE",
        "polyline": np.column_stack([
            np.linspace(-100, 100, 100),
            np.full(100, -3.6),
            np.zeros(100),
        ]),
    }

    # Crosswalk
    crosswalk = {
        "type": "CROSSWALK",
        "polygon": np.array([
            [18, -4, 0],
            [22, -4, 0],
            [22, 4, 0],
            [18, 4, 0],
        ]),
    }

    # Stop sign
    stop_sign = {
        "type": "STOP_SIGN",
        "position": np.array([8, 15, 0]),
        "lane": ["lane_ns_1"],
    }

    # Traffic light states
    traffic_light = {
        "type": "TRAFFIC_LIGHT",
        "state": {
            "object_state": (
                ["LANE_STATE_GO"] * 40 +
                ["LANE_STATE_CAUTION"] * 10 +
                ["LANE_STATE_STOP"] * 50
            ),
        },
        "lane": "lane_ew_1",
        "stop_point": np.array([-12, -1.8, 0]),
        "metadata": {
            "track_length": track_length,
            "type": "TRAFFIC_LIGHT",
            "object_id": "traffic_light_1",
            "dataset": "synthetic",
        },
    }

    # Assemble scenario
    scenario = {
        "id": "intersection_scenario",
        "version": "synthetic_v1.0",
        "length": track_length,
        "tracks": {
            "ego": ego_track,
            "vehicle_2": vehicle2_track,
            "vehicle_3": vehicle3_track,
            "pedestrian_1": pedestrian_track,
            "cyclist_1": cyclist_track,
        },
        "map_features": {
            "lane_ew_1": lane_ew_1,
            "lane_ew_2": lane_ew_2,
            "lane_junction_1": lane_junction_1,
            "lane_ns_1": lane_ns_1,
            "lane_ns_2": lane_ns_2,
            "lane_junction_2": lane_junction_2,
            "road_line_center": road_line_center,
            "road_line_edge_south": road_line_edge_south,
            "crosswalk_1": crosswalk,
            "stop_sign_1": stop_sign,
        },
        "dynamic_map_states": {
            "traffic_light_1": traffic_light,
        },
        "metadata": {
            "id": "intersection_scenario",
            "scenario_id": "intersection_scenario",
            "dataset": "synthetic",
            "coordinate": "right-handed",
            "timestep": timesteps,
            "sdc_id": "ego",
            "track_length": track_length,
        },
    }

    return scenario


def main():
    # Create output directory
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "examples",
        "openx_export"
    )
    os.makedirs(output_dir, exist_ok=True)

    print("Creating synthetic intersection scenario...")
    scenario = create_intersection_scenario()

    # Export OpenDRIVE
    xodr_path = os.path.join(output_dir, "intersection_scenario.xodr")
    print(f"Exporting OpenDRIVE to: {xodr_path}")
    OpenDriveExporter.export_scenario(scenario, xodr_path)

    # Export OpenSCENARIO
    xosc_path = os.path.join(output_dir, "intersection_scenario.xosc")
    print(f"Exporting OpenSCENARIO to: {xosc_path}")
    OpenScenarioExporter.export_scenario(
        scenario, xosc_path,
        opendrive_path="intersection_scenario.xodr"
    )

    # Verify files
    print("\n=== Export Summary ===")
    for path in [xodr_path, xosc_path]:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"✓ {os.path.basename(path)}: {size:,} bytes")
        else:
            print(f"✗ {os.path.basename(path)}: NOT CREATED")

    # Display file contents
    print("\n=== OpenDRIVE Content (first 50 lines) ===")
    with open(xodr_path) as f:
        for i, line in enumerate(f):
            if i >= 50:
                print("... (truncated)")
                break
            print(line.rstrip())

    print("\n=== OpenSCENARIO Content (first 80 lines) ===")
    with open(xosc_path) as f:
        for i, line in enumerate(f):
            if i >= 80:
                print("... (truncated)")
                break
            print(line.rstrip())

    print(f"\n✓ Example files saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
