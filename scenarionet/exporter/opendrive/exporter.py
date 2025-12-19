"""
OpenDRIVE Exporter

Converts ScenarioNet map features to OpenDRIVE 1.6 format.

OpenDRIVE is an open format for the logical description of road networks.
It is used by various driving simulation tools and autonomous driving platforms.

Reference: https://www.asam.net/standards/detail/opendrive/
"""

import logging
import math
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from xml.dom import minidom

import numpy as np

logger = logging.getLogger(__name__)


class OpenDriveExporter:
    """
    Exports ScenarioNet map features to OpenDRIVE format.

    The exporter converts the lane-centric representation used by ScenarioNet
    to the road-centric representation used by OpenDRIVE.
    """

    # OpenDRIVE version
    OPENDRIVE_VERSION = "1.6"

    # Lane type mapping from ScenarioNet/MetaDrive types to OpenDRIVE
    LANE_TYPE_MAP = {
        "LANE_SURFACE_STREET": "driving",
        "LANE_FREEWAY": "driving",
        "LANE_BIKE_LANE": "biking",
        "LANE_UNKNOWN": "driving",
        "LANE_SURFACE_UNSTRUCTURE": "driving",
    }

    # Road mark type mapping
    ROAD_MARK_TYPE_MAP = {
        "ROAD_LINE_BROKEN_SINGLE_WHITE": ("broken", "standard", "white"),
        "ROAD_LINE_SOLID_SINGLE_WHITE": ("solid", "standard", "white"),
        "ROAD_LINE_SOLID_DOUBLE_WHITE": ("solid solid", "standard", "white"),
        "ROAD_LINE_BROKEN_SINGLE_YELLOW": ("broken", "standard", "yellow"),
        "ROAD_LINE_BROKEN_DOUBLE_YELLOW": ("broken broken", "standard", "yellow"),
        "ROAD_LINE_SOLID_SINGLE_YELLOW": ("solid", "standard", "yellow"),
        "ROAD_LINE_SOLID_DOUBLE_YELLOW": ("solid solid", "standard", "yellow"),
        "ROAD_LINE_PASSING_DOUBLE_YELLOW": ("solid broken", "standard", "yellow"),
        "UNKNOWN": ("none", "standard", "white"),
        "LINE_UNKNOWN": ("none", "standard", "white"),
    }

    def __init__(self, scenario: Dict[str, Any]):
        """
        Initialize the exporter with a ScenarioNet scenario.

        Args:
            scenario: A ScenarioNet scenario dict containing MAP_FEATURES
        """
        self.scenario = scenario
        self.map_features = scenario.get("map_features", {})
        self.metadata = scenario.get("metadata", {})

        # Extract lanes, road lines, and other features
        self.lanes = {}
        self.road_lines = {}
        self.road_edges = {}
        self.crosswalks = {}
        self.stop_signs = {}
        self.speed_bumps = {}
        self.driveways = {}

        self._categorize_features()

        # Road network structure (computed during export)
        self.roads = {}
        self.junctions = {}

    def _categorize_features(self):
        """Categorize map features by type."""
        for feat_id, feature in self.map_features.items():
            feat_type = feature.get("type", "")

            if "LANE" in feat_type:
                self.lanes[feat_id] = feature
            elif "ROAD_LINE" in feat_type or "LINE" in feat_type:
                self.road_lines[feat_id] = feature
            elif "ROAD_EDGE" in feat_type or "EDGE" in feat_type:
                self.road_edges[feat_id] = feature
            elif feat_type == "CROSSWALK":
                self.crosswalks[feat_id] = feature
            elif feat_type == "STOP_SIGN":
                self.stop_signs[feat_id] = feature
            elif feat_type == "SPEED_BUMP":
                self.speed_bumps[feat_id] = feature
            elif feat_type == "DRIVEWAY":
                self.driveways[feat_id] = feature

    def _compute_polyline_length(self, polyline: np.ndarray) -> float:
        """Compute the total length of a polyline."""
        if len(polyline) < 2:
            return 0.0
        diffs = np.diff(polyline[:, :2], axis=0)
        lengths = np.sqrt(np.sum(diffs**2, axis=1))
        return float(np.sum(lengths))

    def _compute_heading(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Compute heading angle from p1 to p2 in radians."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.atan2(dy, dx)

    def _polyline_to_geometry(self, polyline: np.ndarray) -> List[Dict]:
        """
        Convert a polyline to OpenDRIVE geometry elements.

        Uses line segments to approximate the polyline.
        Each segment is represented as a <geometry> element with type "line".

        Args:
            polyline: Nx2 or Nx3 array of points

        Returns:
            List of geometry dictionaries with s, x, y, hdg, length, type
        """
        geometries = []
        s = 0.0

        for i in range(len(polyline) - 1):
            p1 = polyline[i]
            p2 = polyline[i + 1]

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx**2 + dy**2)

            if length < 1e-6:
                continue

            hdg = math.atan2(dy, dx)

            geometries.append({
                "s": s,
                "x": float(p1[0]),
                "y": float(p1[1]),
                "hdg": hdg,
                "length": length,
                "type": "line"
            })

            s += length

        return geometries

    def _build_road_network(self):
        """
        Build road network from lanes.

        Groups lanes into roads based on connectivity and parallel lanes.
        In this simplified implementation, each lane becomes a road with
        a single lane section.
        """
        self.roads = {}
        road_id = 0

        for lane_id, lane in self.lanes.items():
            polyline = lane.get("polyline", np.array([]))
            if len(polyline) < 2:
                continue

            road_length = self._compute_polyline_length(polyline)
            if road_length < 0.1:
                continue

            # Get lane properties
            lane_type = lane.get("type", "LANE_SURFACE_STREET")
            speed_limit = lane.get("speed_limit_kmh", lane.get("speed_limit_mps", 0) * 3.6)
            if speed_limit is None:
                speed_limit = 50.0  # Default 50 km/h

            # Get width information
            width = lane.get("width", None)
            if width is not None and len(width) > 0:
                avg_width = float(np.mean(width[:, 0] + width[:, 1]))
                if avg_width < 0.1:
                    avg_width = 3.5  # Default lane width
            else:
                avg_width = 3.5

            self.roads[str(road_id)] = {
                "id": str(road_id),
                "name": f"Road_{lane_id}",
                "length": road_length,
                "lane_id": lane_id,
                "lane": lane,
                "polyline": polyline,
                "lane_type": lane_type,
                "speed_limit": speed_limit,
                "width": avg_width,
                "entry_lanes": lane.get("entry_lanes", []),
                "exit_lanes": lane.get("exit_lanes", []),
            }
            road_id += 1

        # Build junction information from lane connectivity
        self._build_junctions()

    def _build_junctions(self):
        """Build junctions from lane connectivity."""
        # Find lanes that have multiple entry or exit lanes (junction indicators)
        junction_lanes = set()

        for lane_id, lane in self.lanes.items():
            entry_lanes = lane.get("entry_lanes", [])
            exit_lanes = lane.get("exit_lanes", [])

            # A junction is indicated by multiple incoming or outgoing connections
            if len(entry_lanes) > 1 or len(exit_lanes) > 1:
                junction_lanes.add(lane_id)

        # Create junctions (simplified - group nearby junction lanes)
        junction_id = 0
        for lane_id in junction_lanes:
            lane = self.lanes[lane_id]
            polyline = lane.get("polyline", np.array([]))
            if len(polyline) < 2:
                continue

            # Get center point of lane
            center = polyline[len(polyline) // 2]

            self.junctions[str(junction_id)] = {
                "id": str(junction_id),
                "name": f"Junction_{junction_id}",
                "lane_id": lane_id,
                "center": center,
                "entry_lanes": lane.get("entry_lanes", []),
                "exit_lanes": lane.get("exit_lanes", []),
            }
            junction_id += 1

    def _create_header(self, root: ET.Element):
        """Create the OpenDRIVE header element."""
        header = ET.SubElement(root, "header")
        header.set("revMajor", "1")
        header.set("revMinor", "6")
        header.set("name", self.metadata.get("scenario_id", "ScenarioNet_Export"))
        header.set("version", "1.0")
        header.set("date", datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))

        # Add geo reference if available
        geo_ref = ET.SubElement(header, "geoReference")
        geo_ref.text = "+proj=utm +zone=10 +datum=WGS84"  # Default UTM projection

        # Add user data with ScenarioNet metadata
        user_data = ET.SubElement(header, "userData")
        user_data.set("code", "scenarionet")
        user_data.set("value", self.metadata.get("dataset", "unknown"))

    def _create_road(self, root: ET.Element, road_data: Dict):
        """Create a road element."""
        road = ET.SubElement(root, "road")
        road.set("id", road_data["id"])
        road.set("name", road_data["name"])
        road.set("length", f"{road_data['length']:.6f}")
        road.set("junction", "-1")  # -1 means not in a junction

        # Create type element with speed
        road_type = ET.SubElement(road, "type")
        road_type.set("s", "0")
        road_type.set("type", "town")  # Default road type

        speed = ET.SubElement(road_type, "speed")
        speed.set("max", f"{road_data['speed_limit']:.1f}")
        speed.set("unit", "km/h")

        # Create planView with geometry
        plan_view = ET.SubElement(road, "planView")
        geometries = self._polyline_to_geometry(road_data["polyline"])

        for geom in geometries:
            geometry = ET.SubElement(plan_view, "geometry")
            geometry.set("s", f"{geom['s']:.6f}")
            geometry.set("x", f"{geom['x']:.6f}")
            geometry.set("y", f"{geom['y']:.6f}")
            geometry.set("hdg", f"{geom['hdg']:.6f}")
            geometry.set("length", f"{geom['length']:.6f}")

            # Add geometry type (line for simplified export)
            line = ET.SubElement(geometry, "line")

        # Create elevation profile (flat by default)
        elevation_profile = ET.SubElement(road, "elevationProfile")
        elevation = ET.SubElement(elevation_profile, "elevation")
        elevation.set("s", "0")
        elevation.set("a", "0")  # elevation at s=0
        elevation.set("b", "0")  # slope
        elevation.set("c", "0")  # curvature
        elevation.set("d", "0")  # third derivative

        # Create lateral profile (flat by default)
        lateral_profile = ET.SubElement(road, "lateralProfile")

        # Create lanes
        lanes_elem = ET.SubElement(road, "lanes")

        # Lane offset (center of road)
        lane_offset = ET.SubElement(lanes_elem, "laneOffset")
        lane_offset.set("s", "0")
        lane_offset.set("a", "0")
        lane_offset.set("b", "0")
        lane_offset.set("c", "0")
        lane_offset.set("d", "0")

        # Lane section
        lane_section = ET.SubElement(lanes_elem, "laneSection")
        lane_section.set("s", "0")

        # Center lane (reference line)
        center = ET.SubElement(lane_section, "center")
        center_lane = ET.SubElement(center, "lane")
        center_lane.set("id", "0")
        center_lane.set("type", "none")
        center_lane.set("level", "false")

        # Right lane (driving lane)
        right = ET.SubElement(lane_section, "right")
        right_lane = ET.SubElement(right, "lane")
        right_lane.set("id", "-1")
        right_lane.set("type", self.LANE_TYPE_MAP.get(road_data["lane_type"], "driving"))
        right_lane.set("level", "false")

        # Lane width
        width = ET.SubElement(right_lane, "width")
        width.set("sOffset", "0")
        width.set("a", f"{road_data['width']:.3f}")
        width.set("b", "0")
        width.set("c", "0")
        width.set("d", "0")

        # Road mark
        road_mark = ET.SubElement(right_lane, "roadMark")
        road_mark.set("sOffset", "0")
        road_mark.set("type", "solid")
        road_mark.set("weight", "standard")
        road_mark.set("color", "white")
        road_mark.set("width", "0.15")

        # Create link element for connectivity
        link = ET.SubElement(road, "link")

        # Add predecessor/successor based on lane connectivity
        if road_data["entry_lanes"]:
            for entry_id in road_data["entry_lanes"]:
                # Find road that contains this entry lane
                for rid, rdata in self.roads.items():
                    if rdata["lane_id"] == str(entry_id):
                        predecessor = ET.SubElement(link, "predecessor")
                        predecessor.set("elementType", "road")
                        predecessor.set("elementId", rid)
                        predecessor.set("contactPoint", "end")
                        break

        if road_data["exit_lanes"]:
            for exit_id in road_data["exit_lanes"]:
                # Find road that contains this exit lane
                for rid, rdata in self.roads.items():
                    if rdata["lane_id"] == str(exit_id):
                        successor = ET.SubElement(link, "successor")
                        successor.set("elementType", "road")
                        successor.set("elementId", rid)
                        successor.set("contactPoint", "start")
                        break

        return road

    def _create_junction(self, root: ET.Element, junction_data: Dict):
        """Create a junction element."""
        junction = ET.SubElement(root, "junction")
        junction.set("id", junction_data["id"])
        junction.set("name", junction_data["name"])
        junction.set("type", "default")

        # Create connections
        connection_id = 0
        for entry_id in junction_data["entry_lanes"]:
            for exit_id in junction_data["exit_lanes"]:
                # Find corresponding roads
                incoming_road = None
                connecting_road = None

                for rid, rdata in self.roads.items():
                    if rdata["lane_id"] == str(entry_id):
                        incoming_road = rid
                    if rdata["lane_id"] == junction_data["lane_id"]:
                        connecting_road = rid

                if incoming_road and connecting_road:
                    connection = ET.SubElement(junction, "connection")
                    connection.set("id", str(connection_id))
                    connection.set("incomingRoad", incoming_road)
                    connection.set("connectingRoad", connecting_road)
                    connection.set("contactPoint", "start")

                    lane_link = ET.SubElement(connection, "laneLink")
                    lane_link.set("from", "-1")
                    lane_link.set("to", "-1")

                    connection_id += 1

        return junction

    def _create_objects(self, road: ET.Element, road_data: Dict):
        """Add objects (stop signs, etc.) to a road if nearby."""
        objects = ET.SubElement(road, "objects")

        obj_id = 0
        polyline = road_data["polyline"]

        # Check stop signs
        for sign_id, sign in self.stop_signs.items():
            position = sign.get("position", None)
            if position is None:
                continue

            # Check if stop sign is near this road
            min_dist = float('inf')
            nearest_s = 0
            s = 0

            for i in range(len(polyline) - 1):
                p1 = polyline[i]
                dist = np.sqrt((position[0] - p1[0])**2 + (position[1] - p1[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_s = s
                s += np.sqrt((polyline[i+1][0] - p1[0])**2 + (polyline[i+1][1] - p1[1])**2)

            if min_dist < 10.0:  # Within 10 meters
                obj = ET.SubElement(objects, "object")
                obj.set("id", str(obj_id))
                obj.set("name", f"StopSign_{sign_id}")
                obj.set("s", f"{nearest_s:.3f}")
                obj.set("t", f"{min_dist:.3f}")
                obj.set("zOffset", "0")
                obj.set("type", "signal")
                obj.set("subtype", "stopSign")
                obj.set("orientation", "+")
                obj.set("width", "0.6")
                obj.set("height", "0.6")
                obj_id += 1

        return objects

    def export(self, output_path: str, pretty_print: bool = True) -> str:
        """
        Export the scenario to OpenDRIVE format.

        Args:
            output_path: Path to save the .xodr file
            pretty_print: Whether to format the XML with indentation

        Returns:
            Path to the exported file
        """
        # Build road network
        self._build_road_network()

        # Create root element
        root = ET.Element("OpenDRIVE")

        # Create header
        self._create_header(root)

        # Create roads
        for road_id, road_data in self.roads.items():
            road_elem = self._create_road(root, road_data)
            self._create_objects(road_elem, road_data)

        # Create junctions
        for junction_id, junction_data in self.junctions.items():
            self._create_junction(root, junction_data)

        # Generate XML string
        if pretty_print:
            xml_str = minidom.parseString(ET.tostring(root, encoding='unicode')).toprettyxml(indent="  ")
            # Remove extra blank lines
            lines = [line for line in xml_str.split('\n') if line.strip()]
            xml_str = '\n'.join(lines)
        else:
            xml_str = ET.tostring(root, encoding='unicode')

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            # Remove the duplicate declaration from pretty print
            if xml_str.startswith('<?xml'):
                xml_str = xml_str.split('?>', 1)[1].strip()
            f.write(xml_str)

        logger.info(f"OpenDRIVE file exported to: {output_path}")
        return output_path

    @classmethod
    def export_scenario(cls, scenario: Dict[str, Any], output_path: str, **kwargs) -> str:
        """
        Class method to export a scenario directly.

        Args:
            scenario: ScenarioNet scenario dict
            output_path: Path to save the .xodr file
            **kwargs: Additional arguments passed to export()

        Returns:
            Path to the exported file
        """
        exporter = cls(scenario)
        return exporter.export(output_path, **kwargs)
