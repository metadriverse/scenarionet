"""
OpenSCENARIO Exporter

Converts ScenarioNet scenarios to OpenSCENARIO 1.0/1.1 format.

OpenSCENARIO is an open format for the description of dynamic driving scenarios.
It is used to define test scenarios for autonomous driving validation.

Reference: https://www.asam.net/standards/detail/openscenario/
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


class OpenScenarioExporter:
    """
    Exports ScenarioNet scenarios to OpenSCENARIO format.

    The exporter converts trajectory data and dynamic states into
    OpenSCENARIO's storyboard structure with actors and maneuvers.
    """

    # OpenSCENARIO version
    OPENSCENARIO_VERSION = "1.1"

    # Entity type mapping from MetaDrive types to OpenSCENARIO
    ENTITY_TYPE_MAP = {
        "VEHICLE": ("Vehicle", "car"),
        "PEDESTRIAN": ("Pedestrian", "pedestrian"),
        "CYCLIST": ("MiscObject", "bicycle"),
        "TRAFFIC_CONE": ("MiscObject", "obstacle"),
        "TRAFFIC_BARRIER": ("MiscObject", "barrier"),
        "BUS": ("Vehicle", "bus"),
        "TRUCK": ("Vehicle", "truck"),
        "MOTORCYCLE": ("Vehicle", "motorbike"),
        "UNSET": ("MiscObject", "none"),
    }

    # Default vehicle dimensions (length, width, height)
    DEFAULT_VEHICLE_DIMENSIONS = {
        "car": (4.5, 1.8, 1.5),
        "bus": (12.0, 2.5, 3.2),
        "truck": (10.0, 2.5, 3.5),
        "motorbike": (2.2, 0.8, 1.5),
    }

    DEFAULT_PEDESTRIAN_DIMENSIONS = (0.5, 0.5, 1.8)
    DEFAULT_MISC_DIMENSIONS = (1.0, 1.0, 1.0)

    def __init__(self, scenario: Dict[str, Any], opendrive_path: Optional[str] = None):
        """
        Initialize the exporter with a ScenarioNet scenario.

        Args:
            scenario: A ScenarioNet scenario dict containing TRACKS, DYNAMIC_MAP_STATES, etc.
            opendrive_path: Optional path to the corresponding OpenDRIVE file
        """
        self.scenario = scenario
        self.tracks = scenario.get("tracks", {})
        self.dynamic_map_states = scenario.get("dynamic_map_states", {})
        self.metadata = scenario.get("metadata", {})
        self.opendrive_path = opendrive_path

        # Get scenario timing
        self.timesteps = self.metadata.get("timestep", np.arange(0, 10, 0.1))
        if isinstance(self.timesteps, list):
            self.timesteps = np.array(self.timesteps)

        self.track_length = scenario.get("length", len(self.timesteps))
        self.sdc_id = str(self.metadata.get("sdc_id", "ego"))

        # Computed during export
        self.entities = {}
        self.trajectories = {}

    def _get_entity_type(self, track_type: str) -> Tuple[str, str]:
        """Get OpenSCENARIO entity category and model type from track type."""
        # Handle both direct type strings and MetaDrive type format
        type_key = track_type.upper() if isinstance(track_type, str) else "VEHICLE"

        for key, value in self.ENTITY_TYPE_MAP.items():
            if key in type_key:
                return value

        return ("Vehicle", "car")  # Default

    def _get_dimensions(self, track: Dict, entity_type: Tuple[str, str]) -> Tuple[float, float, float]:
        """Get entity dimensions from track data or defaults."""
        state = track.get("state", {})

        # Try to get from track state
        length = state.get("length", None)
        width = state.get("width", None)
        height = state.get("height", None)

        # Handle array values (take first valid value)
        if isinstance(length, np.ndarray):
            valid_mask = length > 0
            length = float(length[valid_mask][0]) if valid_mask.any() else None
        if isinstance(width, np.ndarray):
            valid_mask = width > 0
            width = float(width[valid_mask][0]) if valid_mask.any() else None
        if isinstance(height, np.ndarray):
            valid_mask = height > 0
            height = float(height[valid_mask][0]) if valid_mask.any() else None

        # Use defaults if not available
        category, model = entity_type

        if category == "Vehicle":
            default = self.DEFAULT_VEHICLE_DIMENSIONS.get(model, (4.5, 1.8, 1.5))
        elif category == "Pedestrian":
            default = self.DEFAULT_PEDESTRIAN_DIMENSIONS
        else:
            default = self.DEFAULT_MISC_DIMENSIONS

        return (
            length if length and length > 0 else default[0],
            width if width and width > 0 else default[1],
            height if height and height > 0 else default[2],
        )

    def _extract_trajectory(self, track_id: str, track: Dict) -> Dict:
        """Extract trajectory data from a track."""
        state = track.get("state", {})
        position = state.get("position", np.array([]))
        heading = state.get("heading", np.array([]))
        velocity = state.get("velocity", np.array([]))
        valid = state.get("valid", np.ones(len(position), dtype=bool))

        # Build trajectory points
        trajectory_points = []

        for i in range(min(len(position), self.track_length)):
            if not valid[i]:
                continue

            pos = position[i]
            hdg = heading[i] if i < len(heading) else 0.0
            vel = velocity[i] if i < len(velocity) else np.array([0.0, 0.0])

            # Compute speed from velocity
            if isinstance(vel, np.ndarray) and len(vel) >= 2:
                speed = float(np.sqrt(vel[0]**2 + vel[1]**2))
            else:
                speed = 0.0

            # Get time from timesteps
            time = float(self.timesteps[i]) if i < len(self.timesteps) else i * 0.1

            trajectory_points.append({
                "time": time,
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]) if len(pos) > 2 else 0.0,
                "heading": float(hdg),
                "speed": speed,
            })

        return {
            "track_id": track_id,
            "points": trajectory_points,
        }

    def _build_entities(self):
        """Build entity definitions from tracks."""
        self.entities = {}
        self.trajectories = {}

        for track_id, track in self.tracks.items():
            track_type = track.get("type", "VEHICLE")
            entity_type = self._get_entity_type(track_type)
            dimensions = self._get_dimensions(track, entity_type)

            # Generate entity name
            if str(track_id) == str(self.sdc_id):
                entity_name = "Ego"
            else:
                entity_name = f"Entity_{track_id}"

            self.entities[track_id] = {
                "name": entity_name,
                "category": entity_type[0],
                "model": entity_type[1],
                "length": dimensions[0],
                "width": dimensions[1],
                "height": dimensions[2],
                "is_ego": str(track_id) == str(self.sdc_id),
            }

            # Extract trajectory
            trajectory = self._extract_trajectory(track_id, track)
            if trajectory["points"]:
                self.trajectories[track_id] = trajectory

    def _create_file_header(self, root: ET.Element):
        """Create the FileHeader element."""
        header = ET.SubElement(root, "FileHeader")
        header.set("revMajor", "1")
        header.set("revMinor", "1")
        header.set("date", datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
        header.set("description", f"ScenarioNet Export - {self.metadata.get('scenario_id', 'unknown')}")
        header.set("author", "ScenarioNet")

    def _create_parameter_declarations(self, root: ET.Element):
        """Create ParameterDeclarations element."""
        param_decl = ET.SubElement(root, "ParameterDeclarations")
        # Can add parameters here if needed

    def _create_catalog_locations(self, root: ET.Element):
        """Create CatalogLocations element."""
        catalog_locs = ET.SubElement(root, "CatalogLocations")
        # Empty by default, can reference external catalogs

    def _create_road_network(self, root: ET.Element):
        """Create RoadNetwork element."""
        road_network = ET.SubElement(root, "RoadNetwork")

        if self.opendrive_path:
            logic_file = ET.SubElement(road_network, "LogicFile")
            logic_file.set("filepath", self.opendrive_path)
        else:
            # Use scene graph file reference or empty
            scene_graph = ET.SubElement(road_network, "SceneGraphFile")
            scene_graph.set("filepath", "")

    def _create_entities(self, root: ET.Element):
        """Create Entities element with all scenario participants."""
        entities = ET.SubElement(root, "Entities")

        for track_id, entity in self.entities.items():
            scenario_obj = ET.SubElement(entities, "ScenarioObject")
            scenario_obj.set("name", entity["name"])

            if entity["category"] == "Vehicle":
                vehicle = ET.SubElement(scenario_obj, "Vehicle")
                vehicle.set("name", entity["name"])
                vehicle.set("vehicleCategory", entity["model"])

                # Bounding box
                bbox = ET.SubElement(vehicle, "BoundingBox")
                center = ET.SubElement(bbox, "Center")
                center.set("x", f"{entity['length']/2:.2f}")
                center.set("y", "0.0")
                center.set("z", f"{entity['height']/2:.2f}")

                dimensions = ET.SubElement(bbox, "Dimensions")
                dimensions.set("width", f"{entity['width']:.2f}")
                dimensions.set("length", f"{entity['length']:.2f}")
                dimensions.set("height", f"{entity['height']:.2f}")

                # Performance
                performance = ET.SubElement(vehicle, "Performance")
                performance.set("maxSpeed", "50")
                performance.set("maxAcceleration", "10")
                performance.set("maxDeceleration", "10")

                # Axles
                axles = ET.SubElement(vehicle, "Axles")
                front = ET.SubElement(axles, "FrontAxle")
                front.set("maxSteering", "0.5")
                front.set("wheelDiameter", "0.6")
                front.set("trackWidth", f"{entity['width'] * 0.8:.2f}")
                front.set("positionX", f"{entity['length'] * 0.7:.2f}")
                front.set("positionZ", "0.3")

                rear = ET.SubElement(axles, "RearAxle")
                rear.set("maxSteering", "0")
                rear.set("wheelDiameter", "0.6")
                rear.set("trackWidth", f"{entity['width'] * 0.8:.2f}")
                rear.set("positionX", f"{entity['length'] * 0.2:.2f}")
                rear.set("positionZ", "0.3")

                # Properties
                properties = ET.SubElement(vehicle, "Properties")
                prop = ET.SubElement(properties, "Property")
                prop.set("name", "type")
                prop.set("value", entity["model"])

            elif entity["category"] == "Pedestrian":
                pedestrian = ET.SubElement(scenario_obj, "Pedestrian")
                pedestrian.set("name", entity["name"])
                pedestrian.set("model", "pedestrian")
                pedestrian.set("pedestrianCategory", "pedestrian")
                pedestrian.set("mass", "70")

                bbox = ET.SubElement(pedestrian, "BoundingBox")
                center = ET.SubElement(bbox, "Center")
                center.set("x", "0")
                center.set("y", "0")
                center.set("z", f"{entity['height']/2:.2f}")

                dimensions = ET.SubElement(bbox, "Dimensions")
                dimensions.set("width", f"{entity['width']:.2f}")
                dimensions.set("length", f"{entity['length']:.2f}")
                dimensions.set("height", f"{entity['height']:.2f}")

                properties = ET.SubElement(pedestrian, "Properties")

            else:  # MiscObject
                misc = ET.SubElement(scenario_obj, "MiscObject")
                misc.set("name", entity["name"])
                misc.set("miscObjectCategory", entity["model"])
                misc.set("mass", "100")

                bbox = ET.SubElement(misc, "BoundingBox")
                center = ET.SubElement(bbox, "Center")
                center.set("x", "0")
                center.set("y", "0")
                center.set("z", f"{entity['height']/2:.2f}")

                dimensions = ET.SubElement(bbox, "Dimensions")
                dimensions.set("width", f"{entity['width']:.2f}")
                dimensions.set("length", f"{entity['length']:.2f}")
                dimensions.set("height", f"{entity['height']:.2f}")

                properties = ET.SubElement(misc, "Properties")

    def _create_init_actions(self, init: ET.Element):
        """Create initial position actions for all entities."""
        actions = ET.SubElement(init, "Actions")

        for track_id, entity in self.entities.items():
            trajectory = self.trajectories.get(track_id)
            if not trajectory or not trajectory["points"]:
                continue

            # Get initial position
            initial = trajectory["points"][0]

            private = ET.SubElement(actions, "Private")
            private.set("entityRef", entity["name"])

            # Teleport action for initial position
            private_action = ET.SubElement(private, "PrivateAction")
            teleport = ET.SubElement(private_action, "TeleportAction")
            position = ET.SubElement(teleport, "Position")
            world_pos = ET.SubElement(position, "WorldPosition")
            world_pos.set("x", f"{initial['x']:.6f}")
            world_pos.set("y", f"{initial['y']:.6f}")
            world_pos.set("z", f"{initial['z']:.6f}")
            world_pos.set("h", f"{initial['heading']:.6f}")
            world_pos.set("p", "0")
            world_pos.set("r", "0")

            # Initial speed action
            if initial["speed"] > 0:
                speed_action = ET.SubElement(private, "PrivateAction")
                longitudinal = ET.SubElement(speed_action, "LongitudinalAction")
                speed_act = ET.SubElement(longitudinal, "SpeedAction")

                speed_dynamics = ET.SubElement(speed_act, "SpeedActionDynamics")
                speed_dynamics.set("dynamicsShape", "step")
                speed_dynamics.set("value", "0")
                speed_dynamics.set("dynamicsDimension", "time")

                speed_target = ET.SubElement(speed_act, "SpeedActionTarget")
                absolute = ET.SubElement(speed_target, "AbsoluteTargetSpeed")
                absolute.set("value", f"{initial['speed']:.2f}")

    def _create_trajectory_action(self, maneuver: ET.Element, entity_name: str, trajectory: Dict):
        """Create a FollowTrajectoryAction for an entity."""
        event = ET.SubElement(maneuver, "Event")
        event.set("name", f"TrajectoryEvent_{entity_name}")
        event.set("priority", "overwrite")

        # Action
        action = ET.SubElement(event, "Action")
        action.set("name", f"TrajectoryAction_{entity_name}")

        private_action = ET.SubElement(action, "PrivateAction")
        routing = ET.SubElement(private_action, "RoutingAction")
        follow_trajectory = ET.SubElement(routing, "FollowTrajectoryAction")

        # Trajectory definition
        traj = ET.SubElement(follow_trajectory, "Trajectory")
        traj.set("name", f"Trajectory_{entity_name}")
        traj.set("closed", "false")

        # Trajectory shape using polyline
        shape = ET.SubElement(traj, "Shape")
        polyline = ET.SubElement(shape, "Polyline")

        for point in trajectory["points"]:
            vertex = ET.SubElement(polyline, "Vertex")
            vertex.set("time", f"{point['time']:.3f}")

            position = ET.SubElement(vertex, "Position")
            world_pos = ET.SubElement(position, "WorldPosition")
            world_pos.set("x", f"{point['x']:.6f}")
            world_pos.set("y", f"{point['y']:.6f}")
            world_pos.set("z", f"{point['z']:.6f}")
            world_pos.set("h", f"{point['heading']:.6f}")
            world_pos.set("p", "0")
            world_pos.set("r", "0")

        # Timing mode
        timing_ref = ET.SubElement(follow_trajectory, "TimeReference")
        timing = ET.SubElement(timing_ref, "Timing")
        timing.set("domainAbsoluteRelative", "absolute")
        timing.set("scale", "1.0")
        timing.set("offset", "0.0")

        # Trajectory following mode
        following_mode = ET.SubElement(follow_trajectory, "TrajectoryFollowingMode")
        following_mode.set("followingMode", "position")

        # Start trigger - simulation time = 0
        start_trigger = ET.SubElement(event, "StartTrigger")
        condition_group = ET.SubElement(start_trigger, "ConditionGroup")
        condition = ET.SubElement(condition_group, "Condition")
        condition.set("name", "StartCondition")
        condition.set("delay", "0")
        condition.set("conditionEdge", "rising")

        by_value = ET.SubElement(condition, "ByValueCondition")
        sim_time = ET.SubElement(by_value, "SimulationTimeCondition")
        sim_time.set("value", "0")
        sim_time.set("rule", "greaterOrEqual")

    def _create_storyboard(self, root: ET.Element):
        """Create the Storyboard element with all scenario dynamics."""
        storyboard = ET.SubElement(root, "Storyboard")

        # Init section
        init = ET.SubElement(storyboard, "Init")
        self._create_init_actions(init)

        # Story for trajectory following
        story = ET.SubElement(storyboard, "Story")
        story.set("name", "ScenarioNet_Story")

        # Create acts for each entity with trajectory
        for track_id, entity in self.entities.items():
            trajectory = self.trajectories.get(track_id)
            if not trajectory or len(trajectory["points"]) < 2:
                continue

            # Act for this entity
            act = ET.SubElement(story, "Act")
            act.set("name", f"Act_{entity['name']}")

            # ManeuverGroup
            maneuver_group = ET.SubElement(act, "ManeuverGroup")
            maneuver_group.set("name", f"ManeuverGroup_{entity['name']}")
            maneuver_group.set("maximumExecutionCount", "1")

            # Actors
            actors = ET.SubElement(maneuver_group, "Actors")
            actors.set("selectTriggeringEntities", "false")
            entity_ref = ET.SubElement(actors, "EntityRef")
            entity_ref.set("entityRef", entity["name"])

            # Maneuver with trajectory action
            maneuver = ET.SubElement(maneuver_group, "Maneuver")
            maneuver.set("name", f"Maneuver_{entity['name']}")

            self._create_trajectory_action(maneuver, entity["name"], trajectory)

            # Act start trigger
            start_trigger = ET.SubElement(act, "StartTrigger")
            condition_group = ET.SubElement(start_trigger, "ConditionGroup")
            condition = ET.SubElement(condition_group, "Condition")
            condition.set("name", "ActStartCondition")
            condition.set("delay", "0")
            condition.set("conditionEdge", "rising")

            by_value = ET.SubElement(condition, "ByValueCondition")
            sim_time = ET.SubElement(by_value, "SimulationTimeCondition")
            sim_time.set("value", "0")
            sim_time.set("rule", "greaterOrEqual")

        # Stop trigger for the storyboard
        stop_trigger = ET.SubElement(storyboard, "StopTrigger")
        condition_group = ET.SubElement(stop_trigger, "ConditionGroup")
        condition = ET.SubElement(condition_group, "Condition")
        condition.set("name", "StopCondition")
        condition.set("delay", "0")
        condition.set("conditionEdge", "rising")

        # Stop when simulation time exceeds scenario length
        by_value = ET.SubElement(condition, "ByValueCondition")
        sim_time = ET.SubElement(by_value, "SimulationTimeCondition")
        max_time = float(self.timesteps[-1]) if len(self.timesteps) > 0 else 10.0
        sim_time.set("value", f"{max_time:.1f}")
        sim_time.set("rule", "greaterOrEqual")

    def export(self, output_path: str, pretty_print: bool = True) -> str:
        """
        Export the scenario to OpenSCENARIO format.

        Args:
            output_path: Path to save the .xosc file
            pretty_print: Whether to format the XML with indentation

        Returns:
            Path to the exported file
        """
        # Build entities and trajectories
        self._build_entities()

        # Create root element
        root = ET.Element("OpenSCENARIO")

        # Create all sections
        self._create_file_header(root)
        self._create_parameter_declarations(root)
        self._create_catalog_locations(root)
        self._create_road_network(root)
        self._create_entities(root)
        self._create_storyboard(root)

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

        logger.info(f"OpenSCENARIO file exported to: {output_path}")
        return output_path

    @classmethod
    def export_scenario(
        cls,
        scenario: Dict[str, Any],
        output_path: str,
        opendrive_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Class method to export a scenario directly.

        Args:
            scenario: ScenarioNet scenario dict
            output_path: Path to save the .xosc file
            opendrive_path: Optional path to corresponding OpenDRIVE file
            **kwargs: Additional arguments passed to export()

        Returns:
            Path to the exported file
        """
        exporter = cls(scenario, opendrive_path=opendrive_path)
        return exporter.export(output_path, **kwargs)
