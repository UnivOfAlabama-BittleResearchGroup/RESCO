from dataclasses import dataclass
import queue
from typing import Dict, List, Set, Tuple, Union
from collections import UserDict
import traci
import copy
import re
from resco_benchmark.config.signal_config import signal_configs


REVERSED_DIRECTIONS = {"N": "S", "E": "W", "S": "N", "W": "E"}


def create_yellows(phases, yellow_length):
    new_phases = copy.copy(phases)
    yellow_dict = (
        {}
    )  # current phase + next phase keyed to corresponding yellow phase index
    # Automatically create yellow phases, traci will report missing phases as it assumes execution by index order
    for i in range(len(phases)):
        for j in range(len(phases)):
            if i != j:
                need_yellow, yellow_str = False, ""
                for sig_idx in range(len(phases[i].state)):
                    if phases[i].state[sig_idx] in ["G", "g"] and phases[j].state[
                        sig_idx
                    ] in ["r", "s"]:
                        need_yellow = True
                        yellow_str += "y"
                    else:
                        yellow_str += phases[i].state[sig_idx]
                if need_yellow:  # If a yellow is required
                    new_phases.append(
                        traci.trafficlight.Phase(yellow_length, yellow_str)
                    )
                    yellow_dict[f"{str(i)}_{str(j)}"] = len(new_phases) - 1
    return new_phases, yellow_dict


class _Dict:

    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __delitem__(self, key):
        del self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

@dataclass
class VehicleMeasures(_Dict):
    id: str
    speed: float
    wait: float
    acceleration: float
    position: float
    type: str


@dataclass
class LaneMeasures(_Dict):
    queue: int
    approach: int
    total_wait: int
    max_wait: int
    _vehicles: List[VehicleMeasures] = None


    def add_vehicle(self, veh: VehicleMeasures) -> None:
        if veh.wait > 0:
            self.total_wait += veh.wait
            self.queue += 1
            self.max_wait = max(self.max_wait, veh.wait)
        else:
            self.approach += 1
    
    @property
    def vehicles(
        self,
    ) -> List[VehicleMeasures]:
        return self._vehicles.copy()


class FullObservation(_Dict):
    def __init__(self) -> None:

        self._lane_observations: Dict[str, LaneMeasures] = {}
        self._num_vehicles: Set[str] = set()
        self._arrivals: Set[str] = set()
        self._departures: Set[str] = set()
        self._vehicles: Set[str] = set()
        self._last_step_vehicles: Set[str] = set()

    def __getitem__(self, key: str) -> Union[LaneMeasures, set]:
        try:
            return self._lane_observations[key]
        except KeyError:
            return super().__getitem__(key)

    @property
    def lane_observations(
        self,
    ) -> Dict[str, LaneMeasures]:
        return self._lane_observations

    @property
    def arrivals(
        self,
    ) -> Set[str]:
        return self._arrivals

    @property
    def departures(
        self,
    ) -> Set[str]:
        return self._departures

    def update_vehicles(self, all_vehicles: Set[str]) -> None:
        if self._last_step_vehicles:
            self._arrivals = self._last_step_vehicles.difference(all_vehicles)
            self._departures = all_vehicles.difference(self._last_step_vehicles)
        else:
            self._arrivals = all_vehicles
        self._last_step_vehicles = all_vehicles


class Signal:
    def __init__(self, map_name, sumo, id, yellow_length, phases):
        # sourcery skip: raise-specific-error
        self.sumo: traci.Connection = sumo
        self.id: str = id
        self.yellow_time: float = yellow_length
        self.next_phase: int = 0

        # Unique lanes
        self.lanes = []
        self.outbound_lanes = []

        # Group of lanes constituting a direction of traffic
        self.build_config(signal_configs[map_name])

        self.waiting_times: Dict[str, float] = {}

        self.phases, self.yellow_dict = create_yellows(phases, yellow_length)

        # logic = self.sumo.trafficlight.Logic(id, 0, 0, phases=self.phases) # not compatible with libsumo
        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.phases
        self.sumo.trafficlight.setProgramLogic(self.id, logic)

        self.signals: List[Signal] = None  # Used to allow signal sharing
        self.full_observation: FullObservation = FullObservation()

    def build_config(self, myconfig):  
        # sourcery skip: low-code-quality, raise-specific-error
        if self.id not in myconfig:
            raise NotImplementedError(f"{self.id} should be in configuration")

        self.lane_sets = myconfig[self.id]["lane_sets"]
        self.lane_sets_outbound = self.lane_sets.fromkeys(self.lane_sets, [])
        self.downstream = myconfig[self.id]["downstream"]
        self.inbounds_fr_direction = {}

        # inbound lanes
        for direction, lanes in self.lane_sets.items():
            for lane in lanes:
                inbound_to_direction = direction.split("-")[0]
                inbound_fr_direction = REVERSED_DIRECTIONS[inbound_to_direction]
                if inbound_fr_direction in self.inbounds_fr_direction:
                    dir_lanes = self.inbounds_fr_direction[inbound_fr_direction]
                    if lane not in dir_lanes:
                        dir_lanes.append(lane)
                else:
                    self.inbounds_fr_direction[inbound_fr_direction] = [lane]
                if lane not in self.lanes:
                    self.lanes.append(lane)

        # Populate outbound lane information
        self.out_lane_to_signalid = {}
        for direction, dwn_signal in self.downstream.items():
            if dwn_signal is not None:  # A downstream intersection exists
                dwn_lane_sets = myconfig[dwn_signal][
                    "lane_sets"
                ]  # Get downstream signal's lanes
                for key in dwn_lane_sets:  # Find all inbound lanes from upstream
                    if key.split("-")[0] == direction:  # Downstream direction matches
                        dwn_lane_set = dwn_lane_sets[key]
                        if dwn_lane_set is None:
                            raise Exception("Invalid signal config")
                        for lane in dwn_lane_set:
                            if lane not in self.outbound_lanes:
                                self.outbound_lanes.append(lane)
                            self.out_lane_to_signalid[lane] = dwn_signal
                            for selfkey in self.lane_sets:
                                if (
                                    selfkey.split("-")[1] == key.split("-")[0]
                                ):  # Out dir. matches dwnstrm in dir.
                                    self.lane_sets_outbound[selfkey] += dwn_lane_set
        for key in self.lane_sets_outbound:  # Remove duplicates
            self.lane_sets_outbound[key] = list(set(self.lane_sets_outbound[key]))

    # def generate_config(self):
    #     print("GENERATING CONFIG")
    #     # TODO raise Exception('Invalid signal config')
    #     index_to_movement = {
    #         0: "S-W",
    #         1: "S-S",
    #         2: "S-E",
    #         3: "W-N",
    #         4: "W-W",
    #         5: "W-S",
    #         6: "N-E",
    #         7: "N-N",
    #         8: "N-W",
    #         9: "E-S",
    #         10: "E-E",
    #         11: "E-N",
    #     }
    #     self.lane_sets = {movement: [] for movement in index_to_movement.values()}
    #     self.lane_sets_outbound = {}
    #     self.downstream = {"N": None, "E": None, "S": None, "W": None}

    #     links = self.sumo.trafficlight.getControlledLinks(self.id)
    #     # print(self.id, links)
    #     for i, link in enumerate(links):
    #         link = link[0]  # unpack so link[0] is inbound, link[1] outbound
    #         if link[0] not in self.lanes:
    #             self.lanes.append(link[0])
    #         # Group of lanes constituting a direction of traffic
    #         if i % 3 == 0:
    #             index = int(i / 3)
    #             self.lane_sets[index_to_movement[index]].append(link[0])
    #     # print(self.id, self.lane_sets)
    #     """split = self.lane_sets['S-W'][0].split('_')[0]
    #     if 'np' not in split: self.downstream['N'] = split
    #     split = self.lane_sets['W-N'][0].split('_')[0]
    #     if 'np' not in split: self.downstream['E'] = split
    #     split = self.lane_sets['N-E'][0].split('_')[0]
    #     if 'np' not in split: self.downstream['S'] = split
    #     split = self.lane_sets['E-S'][0].split('_')[0]
    #     if 'np' not in split: self.downstream['W'] = split"""
    #     lane = self.lane_sets["S-S"][0]
    #     fr_sig = re.findall("[a-zA-Z]+[0-9]+", lane)[0]
    #     fringes, isfringe = ["top", "right", "left", "bottom"], False
    #     for fringe in fringes:
    #         if fringe in fr_sig:
    #             isfringe = True
    #     if not isfringe:
    #         self.downstream["N"] = fr_sig

    #     lane = self.lane_sets["N-N"][0]
    #     fr_sig = re.findall("[a-zA-Z]+[0-9]+", lane)[0]
    #     fringes, isfringe = ["top", "right", "left", "bottom"], False
    #     for fringe in fringes:
    #         if fringe in fr_sig:
    #             isfringe = True
    #     if not isfringe:
    #         self.downstream["S"] = fr_sig

    #     lane = self.lane_sets["W-W"][0]
    #     fr_sig = re.findall("[a-zA-Z]+[0-9]+", lane)[0]
    #     fringes, isfringe = ["top", "right", "left", "bottom"], False
    #     for fringe in fringes:
    #         if fringe in fr_sig:
    #             isfringe = True
    #     if not isfringe:
    #         self.downstream["E"] = fr_sig

    #     lane = self.lane_sets["E-E"][0]
    #     fr_sig = re.findall("[a-zA-Z]+[0-9]+", lane)[0]
    #     fringes, isfringe = ["top", "right", "left", "bottom"], False
    #     for fringe in fringes:
    #         if fringe in fr_sig:
    #             isfringe = True
    #     if not isfringe:
    #         self.downstream["W"] = fr_sig
    #     print("'" + self.id + "'" + ": {")
    #     print("'lane_sets':" + str(self.lane_sets) + ",")
    #     print("'downstream':" + str(self.downstream) + "},")

    # print(self.id)
    # print(self.sumo.trafficlight.getControlledLinks(self.id))
    # print(self.lanes)
    # print(self.outbound_lanes)
    # print(self.lane_sets_outbound)

    @property
    def phase(self):
        return self.sumo.trafficlight.getPhase(self.id)

    def prep_phase(self, new_phase):
        if self.phase == new_phase:
            self.next_phase = self.phase
        else:
            self.next_phase = new_phase
            key = f"{str(self.phase)}_{str(new_phase)}"
            if key in self.yellow_dict:
                yel_idx = self.yellow_dict[key]
                self.sumo.trafficlight.setPhase(self.id, yel_idx)  # turns yellow

    def set_phase(self):
        self.sumo.trafficlight.setPhase(self.id, int(self.next_phase))

    def observe(self, step_length, distance):
        # full_observation = FullObservation()
        all_vehicles = set()
        for lane in self.lanes:
            lane_measures = LaneMeasures(
                0,
                0,
                0,
                0,
            )
            for vehicle in self.get_vehicles(lane, distance):
                all_vehicles.add(vehicle)
                # Update waiting time
                if vehicle in self.waiting_times:
                    self.waiting_times[vehicle] += step_length
                elif (
                    self.sumo.vehicle.getWaitingTime(vehicle) > 0
                ):  # Vehicle stopped here, add it
                    self.waiting_times[vehicle] = self.sumo.vehicle.getWaitingTime(
                        vehicle
                    )
                veh = VehicleMeasures(
                    id=vehicle,
                    wait=(
                        self.waiting_times[vehicle]
                        if vehicle in self.waiting_times
                        else 0
                    ),
                    speed=self.sumo.vehicle.getSpeed(vehicle),
                    acceleration=self.sumo.vehicle.getAcceleration(vehicle),
                    position=self.sumo.vehicle.getLanePosition(vehicle),
                    type=self.sumo.vehicle.getTypeID(vehicle),
                )

                lane_measures.add_vehicle(veh)

            self.full_observation.lane_observations[lane] = lane_measures

        self.full_observation.update_vehicles(all_vehicles)

        # Clear departures from waiting times
        for vehicle in self.full_observation.departures:
            if vehicle in self.waiting_times:
                self.waiting_times.pop(vehicle)

    # Remove undetectable vehicles from lane
    def get_vehicles(self, lane, max_distance):
        detectable = []
        for vehicle in self.sumo.lane.getLastStepVehicleIDs(lane):
            path = self.sumo.vehicle.getNextTLS(vehicle)
            if len(path) > 0:
                next_light = path[0]
                distance = next_light[2]
                if distance <= max_distance:  # Detectors have a max range
                    detectable.append(vehicle)
        return detectable
