from dataclasses import dataclass, field
from enum import Enum
import queue
from typing import Dict, Iterable, List, Set, Tuple, Union, Any
import traci
import traci.constants as tc
import copy
import re

from resco_benchmark.config.prototypes.signal_config import (
    TrafficSignal,
    SignalNetworkConfig,
)


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
    id: str = ""
    speed: float = 0.0
    wait: float = 0.0
    acceleration: float = 0.0
    position: float = 0.0
    type: str = ""
    other: Dict[str, Any] = None  # this is for storing additional subscriptions

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return self.other[key]


@dataclass
class LaneMeasures(_Dict):
    queue: int = field(default=0)
    approach: int = field(default=0)
    total_wait: int = field(default=0)
    max_wait: int = field(default=0)
    _vehicles: List[VehicleMeasures] = field(default_factory=list)

    def add_vehicle(self, veh: VehicleMeasures) -> None:
        if veh.wait > 0:
            self.total_wait += veh.wait
            self.queue += 1
            self.max_wait = max(self.max_wait, veh.wait)
        else:
            self.approach += 1

        self._vehicles.append(veh)

    @property
    def vehicles(
        self,
    ) -> List[VehicleMeasures]:
        return self._vehicles or []

    @property
    def fuel_consumption(self) -> float:
        return sum(v.fuel_consumption for v in self._vehicles)


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
            self._arrivals = all_vehicles.difference(self._last_step_vehicles)
            self._departures = self._last_step_vehicles.difference(all_vehicles)
        else:
            self._arrivals = all_vehicles
        self._last_step_vehicles = all_vehicles


class LightState(Enum):
    RED = 0
    GREEN = 1
    YELLOW = 2


class _SignalTimer:
    """
    There is no (all) red state in the timer for the time being
    """

    # TODO: build support for all red

    def __init__(self, min_yellow: float, min_green: float) -> None:
        self._current_state: LightState = LightState.RED
        self._start_time: int = 0
        self._min_yellow: float = min_yellow
        self._min_green: float = min_green

    @property
    def state(self) -> LightState:
        return self._current_state

    def reset(self) -> None:
        self._current_state = LightState.RED
        self._start_time = 0

    def okay_to_switch(self, current_time: int) -> bool:
        # if self._current_state == LightState.GREEN:
        #     return True
        if self._current_state == LightState.YELLOW:
            return current_time - self._start_time >= self._min_yellow
        else:
            return current_time - self._start_time >= self._min_green

    def update(self, current_time: int) -> None:
        if self._current_state == LightState.YELLOW:
            self._current_state = LightState.GREEN
        else:
            self._current_state = LightState.YELLOW

        self._start_time = current_time


class Signal:

    SUBSCRIPTIONS = {
        "lane": [tc.LAST_STEP_VEHICLE_ID_LIST],
        "vehicle": [
            tc.VAR_NEXT_TLS,
            tc.VAR_WAITING_TIME,
            tc.VAR_SPEED,
            tc.VAR_ACCELERATION,
            tc.VAR_LANEPOSITION,
            tc.VAR_TYPE,
        ],
    }

    def __init__(
        self,
        id: str,
        yellow_length: float,
        min_green: float,
        phases: Dict[str, List[traci.trafficlight.Phase]],
        reward_subscriptions: List[str],
        state_subscriptions: List[str],
        signal_configs: SignalNetworkConfig,
        libsumo: bool = False,
    ):

        # sourcery skip: raise-specific-error
        self.sumo: traci = None  # this is set later
        self.id: str = id
        self.yellow_time: float = yellow_length
        self.next_phase: int = 0
        self._libsumo: bool = libsumo

        # update the subscription dict
        for subs in (reward_subscriptions, state_subscriptions):
            for type_ in ("lane", "vehicle"):
                if type_ in subs:
                    self.SUBSCRIPTIONS[type_].extend(subs[type_])
                    self.SUBSCRIPTIONS[type_] = list(set(self.SUBSCRIPTIONS[type_]))

        # Unique lanes
        self.lanes = []
        self.outbound_lanes = []

        # store waiting times
        self.waiting_times: Dict[str, float] = {}

        # Group of lanes constituting a direction of traffic
        self.build_config(signal_configs)
        self.phases, self.yellow_dict = create_yellows(phases, yellow_length)

        # logic = self.sumo.trafficlight.Logic(id, 0, 0, phases=self.phases) # not compatible with libsumo
        self.signals: List[Signal] = None  # Used to allow signal sharing
        self.full_observation: FullObservation = FullObservation()

        # build the internal state of the signal
        self._timer = _SignalTimer(min_yellow=yellow_length, min_green=min_green)

        # set the phase equal to the first phase
        self._phase: int = None

    def init_traffic_light(self) -> None:
        if self._libsumo:
            # this is a hack to get the sumolib -> libsumo conversion to work
            self.phases = [
                self.sumo.trafficlight.Phase(
                    p.duration, p.state, p.minDur, p.maxDur, p.next
                )
                for p in self.phases
            ]
        logic = self.sumo.trafficlight.getAllProgramLogics(self.id)[0]
        logic.phases = self.phases
        self.sumo.trafficlight.setProgramLogic(self.id, logic)

    def reintialize(self, sumo: traci) -> None:
        self.sumo = sumo
        self.init_traffic_light()
        self._sub_lane_info()
        self._phase = self.sumo.trafficlight.getPhase(self.id)
        self._timer.reset()
        self.full_observation = FullObservation()
        self.waiting_times = {}
        self.next_phase = -1  # reset so that the first action is observed

    def _sub_lane_info(
        self,
    ) -> None:
        for lane in self.lanes:
            self.sumo.lane.subscribe(lane, self.SUBSCRIPTIONS["lane"])

    def build_config(self, myconfig: SignalNetworkConfig):
        # sourcery skip: low-code-quality, raise-specific-error
        if self.id not in myconfig.traffic_signals:
            raise NotImplementedError(f"{self.id} should be in configuration")

        self.lane_sets = myconfig.traffic_signals[self.id].lane_sets
        self.lane_sets_outbound = self.lane_sets.fromkeys(self.lane_sets, [])
        self.downstream = myconfig.traffic_signals[self.id].downstream
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
                dwn_lane_sets = myconfig.traffic_signals[dwn_signal].lane_sets
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

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value

    def set_phase(self, time_: float, new_phase: int) -> None:
        if not self._timer.okay_to_switch(time_):
            # there could be some kind of a reward penalty here,
            # the controller should learn to not suggest a phase change unless it is time
            return

        # should we allow a next phase override? (or is an action lasting)
        self.next_phase = new_phase

        if self._timer.state == LightState.GREEN:
            if self._phase == new_phase:
                # we are already in the right phase, have to write to the sumo
                return self.sumo.trafficlight.setPhase(self.id, self._phase)

            # a yellow phase is needed
            key = f"{str(self.phase)}_{self.next_phase}"
            # not sure why this gaurd is here, what do we do if we don't have a yellow time?
            if key in self.yellow_dict:
                yel_idx = self.yellow_dict[key]
                self.sumo.trafficlight.setPhase(self.id, yel_idx)  # turns yellow
                self._timer.update(time_)

        else:
            # we have to go to green
            self.sumo.trafficlight.setPhase(self.id, self.next_phase)
            self.phase = self.next_phase
            self._timer.update(time_)

    def observation_dry_run(
        self,
    ) -> None:
        for lane in self.lanes:
            lane_measure = LaneMeasures()
            lane_measure.add_vehicle(VehicleMeasures())
            self.full_observation.lane_observations[lane] = lane_measure

        self.full_observation.update_vehicles({""})

    def observe(
        self,
        distance: float,
        veh_observations: Dict[str, Any],
        lane_observations: Dict[str, Any],
    ) -> None:
        all_vehicles = set()
        for lane in self.lanes:
            lane_measures = LaneMeasures()
            for vehicle_id, vehicle_dict in self.filter_vehicles(
                lane_observations[lane], distance, veh_observations
            ):
                all_vehicles.add(vehicle_id)
                self.waiting_times[vehicle_id] = vehicle_dict.pop(tc.VAR_WAITING_TIME)
                lane_measures.add_vehicle(
                    VehicleMeasures(
                        id=vehicle_id,
                        wait=(
                            self.waiting_times[vehicle_id]
                            if vehicle_id in self.waiting_times
                            else 0
                        ),
                        speed=vehicle_dict.pop(tc.VAR_SPEED),
                        acceleration=vehicle_dict.pop(tc.VAR_ACCELERATION),
                        position=vehicle_dict.pop(tc.VAR_LANEPOSITION),
                        type=vehicle_dict.pop(tc.VAR_TYPE),
                        other=vehicle_dict,
                    )
                )

            self.full_observation.lane_observations[lane] = lane_measures

        self.full_observation.update_vehicles(all_vehicles)
        # Clear departures from waiting times
        for vehicle in self.full_observation.departures:
            if vehicle in self.waiting_times:
                self.waiting_times.pop(vehicle)

    # Remove undetectable vehicles from lane
    def filter_vehicles(
        self, lane_dict: Dict[str, Any], max_distance: float, veh_dict: Dict[str, Any]
    ) -> Iterable[Tuple[str, Dict]]:
        for vehicle in lane_dict[tc.LAST_STEP_VEHICLE_ID_LIST]:
            path = (
                self.sumo.vehicle.getNextTLS(vehicle)
                if self._libsumo
                else veh_dict[vehicle][tc.VAR_NEXT_TLS]
            )

            if len(path) > 0:
                next_light = path[0]
                distance = next_light[2]
                if distance <= max_distance:
                    yield (vehicle, veh_dict[vehicle])
