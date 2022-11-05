from typing import Dict, List
import functools

import numpy as np

import traci.constants as tc

from resco_benchmark.config.mdp_config import mdp_configs
from resco_benchmark.traffic_signal import Signal
from resco_benchmark.utils.traci_help import add_traci_subcriptions


@add_traci_subcriptions({"vehicle": [tc.VAR_FUELCONSUMPTION]})
def fuel_consumption(signals: List[Signal]):
    rewards = {}
    for signal_id in signals:
        signal = signals[signal_id]
        reward = sum(
            sum(veh[tc.VAR_FUELCONSUMPTION] for veh in signal.full_observation[lane].vehicles)
            for lane in signal.lanes
        )
        rewards[signal_id] = -reward
    return rewards


# the total wait is a default traci subscription
@add_traci_subcriptions({})
def wait(signals: Dict[str, Signal]):
    rewards = {}
    for signal_id in signals:
        total_wait = sum(
            signals[signal_id].full_observation[lane]["total_wait"]
            for lane in signals[signal_id].lanes
        )
        rewards[signal_id] = -total_wait
    return rewards


@add_traci_subcriptions()
def wait_norm(signals: Dict[str, Signal]):
    rewards = {}
    for signal_id in signals:
        total_wait = sum(
            signals[signal_id].full_observation[lane]["total_wait"]
            for lane in signals[signal_id].lanes
        )
        rewards[signal_id] = np.clip(-total_wait / 224, -4, 4).astype(np.float32)
    return rewards


@add_traci_subcriptions()
def pressure(signals: Dict[str, Signal]):
    rewards = {}
    for signal_id in signals:
        queue_length = sum(
            signals[signal_id].full_observation[lane]["queue"]
            for lane in signals[signal_id].lanes
        )
        for lane in signals[signal_id].outbound_lanes:
            dwn_signal = signals[signal_id].out_lane_to_signalid[lane]
            if dwn_signal in signals[signal_id].signals:
                queue_length -= (
                    signals[signal_id]
                    .signals[dwn_signal]
                    .full_observation[lane]["queue"]
                )

        rewards[signal_id] = -queue_length
    return rewards


@add_traci_subcriptions()
def queue_maxwait(signals: Dict[str, Signal]):
    rewards = {}
    for signal_id, signal in signals.items():
        reward = 0
        for lane in signal.lanes:
            reward += signal.full_observation[lane]["queue"]
            reward += (
                signal.full_observation[lane]["max_wait"] * mdp_configs["MA2C"]["coef"]
            )
        rewards[signal_id] = -reward
    return rewards


@add_traci_subcriptions()
def queue_maxwait_neighborhood(signals: Dict[str, Signal]):
    rewards = queue_maxwait(signals)
    neighborhood_rewards = {}
    for signal_id, signal in signals.items():
        sum_reward = rewards[signal_id]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None:
                sum_reward += mdp_configs["MA2C"]["coop_gamma"] * rewards[neighbor]
        neighborhood_rewards[signal_id] = sum_reward

    return neighborhood_rewards


@add_traci_subcriptions()
def fma2c(signals: Dict[str, Signal]):
    fma2c_config = mdp_configs["FMA2C"]
    management = fma2c_config["management"]
    supervisors = fma2c_config["supervisors"]  # reverse of management
    management_neighbors = fma2c_config["management_neighbors"]

    region_fringes = {}
    fringe_arrivals = {}
    liquidity = {}
    for manager in management:
        region_fringes[manager] = []
        fringe_arrivals[manager] = 0
        liquidity[manager] = 0

    for signal_id, signal in signals.items():
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is None or supervisors[neighbor] != supervisors[signal_id]:
                inbounds = signal.inbounds_fr_direction.get(key)
                if inbounds is not None:
                    mgr = supervisors[signal_id]
                    region_fringes[mgr] += inbounds

    for signal_id in signals:
        signal = signals[signal_id]
        manager = supervisors[signal_id]
        fringes = region_fringes[manager]
        arrivals = signal.full_observation["arrivals"]
        liquidity[manager] += len(signal.full_observation["departures"]) - len(
            signal.full_observation["arrivals"]
        )
        for lane in signal.lanes:
            if lane in fringes:
                for vehicle in signal.full_observation[lane]["vehicles"]:
                    if vehicle["id"] in arrivals:
                        fringe_arrivals[manager] += 1

    management_neighborhood = {}
    for manager in management:
        mgr_rew = fringe_arrivals[manager] + liquidity[manager]
        for neighbor in management_neighbors[manager]:
            mgr_rew += fma2c_config["alpha"] * (
                fringe_arrivals[neighbor] + liquidity[neighbor]
            )
        management_neighborhood[manager] = mgr_rew

    rewards = {}
    for signal_id in signals:
        signal = signals[signal_id]
        reward = 0
        for lane in signal.lanes:
            reward += signal.full_observation[lane]["queue"]
            reward += (
                signal.full_observation[lane]["max_wait"] * mdp_configs["FMA2C"]["coef"]
            )
        rewards[signal_id] = -reward

    neighborhood_rewards = {}
    for signal_id in signals:
        signal = signals[signal_id]
        sum_reward = rewards[signal_id]

        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None and supervisors[neighbor] == supervisors[signal_id]:
                sum_reward += fma2c_config["alpha"] * rewards[neighbor]
        neighborhood_rewards[signal_id] = sum_reward

    neighborhood_rewards |= management_neighborhood
    return neighborhood_rewards


@add_traci_subcriptions()
def fma2c_full(signals: Dict[str, Signal]):
    fma2c_config = mdp_configs["FMA2CFull"]
    management = fma2c_config["management"]
    supervisors = fma2c_config["supervisors"]  # reverse of management
    management_neighbors = fma2c_config["management_neighbors"]

    region_fringes = {}
    fringe_arrivals = {}
    liquidity = {}
    for manager in management:
        region_fringes[manager] = []
        fringe_arrivals[manager] = 0
        liquidity[manager] = 0

    for signal_id, signal in signals.items():
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is None or supervisors[neighbor] != supervisors[signal_id]:
                inbounds = signal.inbounds_fr_direction.get(key)
                if inbounds is not None:
                    mgr = supervisors[signal_id]
                    region_fringes[mgr] += inbounds

    for signal_id in signals:
        signal = signals[signal_id]
        manager = supervisors[signal_id]
        fringes = region_fringes[manager]
        arrivals = signal.full_observation["arrivals"]
        liquidity[manager] += len(signal.full_observation["departures"]) - len(
            signal.full_observation["arrivals"]
        )
        for lane in signal.lanes:
            if lane in fringes:
                for vehicle in signal.full_observation[lane]["vehicles"]:
                    if vehicle["id"] in arrivals:
                        fringe_arrivals[manager] += 1

    management_neighborhood = {}
    for manager in management:
        mgr_rew = fringe_arrivals[manager] + liquidity[manager]
        for neighbor in management_neighbors[manager]:
            mgr_rew += fma2c_config["alpha"] * (
                fringe_arrivals[neighbor] + liquidity[neighbor]
            )
        management_neighborhood[manager] = mgr_rew

    rewards = {}
    for signal_id in signals:
        signal = signals[signal_id]
        reward = 0
        for lane in signal.lanes:
            reward += signal.full_observation[lane]["queue"]
            reward += (
                signal.full_observation[lane]["max_wait"]
                * mdp_configs["FMA2CFull"]["coef"]
            )
        rewards[signal_id] = -reward

    neighborhood_rewards = {}
    for signal_id in signals:
        signal = signals[signal_id]
        sum_reward = rewards[signal_id]

        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None and supervisors[neighbor] == supervisors[signal_id]:
                sum_reward += fma2c_config["alpha"] * rewards[neighbor]
        neighborhood_rewards[signal_id] = sum_reward

    neighborhood_rewards |= management_neighborhood
    return neighborhood_rewards
