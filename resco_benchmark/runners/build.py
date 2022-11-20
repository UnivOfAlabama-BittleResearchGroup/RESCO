import os
from typing import Tuple, Union
import argparse

from resco_benchmark.config.agent_config import agent_configs
from resco_benchmark.config.map_config import map_configs
from resco_benchmark.config.mdp_config import mdp_configs
from resco_benchmark.config.signal_config import signal_configs
from resco_benchmark.multi_signal import MultiSignal
from resco_benchmark.agents.agent import IndependentAgent, SharedAgent
from resco_benchmark.config.prototypes import (
    AgentConfig,
    MapConfig,
    SignalNetworkConfig,
)


def build_configs(
    args: argparse.Namespace,
) -> Tuple[AgentConfig, MapConfig, SignalNetworkConfig, int, list]:
    """
    Build agent, map, and mdp configs

    Args:
        args (argparse.Namespace): Command line arguments

    Raises:
        EnvironmentError: If map config is not found

    Returns:
        Tuple[dict, dict, int, list]: Agent config, map config, number of steps per episode, route
    """

    # Get the MDP Config
    mdp_config = mdp_configs.get(args.agent)
    if mdp_config is not None:
        mdp_map_config = mdp_config.get(args.map)
        if mdp_map_config is not None:
            mdp_config = mdp_map_config
        mdp_configs[args.agent] = mdp_config

    # Setup the Agent
    agt_config = agent_configs[args.agent]

    if mdp_config is not None:
        agt_config["mdp"] = mdp_config
        management = agt_config["mdp"].get("management")
        if management is not None:  # Save some time and precompute the reverse mapping
            supervisors = {}
            for manager in management:
                workers = management[manager]
                for worker in workers:
                    supervisors[worker] = manager
            mdp_config["supervisors"] = supervisors

    map_config = map_configs[args.map]
    route = map_config.route
    if route is not None:
        route = os.path.join(args.pwd, route)
    if args.map in ["grid4x4", "arterial4x4"] and not os.path.exists(route):
        raise EnvironmentError(
            "You must decompress environment files defining traffic flow"
        )

    num_steps_eps = int(
        (map_config.end_time - map_config.start_time) / map_config.step_length
    )
    return agt_config, map_config, signal_configs[args.map], num_steps_eps, route


def build_agent_n_env(
    args: argparse.Namespace, trial: Union[int, str]
) -> Tuple[Union[IndependentAgent, SharedAgent], MultiSignal]:
    """
    Build agent and environment

    Args:
        args (_type_): Command line arguments
        trial (_type_): unique identifier for the trial

    Returns:
        Tuple[SharedAgent, MultiSignal]: Agent and environment
    """

    agt_config, map_config, signal_config, num_steps_eps, route = build_configs(args)
    env = MultiSignal(
        f"{agt_config.agent.__name__}-tr{trial}",
        args.map,
        os.path.join(args.pwd, map_config.net),
        agent_config=agt_config,
        signal_config=signal_config,
        map_config=map_config,
        gui=args.gui,
        log_dir=args.log_dir,
        libsumo=args.libsumo,
    )

    agt_config.episodes = int(args.eps * 0.8)  # schedulers decay over 80% of steps
    agt_config.steps = agt_config.episodes * num_steps_eps
    agt_config.log_dir = os.path.join(args.log_dir, env.connection_name)
    agt_config.num_lights = len(env.all_ts_ids)

    # Create Agent
    agent = agt_config.agent(signal_config, env.gym_shape, args.map, trial)
    return agent, env
