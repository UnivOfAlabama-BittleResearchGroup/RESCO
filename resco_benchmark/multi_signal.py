import contextlib
import os
from os import PathLike
import re
from typing import Dict, List, Set, Tuple
import numpy as np
import traci
import sumolib
import gym

from resco_benchmark.config.prototypes import SignalNetworkConfig
from resco_benchmark.config.prototypes import MapConfig
from resco_benchmark.config.prototypes import AgentConfig
from resco_benchmark.traffic_signal import Signal


def find_net(file: PathLike) -> Tuple[PathLike, PathLike]:
    """
    Find the net file

    Args:
        file (PathLike): Path to the net file (or config file)
    Returns:
        PathLike: Path to the net file
        PathLike: Path to the config file
    """
    # try to parse it with sumolib. If it fails, it is not a net file
    n = sumolib.net.readNet(file)
    if n.getEdges():
        return file, None
    # if it is a config file, try to find the net file
    with open(file, "r") as f:
        regex = r"<net-file value=\"(\w+.+)\""
        for line in f:
            if match := re.search(regex, line):
                # if the match is relative, make it absolute
                return (
                    (match[1], file)
                    if os.path.isabs(match[1])
                    else (os.path.join(os.path.dirname(file), match[1]), file)
                )

    raise FileNotFoundError("Net file not found")


class MultiSignal(gym.Env):
    def __init__(
        self,
        run_name: str,
        map_name: str,
        net: str,
        agent_config: AgentConfig,
        map_config: MapConfig,
        signal_config: SignalNetworkConfig,
        gui: bool = False,
        log_dir: PathLike = "/",
        libsumo: bool = False,
        gymma: bool = False,
    ):
        self.libsumo = libsumo
        self.gymma = (
            gymma  # gymma expects sequential list of states/rewards instead of dict
        )

        # regex to find <net-file value="cologne3.net.xml"/> in an xml file

        self.net, self.sumo_config = find_net(net)
        self.route = map_config.route
        self.gui = gui

        self.state_fn = agent_config.state
        self.state_subs = agent_config.state_subscriptions
        self.signal_config = signal_config

        self.max_distance = agent_config.max_distance
        self.warmup = map_config.warmup

        # create the reward function and update the subscriptions
        self.reward_fn = agent_config.reward
        self.reward_subs = agent_config.reward_subscriptions

        self.end_time = map_config.end_time
        self.step_length = map_config.step_length

        # TODO: both yellow length and min green should be customizable per signal (and really per phase)
        self.yellow_length = map_config.yellow_length
        self.min_green = map_config.get("min_green", 0)

        # sourcery skip: remove-unnecessary-cast
        self.step_ratio = map_config.get("step_ratio", 1)
        self.connection_name = f"{run_name}-{map_name}---{self.state_fn.__name__}-{self.reward_fn.__name__}"

        self.all_ts_ids = map_config.lights
        self.map_name = map_name

        self.signal_ids: List[str] = []
        self.obs_shape: Dict[str, int] = {}
        self.observation_space: List[int] = []
        self.action_space: List[int] = []
        self.signals: Dict[str, Signal] = {}
        self.vehicle_subscriptions: Set[int] = set()

        self.run = 0
        self.metrics = []
        self.wait_metric = {}

        net = sumolib.net.readNet(self.net, withPrograms=True)
        self._init_phases(net=net)

        # deduce the step length via the initial connection
        self._sumo_step_length: float = 0
        self._sumo_time: float = 0  # store the internal sumo time

        # Pull signal observation shapes
        self._init_agents(
            self.all_ts_ids, self.reward_subs, self.state_subs, self.signal_config, net
        )

        self.connection_name = f"{run_name}-{map_name}-{len(self.signal_ids)}-{self.state_fn.__name__}-{self.reward_fn.__name__}"

        # make the log directory
        self.log_dir = os.path.join(log_dir, self.connection_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.sumo_cmd = None
        print("Connection ID", self.connection_name)

    def _disconnect_sumo(self):
        if not self.libsumo:
            traci.switch(self.connection_name)
        traci.close()

    def _init_agents(
        self,
        signal_ids,
        reward_subs: Dict[str, int],
        state_subs: Dict[str, int],
        signal_config: SignalNetworkConfig,
        net: sumolib.net.Net,
    ):
        # TODO: redo this without opening SUMO the first time!!!
        self.signals = {
            signal_id: Signal(
                id=signal_id,
                yellow_length=self.yellow_length,
                min_green=self.min_green,  # this is a current limitation, need this to be a env parameter
                phases=self.signal_config.traffic_signals[signal_id].phases,
                reward_subscriptions=reward_subs,
                state_subscriptions=state_subs,
                signal_configs=signal_config,
            )
            for signal_id in signal_ids
        }

        # build the sumo subscription dict
        veh_subs = []
        for signal in self.signals.values():
            veh_subs.extend(signal.SUBSCRIPTIONS["vehicle"])
        self.vehicle_subscriptions = set(veh_subs)

        # observe the signal state
        for signal in self.signals.values():
            signal.observation_dry_run()

        observations = self.state_fn(self.signals)

        self.ts_order = []
        for ts in observations:
            if ts in ["top_mgr", "bot_mgr"]:
                continue  # Not a traffic signal
            self.obs_shape[ts] = observations[ts].shape
            self.ts_order.append(ts)
            self.observation_space.append(
                gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape[ts])
            )
            self.action_space.append(gym.spaces.Discrete(len(self.signals[ts].phases)))

    def _init_phases(
        self,
        net: sumolib.net.Net,
    ):

        # try this with sumolib
        self.all_ts_ids = [ts.getID() for ts in net.getTrafficLights()]
        for ts in self.signal_config.traffic_signals:
            tl_progs = net.getTLS(ts).getPrograms()
            for program in tl_progs.values():
                self.signal_config.traffic_signals[ts].set_phases(
                    [
                        phase
                        for phase in program.getPhases()
                        if "y" not in phase.state and "g" in phase.state.lower()
                    ]
                )
                break

        self.ts_starter = len(self.all_ts_ids)
        self.n_agents = self.ts_starter

    def connect_sumo(self, sumo_cmd):
        if self.libsumo:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.connection_name)
            self.sumo = traci.getConnection(self.connection_name)

    def step_sim(self) -> Tuple[Dict, Dict]:
        # The monaco scenario expects .25s steps instead of 1s, account for that here.
        # for _ in range(self.step_ratio):
        self._sumo_time += self._sumo_step_length * self.step_ratio
        self.sumo.simulationStep(self._sumo_time)

        # subscribe to all new vehicles in the network
        for veh in self.sumo.simulation.getDepartedIDList():
            self.sumo.vehicle.subscribe(veh, self.vehicle_subscriptions)

        # return the lane & vehicle observations
        return (
            self.sumo.lane.getAllSubscriptionResults(),
            self.sumo.vehicle.getAllSubscriptionResults(),
        )

    def reset(self):

        if self.run != 0:
            self._disconnect_sumo()
            self.save_metrics()

        self.metrics = []
        self.run += 1

        # Start a new simulation
        self.sumo_cmd = []
        if self.gui:
            self.sumo_cmd.append(sumolib.checkBinary("sumo-gui"))
            self.sumo_cmd.append("--start")
        else:
            self.sumo_cmd.append(sumolib.checkBinary("sumo"))
        if self.route is not None:
            self.sumo_cmd += ["-n", self.net, "-r", f"{self.route}_{self.run}.rou.xml"]
        else:
            self.sumo_cmd += ["-c", self.sumo_config]
        self.sumo_cmd += [
            "--random",
            "--time-to-teleport",
            "-1",
            "--tripinfo-output",
            os.path.join(
                self.log_dir, f"tripinfo_{self.run}.xml"
            ),
            "--tripinfo-output.write-unfinished",
            "--no-step-log",
            "True",
            "--no-warnings",
            "True",
        ]

        if self.libsumo:
            traci.start(self.sumo_cmd)
            self.sumo = traci
        else:
            traci.start(self.sumo_cmd, label=self.connection_name)
            self.sumo = traci.getConnection(self.connection_name)

        # reset the signals
        for signal in self.signals.values():
            signal.reintialize(self.sumo)

        self._sumo_step_length = self.sumo.simulation.getDeltaT()

        # add one to get the initial observations, even though we haven't stepped yet
        for _ in range(self.warmup + 1):
            lane_obs, vehicle_obs = self.step_sim()

        # 'Start' only signals set for control, rest run fixed controllers
        if self.run % 30 == 0 and self.ts_starter < len(self.all_ts_ids):
            self.ts_starter += 1
        self.signal_ids = [self.all_ts_ids[i] for i in range(self.ts_starter)]

        # observe the last step
        for signal in self.signal_ids:
            self.signals[signal].observe(
                self.max_distance,
                veh_observations=vehicle_obs,
                lane_observations=lane_obs,
            )

        if self.gymma:
            states = self.state_fn(self.signals)
            return [states[ts] for ts in self.ts_order]

        # do the sumo call only once
        self._sumo_time = self.sumo.simulation.getTime()

        return self.state_fn(self.signals)

    @property
    def sumo_time(
        self,
    ) -> float:
        return self._sumo_time

    def step(self, act):
        if self.gymma:
            dict_act = {ts: act[i] for i, ts in enumerate(self.ts_order)}
            act = dict_act

        for signal in self.signal_ids:
            self.signals[signal].set_phase(self.sumo_time, act[signal])

        # step the simulation
        lane_obs, vehicle_obs = self.step_sim()

        for signal in self.signal_ids:
            self.signals[signal].observe(
                self.max_distance,
                veh_observations=vehicle_obs,
                lane_observations=lane_obs,
            )

        # observe new state and reward
        observations = self.state_fn(self.signals)
        rewards = self.reward_fn(self.signals)

        self.calc_metrics(rewards)

        done = self.sumo.simulation.getTime() >= self.end_time
        if self.gymma:
            obss, rww = [], []
            for ts in self.ts_order:
                obss.append(observations[ts])
                rww.append(rewards[ts])
            return obss, rww, [done], {"eps": self.run}
        return observations, rewards, done, {"eps": self.run}

    def calc_metrics(self, rewards):
        queue_lengths = {}
        max_queues = {}
        for signal_id in self.signals:
            signal = self.signals[signal_id]
            queue_length, max_queue = 0, 0
            for lane in signal.lanes:
                queue = signal.full_observation[lane]["queue"]
                if queue > max_queue:
                    max_queue = queue
                queue_length += queue
            queue_lengths[signal_id] = queue_length
            max_queues[signal_id] = max_queue
        self.metrics.append(
            {
                "step": self.sumo.simulation.getTime(),
                "reward": rewards,
                "max_queues": max_queues,
                "queue_lengths": queue_lengths,
            }
        )

    def save_metrics(self, additional_files: List[Tuple[str, callable]] = None):
    
        log = os.path.join(self.log_dir, f"metrics_{str(self.run)}.csv")
        print("saving to ", self.log_dir)
        with open(log, "w+") as output_file:
            for line in self.metrics:
                csv_line = ""
                for metric in ["step", "reward", "max_queues", "queue_lengths"]:
                    csv_line = csv_line + str(line[metric]) + ", "
                output_file.write(csv_line + "\n")

        if additional_files is not None:
            for file in additional_files:
                with open(os.path.join(self.log_dir, file[0]), "w") as f:
                    file[1](f)

    def render(self, mode="human"):
        pass

    def close(self, save_metrics=True):
        with contextlib.suppress(traci.TraCIException):
            if not self.libsumo:
                traci.switch(self.connection_name)
            traci.close()
        if save_metrics:
            self.save_metrics()

    def get_total_reward(self):
        return sum(r for metric in self.metrics for r in metric["reward"].values())

    @property
    def gym_shape(self):
        """
        Returns the shape of the observation space for each traffic signal,
        plus the number of phases

        Returns
        -------
        dict
            A dictionary with the shape of the observation space for each traffic signal
        """
        return {
            key: [
                self.obs_shape[key],
                len(self.signals[key].phases) if key in self.signals else None,
            ]
            for key in self.obs_shape
        }
