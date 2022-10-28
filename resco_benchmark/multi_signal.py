import os
from os import PathLike
from typing import Dict, List, Tuple
import numpy as np
import traci
import sumolib
import gym
from resco_benchmark.traffic_signal import Signal


class MultiSignal(gym.Env):
    def __init__(
        self,
        run_name: str,
        map_name: str,
        net: str,
        state_fn: callable,
        reward_fn: callable,
        route: str = None,
        gui: bool = False,
        end_time: float = 3600,
        step_length: float = 10,
        yellow_length: float = 4,
        step_ratio: int = 1,
        max_distance: float = 200,
        lights: Tuple[str] = (),
        log_dir: PathLike = "/",
        libsumo: bool = False,
        warmup: float = 0,
        gymma: bool = False,
    ):
        self.libsumo = libsumo
        self.gymma = (
            gymma  # gymma expects sequential list of states/rewards instead of dict
        )
        print(map_name, net, state_fn.__name__, reward_fn.__name__)
        self.log_dir = log_dir
        self.net = net
        self.route = route
        self.gui = gui
        self.state_fn = state_fn
        self.reward_fn = reward_fn
        self.max_distance = max_distance
        self.warmup = warmup

        self.end_time = end_time
        self.step_length = step_length
        self.yellow_length = yellow_length
        self.step_ratio = int(step_ratio)
        self.connection_name = (
            f"{run_name}-{map_name}---{state_fn.__name__}-{reward_fn.__name__}"
        )

        self.all_ts_ids = lights
        self.map_name = map_name

        self.signal_ids: List[str] = []
        self.obs_shape: Dict[str, int] = {}
        self.observation_space: List[int] = []
        self.action_space: List[int] = []
        self.signals: Dict[str, Signal] = {}

        self.run = 0
        self.metrics = []
        self.wait_metric = {}

        if self.route is not None:
            sumo_cmd = [
                sumolib.checkBinary("sumo"),
                "-n",
                net,
                "-r",
                f"{self.route}_1.rou.xml",
                "--no-warnings",
                "True",
            ]

        else:
            sumo_cmd = [sumolib.checkBinary("sumo"), "-c", net, "--no-warnings", "True"]

        # Start sumo, so that we can detect the phases
        self.connect_sumo(sumo_cmd)
        # Run some steps in the simulation with default light configurations to detect phases
        self._init_phases()

        # deduce the step length via the initial connection
        self._sumo_step_length: float = self.sumo.simulation.getDeltaT()

        # Pull signal observation shapes
        self._init_agents(self.all_ts_ids)

        self._disconnect_sumo()

        self.connection_name = f"{run_name}-{map_name}-{len(lights)}-{state_fn.__name__}-{reward_fn.__name__}"

        if not os.path.exists(log_dir + self.connection_name):
            os.makedirs(log_dir + self.connection_name)

        self.sumo_cmd = None
        print("Connection ID", self.connection_name)

    def _disconnect_sumo(self):
        if not self.libsumo:
            traci.switch(self.connection_name)
        traci.close()

    def _init_agents(self, signal_ids):
        self.signals = {
            signal_id: Signal(
                self.map_name,
                self.sumo,
                signal_id,
                self.yellow_length,
                self.phases[signal_id],
            )
            for signal_id in signal_ids
        }

        for ts in signal_ids:
            self.signals[
                ts
            ].signals = (
                self.signals
            )  # pass all signals to each signal, in the case of multi-agent
            self.signals[ts].observe(
                self.step_length, self.max_distance
            )  # observe the state of the signal

        observations = self.state_fn(self.signals)
        self.ts_order = []

        for ts in observations:
            if ts in ["top_mgr", "bot_mgr"]:
                continue  # Not a traffic signal
            o_shape = observations[ts].shape
            self.obs_shape[ts] = o_shape
            o_shape = gym.spaces.Box(low=-np.inf, high=np.inf, shape=o_shape)
            self.ts_order.append(ts)
            self.observation_space.append(o_shape)
            self.action_space.append(gym.spaces.Discrete(len(self.phases[ts])))

    def _init_phases(self, ):

        self.signal_ids = self.sumo.trafficlight.getIDList()
        print("lights", len(self.signal_ids), self.signal_ids)

        # this should work on all SUMO versions
        self.phases = {
            lightID: [
                p
                for p in self.sumo.trafficlight.getAllProgramLogics(lightID)[
                    0
                ].getPhases()
                if "y" not in p.state and "g" in p.state.lower()
            ]
            for lightID in self.signal_ids
        }

        self.all_ts_ids = self.all_ts_ids or self.sumo.trafficlight.getIDList()
        self.ts_starter = len(self.all_ts_ids)
        self.n_agents = self.ts_starter

    def connect_sumo(self, sumo_cmd):
        if self.libsumo:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.connection_name)
            self.sumo = traci.getConnection(self.connection_name)

    def step_sim(self):
        # The monaco scenario expects .25s steps instead of 1s, account for that here.
        for _ in range(self.step_ratio):
            self.sumo.simulationStep()
        # self.sumo.simulationStep(self.step_ratio)  # * self._sumo_step_length)

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
            self.sumo_cmd += ["-c", self.net]
        self.sumo_cmd += [
            "--random",
            "--time-to-teleport",
            "-1",
            "--tripinfo-output",
            os.path.join(
                self.log_dir, self.connection_name, f"tripinfo_{self.run}.xml"
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

        for _ in range(self.warmup):
            self.step_sim()

        # 'Start' only signals set for control, rest run fixed controllers
        if self.run % 30 == 0 and self.ts_starter < len(self.all_ts_ids):
            self.ts_starter += 1
        self.signal_ids = [self.all_ts_ids[i] for i in range(self.ts_starter)]
        self._init_agents(self.signal_ids)

        if self.gymma:
            states = self.state_fn(self.signals)
            return [states[ts] for ts in self.ts_order]

        return self.state_fn(self.signals)

    def step(self, act):
        if self.gymma:
            dict_act = {ts: act[i] for i, ts in enumerate(self.ts_order)}
            act = dict_act

        # Send actions to their signals
        for signal in self.signals:
            self.signals[signal].prep_phase(act[signal])

        for _ in range(self.yellow_length):
            self.step_sim()

        for signal in self.signal_ids:
            self.signals[signal].set_phase()
        for _ in range(self.step_length - self.yellow_length):
            self.step_sim()
        for signal in self.signal_ids:
            self.signals[signal].observe(self.step_length, self.max_distance)

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

    def save_metrics(self):
        log = os.path.join(
            self.log_dir,
            self.connection_name + os.sep + "metrics_" + str(self.run) + ".csv",
        )
        print("saving", log)
        with open(log, "w+") as output_file:
            for line in self.metrics:
                csv_line = ""
                for metric in ["step", "reward", "max_queues", "queue_lengths"]:
                    csv_line = csv_line + str(line[metric]) + ", "
                output_file.write(csv_line + "\n")

    def render(self, mode="human"):
        pass

    def close(self):
        if not self.libsumo:
            traci.switch(self.connection_name)
        traci.close()
        self.save_metrics()
