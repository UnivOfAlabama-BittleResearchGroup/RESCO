import numpy as np

from resco_benchmark.agents.agent import Agent, IndependentAgent, SharedAgent
from resco_benchmark.states.states import State
from resco_benchmark.config.prototypes import SignalNetworkConfig, TrafficSignal


class MINJUNG(IndependentAgent):
    def __init__(
        self,
        config: SignalNetworkConfig,
        obs_act: dict,
        map_name: str,
        thread_number: int,
    ):

        super().__init__(config, obs_act, map_name, thread_number)

        self.agents = {
            agent_id: MinjugAgent(agent_id, config.traffic_signals[agent_id]) for agent_id in obs_act
        }

    def ga_set_params(self, weights):
        for agent in self.agents.values():
            agent.ga_set_params(weights[: agent.ga_nvars])
            weights = weights[agent.ga_nvars :]

    @property
    def ga_nvars(self):
        return sum(agent.ga_nvars for agent in self.agents.values())


class MinjugAgent(Agent):
    """
    TODO:
    [x] - Implement the act method
        [x] - Get the desired observations
        [x] - Get the valid actions

    Args:
        WaveAgent (_type_): _description_
    """

    def __init__(self, agent_id, signal_config: TrafficSignal):
        super().__init__()
        self._agent_id = agent_id

        self._phase_pairs = signal_config.phase_pairs 

        phases = sorted({p for phase_pair in self._phase_pairs for p in phase_pair[0]})
        self._phase_to_index = {p: i for i, p in enumerate(phases)}

        self._alpha = np.array([1.0 for _ in phases])
        self._beta = np.array([1.0 for _ in phases])
        self._gamma = np.array([1.0 for _ in phases])

    @property
    def ga_nvars(self):
        return self._alpha.shape[0] + self._beta.shape[0] + self._gamma.shape[0]

    def ga_set_params(self, weights: np.ndarray):
        # update th weights
        self._gamma, self._beta, self._gamma = weights.reshape((3, -1))

    def _compute_priority(self, phase_pair, observation):
        """
        Compute the priority of a phase pair given the observation.

        Args:
            phase_pair (tuple): The phase pair to compute the priority for
            observation (State): The observation to compute the priority for

        Returns:
            float: The priority of the phase pair
        """
        alpha = (
            self._alpha[self._phase_to_index[phase_pair[0]]]
            * observation[phase_pair[0]][0]
            + self._alpha[self._phase_to_index[phase_pair[1]]]
            * observation[phase_pair[1]][0]
        )
        beta = (
            self._beta[self._phase_to_index[phase_pair[0]]]
            * observation[phase_pair[0]][1]
            + self._beta[self._phase_to_index[phase_pair[1]]]
            * observation[phase_pair[1]][1]
        )
        gamma = (
            self._gamma[self._phase_to_index[phase_pair[0]]]
            + self._gamma[self._phase_to_index[phase_pair[1]]]
        )
        return alpha + beta + gamma

    def act(self, observations: State, valid_acts=None, reverse_valid=None):
        """
        Handle the act method for the agent. Loop the valid phase pairs and get the

        alpha * veh_speed_factor[p] + beta * 60 * accumulated_wtime[p] + 60 * gamma
        """

        scores = [
            (sumo_idx, self._compute_priority(pp, observations))
            for pp, sumo_idx in self._phase_pairs
        ]
        # append the valid act with the highest score
        return max(scores, key=lambda x: x[1])[0]

    def observe(self, observation, reward, done, info):
        pass
