# To use this code, make sure you
#
#     import json
#
# and then, to convert JSON from a string, do
#
#     result = network_config_from_dict(json.loads(json_string))

import contextlib
from typing import Optional, List, Any, TypeVar, Callable, Type, Union, cast
from resco_benchmark.states import state_function
from resco_benchmark.agents.agent import IndependentAgent, SharedAgent
from resco_benchmark.rewards import reward_function


T = TypeVar("T")


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_num(x: Any) -> int:
    assert isinstance(x, (int, float)) and not isinstance(x, bool)
    return x


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()




class AgentConfig:
    
    agent: Union[IndependentAgent, SharedAgent]
    state: state_function
    reward: reward_function
    max_distance: float
       

    def __init__(
        self,
        agent: Union[IndependentAgent, SharedAgent],
        state: state_function,
        reward: reward_function,
        max_distance: float,
        **kwargs
    ) -> None:

        self.agent = agent
        self.state = state
        self.reward = reward
        self.max_distance = max_distance

        # add all the arguments to the class
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, name):
        return self.__dict__.get(name)

    @staticmethod
    def from_dict(obj: Any) -> "AgentConfig":
        assert isinstance(obj, dict)

        # copy the dictionary
        obj = obj.copy()
        agent = obj.pop("agent")
        # make sure the the agent is an instance of the class
        assert isinstance(agent, (IndependentAgent, SharedAgent))

        state = obj.pop("state")
        # make sure the the state is a function
        assert callable(state)

        reward = obj.pop("reward")
        # make sure the the reward is a function
        assert callable(reward)

        max_distance = from_num(obj.pop("max_distance"))

        return AgentConfig(
            agent=agent,
            state=state,
            reward=reward,
            max_distance=max_distance,
            kwargs=obj
        )

    def to_dict(self) -> dict:
        raise NotImplementedError
