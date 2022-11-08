# To use this code, make sure you
#
#     import json
#
# and then, to convert JSON from a string, do
#
#     result = network_config_from_dict(json.loads(json_string))

from dataclasses import dataclass
from typing import Any, Dict, List, TypeVar, Callable, Type, cast


T = TypeVar("T")


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_dict(f: Callable[[Any], T], x: Any) -> Dict[str, T]:
    assert isinstance(x, dict)
    return { k: f(v) for (k, v) in x.items() }


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


@dataclass
class Downstream:
    n: str
    e: int
    s: str
    w: str

    @staticmethod
    def from_dict(obj: Any) -> 'Downstream':
        assert isinstance(obj, dict)
        n = from_str(obj.get("N"))
        e = int(from_str(obj.get("E")))
        s = from_str(obj.get("S"))
        w = from_str(obj.get("W"))
        return Downstream(n, e, s, w)

    def to_dict(self) -> dict:
        result: dict = {}
        result["N"] = from_str(self.n)
        result["E"] = from_str(str(self.e))
        result["S"] = from_str(self.s)
        result["W"] = from_str(self.w)
        return result


@dataclass
class TrafficSignal:
    lane_sets: Dict[str, List[str]]
    downstream: Downstream

    @staticmethod
    def from_dict(obj: Any) -> 'TrafficSignal':
        assert isinstance(obj, dict)
        lane_sets = from_dict(lambda x: from_list(from_str, x), obj.get("lane_sets"))
        downstream = Downstream.from_dict(obj.get("downstream"))
        return TrafficSignal(lane_sets, downstream)

    def to_dict(self) -> dict:
        result: dict = {}
        result["lane_sets"] = from_dict(lambda x: from_list(from_str, x), self.lane_sets)
        result["downstream"] = to_class(Downstream, self.downstream)
        return result


@dataclass
class NetworkConfig:
    phase_pairs: List[int]
    valid_acts: Dict[str, Dict[str, int]]
    traffic_signals: Dict[str, TrafficSignal]

    @staticmethod
    def from_dict(obj: Any) -> 'NetworkConfig':
        assert isinstance(obj, dict)
        phase_pairs = from_list(from_int, obj.get("phase_pairs"))
        valid_acts = from_dict(lambda x: from_dict(from_int, x), obj.get("valid_acts"))
        traffic_signals = from_dict(TrafficSignal.from_dict, obj.get("traffic_signals"))
        return NetworkConfig(phase_pairs, valid_acts, traffic_signals)

    def to_dict(self) -> dict:
        result: dict = {}
        result["phase_pairs"] = from_list(from_int, self.phase_pairs)
        result["valid_acts"] = from_dict(lambda x: from_dict(from_int, x), self.valid_acts)
        result["traffic_signals"] = from_dict(lambda x: to_class(TrafficSignal, x), self.traffic_signals)
        return result


def network_config_from_dict(s: Any) -> NetworkConfig:
    return NetworkConfig.from_dict(s)


def network_config_to_dict(x: NetworkConfig) -> Any:
    return to_class(NetworkConfig, x)
