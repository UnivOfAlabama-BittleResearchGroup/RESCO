# To use this code, make sure you
#
#     import json
#
# and then, to convert JSON from a string, do
#
#     result = network_config_from_dict(json.loads(json_string))

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple, TypeVar, Callable, Type, cast
from sumolib.net import Phase


T = TypeVar("T")


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_dict(f: Callable[[Any], T], x: Any) -> Dict[str, T]:
    assert isinstance(x, dict)
    return {k: f(v) for (k, v) in x.items()}


def from_list(f: Callable[[Any], T], x: Any, recursive=False) -> List[T]:
    assert isinstance(x, list)
    # try to convert each element
    if recursive:
        return [
            from_list(f, el, recursive) if isinstance(el, list) else f(el) for el in x
        ]
    # make recursive
    else:
        return [f(el) for el in x]


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


@dataclass
class Downstream:
    n: str
    e: str
    s: str
    w: str

    # create a function to iterate
    def __iter__(self) -> Iterable[str]:
        yield from self.items()

    def items(
        self,
    ) -> Iterable[str]:
        return [("N", self.n), ("E", self.e), ("S", self.s), ("W", self.w)]

    @staticmethod
    def from_dict(obj: Any) -> "Downstream":
        assert isinstance(obj, dict)
        n = obj.get("N")
        e = obj.get("E")
        s = obj.get("S")
        w = obj.get("W")
        return Downstream(n, e, s, w)

    def to_dict(self) -> dict:
        result: dict = {"N": from_str(self.n)}
        result["E"] = from_str(str(self.e))
        result["S"] = from_str(self.s)
        result["W"] = from_str(self.w)
        return result


@dataclass
class TrafficSignal:
    lane_sets: Dict[str, List[str]]
    downstream: Downstream
    _phases: List[Phase] = field(default_factory=list)
    _phase_pairs: List[Tuple[int]] = field(default_factory=list)

    @staticmethod
    def from_dict(obj: Any) -> "TrafficSignal":
        assert isinstance(obj, dict)
        lane_sets = from_dict(lambda x: from_list(from_str, x), obj.get("lane_sets"))
        downstream = Downstream.from_dict(obj.get("downstream"))
        return TrafficSignal(lane_sets, downstream)

    def to_dict(self) -> dict:
        result: dict = {
            "lane_sets": from_dict(lambda x: from_list(from_str, x), self.lane_sets)
        }

        result["downstream"] = to_class(Downstream, self.downstream)
        return result

    def set_phases(self, phases: List[str]) -> None:
        self._phases = phases

    def set_phase_pairs(self, valid_acts: Dict, phase_pairs: List[List[int]]) -> None:
        valid_acts = sorted(list(valid_acts.items()), key=lambda x: x[1])
        self._phase_pairs = [(phase_pairs[i], j) for i, j in valid_acts]

    @property
    def phase_pairs(self) -> List[Tuple[Tuple[int], int]]:
        return self._phase_pairs

    @property
    def phases(self) -> List[Phase]:
        return self._phases


@dataclass
class SignalNetworkConfig:
    phase_pairs: List[int]
    valid_acts: Dict[str, Dict[str, int]]
    traffic_signals: Dict[str, TrafficSignal]

    @staticmethod
    def from_dict(obj: Any) -> "SignalNetworkConfig":
        assert isinstance(obj, dict)
        phase_pairs = from_list(from_int, obj.get("phase_pairs"), recursive=True)
        valid_acts = obj.get("valid_acts", {})
        
        if valid_acts:
            valid_acts = from_dict(
                lambda x: from_dict(lambda x: from_int(x), x), valid_acts
            )

        traffic_signals = from_dict(TrafficSignal.from_dict, obj.get("traffic_signals"))
        
        # distribute the valid actions
        for ti, ts in traffic_signals.items():
            ts.set_phase_pairs(
                valid_acts[ti]
                if valid_acts
                else {v: v for v in range(len(phase_pairs))},
                phase_pairs,
            )

        return SignalNetworkConfig(phase_pairs, valid_acts, traffic_signals)

    def to_dict(self) -> dict:
        result: dict = {"phase_pairs": from_list(from_int, self.phase_pairs)}
        result["valid_acts"] = from_dict(
            lambda x: from_dict(from_int, x), self.valid_acts
        )
        result["traffic_signals"] = from_dict(
            lambda x: to_class(TrafficSignal, x), self.traffic_signals
        )
        return result


def network_config_from_dict(s: Any) -> SignalNetworkConfig:
    return SignalNetworkConfig.from_dict(s)


def network_config_to_dict(x: SignalNetworkConfig) -> Any:
    return to_class(SignalNetworkConfig, x)
