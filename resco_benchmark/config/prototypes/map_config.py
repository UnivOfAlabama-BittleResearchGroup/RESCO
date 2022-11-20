# To use this code, make sure you
#
#     import json
#
# and then, to convert JSON from a string, do
#
#     result = network_config_from_dict(json.loads(json_string))

import contextlib
from typing import Optional, List, Any, TypeVar, Callable, Type, cast


T = TypeVar("T")


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def from_none(x: Any) -> Any:
    assert x is None
    return x


def from_union(fs, x):
    for f in fs:
        with contextlib.suppress(Exception):
            return f(x)
    assert False


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


class MapConfig:
    lights: Optional[List[Any]]
    net: Optional[str]
    route: Optional[str]
    step_length: Optional[int]
    yellow_length: Optional[int]
    step_ratio: Optional[int]
    start_time: Optional[int]
    end_time: Optional[int]
    warmup: Optional[int]

    def __init__(
        self,
        lights: Optional[List[Any]],
        net: Optional[str],
        route: Optional[str],
        step_length: Optional[int],
        yellow_length: Optional[int],
        step_ratio: Optional[int],
        start_time: Optional[int],
        end_time: Optional[int],
        warmup: Optional[int],
    ) -> None:
        self.lights = lights
        self.net = net
        self.route = route
        self.step_length = step_length
        self.yellow_length = yellow_length
        self.step_ratio = step_ratio
        self.start_time = start_time
        self.end_time = end_time
        self.warmup = warmup

    @staticmethod
    def from_dict(obj: Any) -> "MapConfig":
        assert isinstance(obj, dict)
        lights = from_union(
            [lambda x: from_list(lambda x: x, x), from_none], obj.get("lights")
        )
        net = from_union([from_str, from_none], obj.get("net"))
        route = from_union([from_str, from_none], obj.get("route"))
        step_length = from_union([from_int, from_none], obj.get("step_length"))
        yellow_length = from_union([from_int, from_none], obj.get("yellow_length"))
        step_ratio = from_union([from_int, from_none], obj.get("step_ratio"))
        start_time = from_union([from_int, from_none], obj.get("start_time"))
        end_time = from_union([from_int, from_none], obj.get("end_time"))
        warmup = from_union([from_int, from_none], obj.get("warmup"))
        return MapConfig(
            lights,
            net,
            route,
            step_length,
            yellow_length,
            step_ratio,
            start_time,
            end_time,
            warmup,
        )

    def to_dict(self) -> dict:
        result: dict = {
            "lights": from_union(
                [lambda x: from_list(lambda x: x, x), from_none], self.lights
            )
        }

        result["net"] = from_union([from_str, from_none], self.net)
        result["route"] = from_union([from_str, from_none], self.route)
        result["step_length"] = from_union([from_int, from_none], self.step_length)
        result["yellow_length"] = from_union([from_int, from_none], self.yellow_length)
        result["step_ratio"] = from_union([from_int, from_none], self.step_ratio)
        result["start_time"] = from_union([from_int, from_none], self.start_time)
        result["end_time"] = from_union([from_int, from_none], self.end_time)
        result["warmup"] = from_union([from_int, from_none], self.warmup)
        return result

    def get(self, key, default=None):
        return getattr(self, key, default)


def network_config_from_dict(s: Any) -> MapConfig:
    return MapConfig.from_dict(s)


def network_config_to_dict(x: MapConfig) -> Any:
    return to_class(MapConfig, x)
