# decorator function
from typing import Dict, List


def add_traci_subcriptions(traci_dict: Dict[str, List[int]] = None):
    def decorator(func):
        return func, traci_dict or {}

    return decorator
