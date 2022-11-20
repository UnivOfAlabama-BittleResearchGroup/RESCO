from typing import List, Dict



# create a prototype for the reward function
def reward_function(signals: List["Signal"]) -> Dict[str, float]:
    return {signal.id: 0.0 for signal in signals}
