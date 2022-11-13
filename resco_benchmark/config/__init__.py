from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")
]


# converting the dicts to their proper classes
from prototypes.map_config import NetworkConfig
from prototypes.signal_config import SignalNetworkConfig
from prototypes.agent_config import AgentConfig

__all__['map_config'] = {key: NetworkConfig.from_dict(obj) for key, obj in __all__['map_config'].items()}
__all__['signal_config'] = {key: SignalNetworkConfig.from_dict(obj) for key, obj in __all__['signal_config'].items()}
__all__['agent_config'] = {key: AgentConfig.from_dict(obj) for key, obj in __all__['agent_config'].items()}
