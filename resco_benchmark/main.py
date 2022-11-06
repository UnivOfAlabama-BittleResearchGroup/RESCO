import os
import multiprocessing as mp
import argparse

from multi_signal import MultiSignal

from resco_benchmark.agents.agent import SharedAgent
from resco_benchmark.config.agent_config import agent_configs
from resco_benchmark.config.map_config import map_configs
from resco_benchmark.config.mdp_config import mdp_configs
from resco_benchmark.runners.default import mp_training_loop, training_loop
from resco_benchmark.runners.ga import ga_optimizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--agent",
        type=str,
        default="STOCHASTIC",
        choices=[
            "STOCHASTIC",
            "MAXWAVE",
            "MAXPRESSURE",
            "IDQN",
            "IPPO",
            "MPLight",
            "MA2C",
            "FMA2C",
            "MPLightFULL",
            "FMA2CFull",
            "FMA2CVAL",
            "MINJUNG"
        ],
    )
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--eps", type=int, default=100)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument(
        "--map",
        type=str,
        default="ingolstadt1",
        choices=[
            "grid4x4",
            "arterial4x4",
            "ingolstadt1",
            "ingolstadt7",
            "ingolstadt21",
            "cologne1",
            "cologne3",
            "cologne8",
        ],
    )
    ap.add_argument("--pwd", type=str, default=os.path.dirname(__file__))
    ap.add_argument(
        "--log_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.getcwd()), f"results{os.sep}"),
    )

    ap.add_argument("--gui", type=bool, default=False)
    ap.add_argument("--libsumo", type=bool, default=False)
    ap.add_argument(
        "--tr", type=int, default=0
    )  # Can't multi-thread with libsumo, provide a trial number
    ap.add_argument(
        "--ga", type=bool, default=False, 
    )  # Can't multi-thread with libsumo, provide a trial number
    args = ap.parse_args()

    if args.libsumo and "LIBSUMO_AS_TRACI" not in os.environ:
        raise EnvironmentError(
            "Set LIBSUMO_AS_TRACI to nonempty value to enable libsumo"
        )

    if args.procs == 1 or args.libsumo:
        if args.ga:
            ga_optimizer(args, args.tr)
        else:
            training_loop(args, args.tr)
    else:
        mp_training_loop(args, args.tr)






if __name__ == "__main__":
    main()
