import argparse
import multiprocessing as mp
from .build import build_agent_n_env


def training_loop(args: argparse.Namespace, trial: int) -> None:
    agent, env = build_agent_n_env(args, trial)
    # Run agent
    for _ in range(args.eps):
        obs = env.reset()
        done = False
        while not done:
            act = agent.act(obs)
            obs, rew, done, info = env.step(act)
            agent.observe(obs, rew, done, info)

    env.close()


def mp_training_loop(
    args: argparse.Namespace,
    trial: int,
) -> None:
    pool = mp.Pool(processes=args.procs)
    for trial in range(1, args.trials + 1):
        pool.apply_async(training_loop, args=(args, trial))
    pool.close()
    pool.join()
