from copy import deepcopy
import os
import uuid

from multi_signal import MultiSignal
import argparse
from resco_benchmark.agents.agent import SharedAgent
from resco_benchmark.runners.build import build_agent_n_env


try:
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.optimize import minimize
    from pymoo.core.problem import DaskParallelization
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.core.result import Result

    NO_PYMOO = False
except ImportError:
    NO_PYMOO = True

try:
    from dask.distributed import Client

    NO_DASK = False
except ImportError:
    NO_DASK = True



class GAProblem(ElementwiseProblem):

        def __init__(self, env_args: argparse.Namespace, *args, **kwargs):
            
            self._eval_counter = 0

            # build the env and agent
            agent, env = build_agent_n_env(env_args, self._eval_counter)
            self._env_args = deepcopy(env_args)

            super().__init__(n_var=agent.ga_nvars, n_obj=1, n_ieq_constr=0, xl=0, xu=5, **kwargs)
            self.elementwise_evaluation = True

        def _evaluate(self, x, out, *args, **kwargs):
            # build the env and agent
            agent, env = build_agent_n_env(self._env_args, uuid.uuid4())
            
            # set the agent parameters
            agent.ga_set_params(x)
            
            # run the agent
            obs = env.reset()
            done = False
            while not done:
                act = agent.act(obs)
                obs, rew, done, info = env.step(act)
                agent.observe(obs, rew, done, info)

            out['F'] = env.get_total_reward()
            env.close()



def ga_optimizer(
    args: argparse.Namespace,
    trial: int,
) -> None:

    if NO_PYMOO:
        raise ImportError("pymoo not installed")
    if NO_DASK:
        raise ImportError("dask[distributed] not installed")

    # start dask client
    client = Client(n_workers=args.procs)
    client.restart()

    # initialize the thread pool and create the runner
    runner = DaskParallelization(client)

    # define the problem by passing the starmap interface of the thread pool
    problem = GAProblem(
        args,
        # elementwise_runner=runner
    )

    from pymoo.operators.sampling.lhs import LHS
    from pymoo.operators.mutation.pm import PolynomialMutation

    res: Result = minimize(problem, GA(
        sampling=LHS(),
        mutation=PolynomialMutation(
            prob=0.4
        ),
    ), termination=("n_gen", 1000), seed=42)

    print(res.X)

    # ga_instance.run()
