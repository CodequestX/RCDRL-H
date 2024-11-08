import argparse
import datetime
import logging
import os
import sys
import time

import numpy as np
import torch

import environment
import rl_agents
import runner
import utils.graph_utils as graph_utils

torch.manual_seed(123)
np.random.seed(123)


# Set up logger
logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO)


def set_of_ints(arg):
    return set(map(int, arg.split(",")))


parser = argparse.ArgumentParser(description="INF-RCDRL-H")
parser.add_argument(
    "--budget", type=int, default=5, help="budget to select the source node set"
)
parser.add_argument(
    "--rumor_originators", type=set_of_ints, default=set(), help="rumor originators set"
)  
parser.add_argument(
    "--graph",
    type=str,
    metavar="GRAPH_PATH",
    default="train_data",
    help="path to the graph file",
)
parser.add_argument(
    "--agent",
    type=str,
    metavar="AGENT_CLASS",
    default="Agent",
    help="class to use for the agent. Must be in the 'agent' module.",
)
parser.add_argument("--model", type=str, default="HyperS2V_DQN", help="model name")
parser.add_argument(
    "--model_file", type=str, default="RCDRL_H.ckpt", help="model file name"
)
parser.add_argument("--path_model_files", type=str, default="", help="model file paths")
parser.add_argument(
    "--epoch", type=int, metavar="nepoch", default=2, help="number of epochs"
)
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--bs", type=int, default=8, help="minibatch size for training")
parser.add_argument("--n_step", type=int, default=1, help="n step transitions in RL")
parser.add_argument("--cpu", action="store_true", default=False, help="use CPU")
parser.add_argument(
    "--test", action="store_true", default=False, help="test performance of model"
)
parser.add_argument(
    "--environment_name",
    metavar="ENV_CLASS",
    type=str,
    default="RC",
    help="Class to use for the environment. Must be in the 'environment' module",
)
parser.add_argument(
    "--method",
    default="prob",
    help="method to compute the rumor spread, prob or MC",
    choices=["prob", "MC"],
)


def main():
    # -----Load Arguments ------
    args = parser.parse_args()
    logging.info("Loading graph %s" % args.graph)

    # -----Set Device -------
    device = torch.device(
        "cuda" if not (args.cpu) and torch.cuda.is_available() else "cpu"
    )
    args.device = device

    # -----Load Hypergraph ------
    # read multiple hypergraphs
    if os.path.isdir(args.graph):
        path_graphs = [
            os.path.join(args.graph, file_g)
            for file_g in os.listdir(args.graph)
            if not file_g.startswith(".")
        ]
    else:  # read one hypergraph
        path_graphs = [args.graph]

    graph_lst = [graph_utils.read_hypergraph(path_g) for path_g in path_graphs]

    for i in range(len(path_graphs)):
        graph_lst[i].path_graph = path_graphs[i]

    args.graphs = graph_lst

    args.double_dqn = True

    if not args.test:  # for training of RCDRL
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        if not os.path.exists(time_stamp):
            os.makedirs(time_stamp)

        args.model_file = os.path.join(time_stamp, args.model_file)

    args.T = 3
    args.memory_size = 50000
    args.reg_hidden = 32
    args.embed_dim = 64

    # -----Load Agent ------
    logging.info(f"Loading agent {args.model}")

    if os.path.isdir(args.model_file):
        path_model_files = [
            os.path.join(args.model_file, file_m)
            for file_m in os.listdir(args.model_file)
            if not file_m.startswith(".")
        ]
    else:
        path_model_files = [args.model_file]

        args.path_model_files = path_model_files

    for mf in path_model_files:
        agent = rl_agents.Agent(args, mf)

        # -----Load environment ------
        logging.info("Loading environment %s" % args.environment_name)
        train_env = environment.Environment(
            args.environment_name,
            graph_lst,
            args.budget,
            args.rumor_originators,
            method=args.method,
            use_cache=True,
        ) 
        test_env = environment.Environment(
            args.environment_name,
            graph_lst,
            args.budget,
            args.rumor_originators,
            method=args.method,
            use_cache=True,
        )

        # -----Load runner ------
        print("Running a single instance simulation")
        my_runner = runner.Runner(train_env, test_env, agent, not (args.test))
        if not (args.test):
            my_runner.train(args.epoch, args.model_file, "list_cumul_reward.txt")
        else:
            my_runner.test(mf, num_trials=1)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total time usage: {end_time - start_time:.2f} seconds")
