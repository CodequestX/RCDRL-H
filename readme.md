# Implementation of "Rumor Containment in Hypergraph Representation of Social Networks: A Deep Reinforcement Learning based Solution"

## Overview

This project implements a deep reinforcement learning solution for containment of rumor spread in hypergraphs. The model uses a hypergraph variant of structure2vec (HyperS2V) combined with Deep Q-Networks (DQN) to learn optimal node selection strategies for rumor containment.

## Key Components

- **HyperS2V_DQN**: Neural network architecture that combines hypergraph structure embedding with Q-learning
- **Environment**: Simulates rumor spreading dynamics in hypergraphs
- **Runner**: Manages training and testing loops
- **Graph Utils**: Handles hypergraph operations and influence computations

## Requirements

- PyTorch
- PyTorch Geometric
- NumPy
- HALP (Hypergraph Algorithms Library in Python)

## Usage

```bash
python main.py [arguments]
```

### Key Arguments

- `--budget`: Number of nodes to select 
- `--rumor_originators`: Initial set of rumor spreading nodes
- `--graph`: Path to hypergraph file/directory
- `--model`: Model name
- `--epoch`: Number of training epochs 
- `--bs`: Batch size for training
- `--test`: Enable test mode
- `--cpu`: Force CPU usage
- `--method`: Method to compute the rumor spread, prob or MC

## Input Format

Hypergraph files should be CSV formatted with each line representing a hyperedge:
```
node1,node2,...,nodeN,weight
```

## Project Structure

- `main.py`: Entry point and argument parsing
- `models.py`: Neural network architectures
- `rl_agents.py`: DQN agent implementation
- `environment.py`: Rumor spread simulation environment
- `runner.py`: Training/testing loop management
- `utils/graph_utils.py`: Hypergraph operations and utilities