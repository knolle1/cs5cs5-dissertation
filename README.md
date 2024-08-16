# Dissertation Project: Continual Reinforcement Learning

Parking environment adapted from [HighwayEnv](https://github.com/eleurent/highway-env) (date accessed: 12/05/2024)

PPO adapted from implementation by [Wenlong Wang](https://github.com/realwenlongwang/PPO-Single-File-Notebook-Implementation) (date accessed: 25/06/2024)

## Dependencies

1. Create and activate virtual environment:
	- With conda:
		- `conda create -n thesis python=3.10`
		- `conda activate thesis`
	- Alternatively with venv:
		- `python -m venv .venv`
		- `source .venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`

## Running Experiments

1. Create configuration JSON file
2. Run `python train_eval.py --config PATH_TO_CONFIG`