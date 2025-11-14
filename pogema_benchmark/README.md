We include the README and LICENSE from the POGEMA Benchmark in this directory, whose original code can be found at
[pogema-benchmark](https://github.com/Cognitive-AI-Systems/pogema-benchmark).
We make use of some of the algorithm implementations provided by them, which can be found in the following directories:
```plaintext
.
├── dcc
├── lacam
├── pogema_benchmark
|   ├── LICENSE
|   └── README.md
└── scrimp
```

Below is the rest of the REAME copied from the POGEMA Benchmark repository.

------------------
# POGEMA Benchmark

Welcome to the official repository for the POGEMA Benchmark. This is an umbrella repository that contains links and information about all the tools and algorithms related to the POGEMA Benchmark.

## Repository Overview

This repository is organized into several key areas:

- **Raw Data Directories**: Contains evaluation results for baseline approaches. Details include:
  - [raw_data_LMAPF](https://github.com/Tviskaron/pogema_benchmark/tree/main/raw_data_LMAPF):  LMAPF challenges, including configurations and maps.
  - [raw_data_MAPF](https://github.com/Tviskaron/pogema_benchmark/tree/main/raw_data_MAPF): MAPF challenges, including configurations and maps.
  - Both directories feature YAML configuration files detailing the evaluation settings (number of agents, maps, seeds, episode length) and a `maps.yaml` file listing all the maps used in the evaluations.

## Installation

### POGEMA Environment
Install the POGEMA environment using pip:
```bash
pip3 install pogema
```
For more details, visit the [POGEMA GitHub repository](https://github.com/AIRI-Institute/pogema).

### POGEMA Toolbox
Install additional tools for POGEMA:
```bash
pip3 install pogema-toolbox
```

### Algorithms
Explore integrated algorithms located in the `algorithms` directory:
- Navigate to the directory using:
  ```bash
  cd algorithms
  ```
- Install necessary dependencies:
  ```bash
  pip3 install -r docker/requirements.txt
  ```
- Optionally, build a Docker image to containerize the environment:
  ```bash
  cd docker && sh build.sh
  ```
  MAMBA baseline requires a separate Docker image:
  ```bash
  cd docker/mamba && sh build.sh
  ```
  
The following table contains links to the original repositories of all the integrated approaches:

| Approach  | Link |
|-----------|------|
| DCC       | [https://github.com/ZiyuanMa/DCC](https://github.com/ZiyuanMa/DCC) |
| Follower  | [https://github.com/AIRI-Institute/learn-to-follow](https://github.com/AIRI-Institute/learn-to-follow) |
| LaCAM     | [https://github.com/Kei18/lacam3](https://github.com/Kei18/lacam3) |
| MATS-LP   | [https://github.com/AIRI-Institute/mats-lp](https://github.com/AIRI-Institute/mats-lp) |
| RHCR      | [https://github.com/Jiaoyang-Li/RHCR](https://github.com/Jiaoyang-Li/RHCR) |
| SCRIMP    | [https://github.com/marmotlab/SCRIMP](https://github.com/marmotlab/SCRIMP) |
| MAMBA     | [https://github.com/jbr-ai-labs/mamba](https://github.com/jbr-ai-labs/mamba) |


## Evaluation

Execute the evaluation script:
```bash
python eval.py
```
## Contents at a Glance

```plaintext
.
├── algorithms
│   ├── Multiple algorithms for benchmarking
│   ├── Docker configuration for container setup
│   ├── eval.py for running evaluations
├── raw_data_LMAPF
│   ├── Data categorized by map types: Random, Mazes, Warehouse, etc.
├── raw_data_MAPF
│   ├── Similar categorization with specific map evaluations
└── README.md
```

## Citation
If you use this repository in your research or wish to cite it, please make a reference to our paper: 
```
@misc{skrynnik2024pogema,
      title={POGEMA: A Benchmark Platform for Cooperative Multi-Agent Navigation}, 
      author={Alexey Skrynnik and Anton Andreychuk and Anatolii Borzilov and Alexander Chernyavskiy and Konstantin Yakovlev and Aleksandr Panov},
      year={2024},
      eprint={2407.14931},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.14931}, 
}
```
