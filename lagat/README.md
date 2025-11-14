# LaGAT

The hybrid approach is our future.

## Requirements

You need [CMake](https://cmake.org/) (≥v3.16).
The code is written in C++(17).

The neural network inference is based on [libtorch](https://pytorch.org/).
For macos users, it is easy to install:

```sh
brew install libtorch
```

For GPU usage, maybe cmake version is critical. See:
https://github.com/pytorch/pytorch/issues/89051

I recommend using camke ≥v3.20.

## Building
Build the project.

```sh
cmake -B build && make -C build -j4
```

## Usage

```sh
build/main -i assets/random-32-32-10-random-1.scen -m assets/random-32-32-10.map -N 50 -v 3
```

The result will be saved in `build/result.txt`.

You can find details of all parameters with:

```sh
build/main --help
```

### With learned models

```sh
build/main -m assets/dense_warehouse.map -N 20 -v 3 --model assets/model.pt
```

The implementation assumes [TorchScript](https://docs.pytorch.org/docs/stable/jit.html) model.
You need to pre-compile your trained model via Python, e.g.:

```py
scripted_model = torch.jit.trace(model, (x, edge_index))
scripted_model.save("gnn_model.pt")
```

### Anytime Refinement

There are two refinement schemes:
- [Tree rewiring](https://kei18.github.io/lacam2/) (i.e., LaCAM\*):
  Guaranteed to converge optimal solutions for __sum-of-loss__, similar to RRT\*.
  It works well in small congested instances, or, highly suboptimal solutions.
  This is enabled with `--star` flag.
  The model inference is off by default during this operation, unless you set `--enable_model_during_star`.
- [LNS refinement](https://kei18.github.io/mapf-IR/):
  Non-guaranteed method, but it generally improves solutions quickly, trying to optimize __sum-of-costs__ (aka. flowtime).
  This is enabled with `--lns` flag.
  You may get better results by adjusting `--plns_num_refiners` (multi-threading).

### Just testing MAGAT (+PIBT collision shielding)

```sh
build/main -m assets/dense_warehouse.map -N 20 -v 3 --model assets/model.pt --pibt_only --sampling prob --tau 1.0
```


## Visualizer

This repository is compatible with [kei18@mapf-visualizer](https://github.com/kei18/mapf-visualizer).
For example,

```sh
mapf-visualizer assets/random-32-32-10.map build/result.txt
```

## Notes

### simple test

```sh
ctest --test-dir ./build
```

### with CUDNN

```sh
cmake -DCAFFE2_USE_CUDNN=True -B build
```
