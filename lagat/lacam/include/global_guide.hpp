/*
 * Implementation of SUO
 *
 * references:
 * Optimizingspaceutilizationformoreeffective multi-robot path planning.
 * Shuai D Han and Jingjin Yu.
 * In Proceedings of IEEE International Conference on Robotics and Automation
 * (ICRA). 2022.
 *
 * Engineering LaCAM $^\ast $: Towards Real-Time, Large-Scale, and Near-Optimal
 * Multi-Agent Pathfinding. Okumura, K. Proceedings of International Conference
 * In Proceedings on Autonomous Agents and Multiagent Systems (AAMAS). 2024.
 *
 * Traffic flow optimisation for lifelong multi-agent path finding.
 * Chen, Zhe, et al.
 * In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI).
 * 2024.
 */
#pragma once

#include "collision_table.hpp"
#include "dist_table.hpp"
#include "graph.hpp"
#include "metrics.hpp"
#include "utils.hpp"

struct GlobalGuide {
  const Instance *ins;
  Deadline deadline;
  std::mt19937 MT;
  const int N;
  const int V_size;
  const int T;  // makespan lower bound
  DistTable *D;

  // outcome
  std::vector<Path> paths;

  // hyperparameters
  static bool ON;
  static int COST_MARGIN;

  void construct();
  void _construct();
  void run_suo();

  GlobalGuide(const Instance *_ins, DistTable *_D, const Deadline *_deadline,
              const int seed = 0);
  ~GlobalGuide();

  int get(const int i, const Vertex *v_from, const Vertex *v_to);
};
