/*
 * Implementation of refiners
 *
 * references:
 * Iterative Refinement for Real-Time Multi-Robot Path Planning.
 * Keisuke Okumura, Yasumasa Tamura, and Xavier Défago.
 * In Proceedings of IEEE/RSJ International Conference on Intelligent Robots and
 * Systems (IROS). 2021.
 *
 * Anytime multi-agent path finding via large neighborhood search.
 * Jiaoyang Li, Zhe Chen, Daniel Harabor, P Stuckey, and Sven Koenig.
 * In Proceedings of International Joint Conference on Artificial Intelligence
 * (IJCAI). 2021.
 */

#pragma once

#include "collision_table.hpp"
#include "dist_table.hpp"
#include "graph.hpp"
#include "instance.hpp"
#include "metrics.hpp"
#include "sipp.hpp"
#include "translator.hpp"
#include "utils.hpp"

struct LNS {
  const Instance *ins;
  DistTable *D;
  Paths solution_paths;
  Solution solution;
  int cost;
  const Deadline *deadline;
  std::mt19937 MT;
  const int verbose;
  const int N;  // number of agents
  const int V_size;

  std::vector<int> order;
  CollisionTable CT;
  int loop_cnt;

  // Hyperparametes
  static bool ON;
  static int MAX_LOOP_CNT;

  LNS(const Instance *_ins, DistTable *_D, Solution &_solution,
      const Deadline *_deadline, const int seed = 0, const int _verbose = 0);
  ~LNS();
  Solution refine();
  void step();

  // utilities
  template <typename... Body>
  void solver_info(const int level, Body &&...body)
  {
    if (verbose < level) return;
    std::cout << "elapsed:" << std::setw(6) << elapsed_ms(deadline) << "ms"
              << "  loop_cnt:" << std::setw(8) << loop_cnt << "\t";
    _info(level, verbose, (body)...);
  }
};
