#pragma once

#include "dist_table.hpp"
#include "instance.hpp"
#include "lns.hpp"
#include "metrics.hpp"
#include "utils.hpp"

struct PLNS {
  const Instance *ins;
  DistTable *D;
  const Deadline *deadline;
  const int verbose;
  std::mt19937 MT;

  Solution solution;
  int iteration;
  int cost_best;

  // hyperparameters
  static int NUM_REFINERS;

  PLNS(const Instance *_ins, DistTable *_D, Solution &_solution,
       const Deadline *_deadline, const int _verbose, const int seed = 0);
  ~PLNS();

  Solution refine();

  // utilities
  template <typename... Body>
  void solver_info(const int level, Body &&...body)
  {
    if (verbose < level) return;
    std::cout << "elapsed:" << std::setw(6) << elapsed_ms(deadline) << "ms"
              << "  iteration:" << std::setw(8) << iteration << "\t";
    _info(level, verbose, (body)...);
  }
};
