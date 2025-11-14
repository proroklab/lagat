#include "../include/plns.hpp"

int PLNS::NUM_REFINERS = 8;

constexpr auto TIME_WAIT = std::chrono::milliseconds(0);

PLNS::PLNS(const Instance *_ins, DistTable *_D, Solution &_solution,
           const Deadline *_deadline, const int _verbose, const int seed)
    : ins(_ins),
      D(_D),
      deadline(_deadline),
      verbose(_verbose),
      MT(seed),
      solution(_solution),
      iteration(0),
      cost_best(get_sum_of_costs(solution))
{
}

PLNS::~PLNS() {}

Solution PLNS::refine()
{
  if (!LNS::ON) return solution;
  solver_info(1, "plns refine starts, sum-of-costs: ", cost_best);

  // set refiner
  LNS::MAX_LOOP_CNT = 1;

  auto step = [&](auto &&sol, const int s) {
    auto refiner = LNS(ins, D, sol, deadline, s, -1);
    return refiner.refine();
  };

  // initialize
  std::list<std::future<Solution> > pool;
  for (auto k = 0; k < NUM_REFINERS; ++k) {
    pool.emplace_back(
        std::async(std::launch::async, step, solution, ++iteration));
  }

  while (!is_expired(deadline)) {
    pool.remove_if([&](auto &proc) {
      if (is_expired(deadline)) return true;
      if (proc.wait_for(TIME_WAIT) != std::future_status::ready) return false;
      auto solution_new = proc.get();
      auto cost = get_sum_of_costs(solution_new);
      if (cost < cost_best) {
        solver_info(3, "sum-of-costs update: ", cost_best, " -> ", cost);
        solution = solution_new;
        cost_best = cost;
      }
      pool.emplace_back(
          std::async(std::launch::async, step, solution, ++iteration));
      return true;
    });
  }

  solver_info(1, "plns refine ends, sum-of-costs: ", cost_best);
  return solution;
}
