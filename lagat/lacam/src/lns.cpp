#include "../include/lns.hpp"

bool LNS::ON = true;
int LNS::MAX_LOOP_CNT = 100000;

LNS::LNS(const Instance *_ins, DistTable *_D, Solution &_solution,
         const Deadline *_deadline, const int seed, const int _verbose)
    : ins(_ins),
      D(_D),
      solution_paths(translateConfigsToPaths(_solution)),
      cost(get_sum_of_costs_paths(solution_paths)),
      deadline(_deadline),
      MT(std::mt19937(seed)),
      verbose(_verbose),
      N(ins->N),
      V_size(ins->G->size()),
      order(N, 0),
      CT(ins),
      loop_cnt(0)
{
  std::iota(order.begin(), order.end(), 0);
  for (auto i = 0; i < N; ++i) CT.enrollPath(i, solution_paths[i]);
}

LNS::~LNS() {}

Solution LNS::refine()
{
  if (!ON) return translatePathsToConfigs(solution_paths);
  if (solution_paths[0].empty()) return Solution();
  solver_info(0, "lns begins, cost: ", cost);
  while (!is_expired(deadline) && loop_cnt < MAX_LOOP_CNT) step();
  solver_info(0, "lns ends,   cost: ", cost);
  return translatePathsToConfigs(solution_paths);
}

void LNS::step()
{
  ++loop_cnt;

  auto cost_before = cost;
  std::shuffle(order.begin(), order.end(), MT);

  const auto num_refine_agents =
      std::max(1, std::min(get_random_int(MT, 1, 30), int(N / 4)));
  solver_info(5, "size of modif set: ", num_refine_agents);
  for (auto k = 0; (k + 1) * num_refine_agents < N; ++k) {
    auto old_cost = 0;
    auto new_cost = 0;

    // compute old cost
    for (auto _i = 0; _i < num_refine_agents; ++_i) {
      const auto i = order[k * num_refine_agents + _i];
      old_cost += get_path_cost(solution_paths[i]);
      CT.clearPath(i, solution_paths[i]);
    }

    // re-planning
    Paths new_paths(num_refine_agents);
    for (auto _i = 0; _i < num_refine_agents; ++_i) {
      const auto i = order[k * num_refine_agents + _i];
      new_paths[_i] = sipp(i, ins->starts[i], ins->goals[i], D, &CT, deadline,
                           old_cost - new_cost - 1);
      if (new_paths[_i].empty()) break;  // failure
      new_cost += get_path_cost(new_paths[_i]);
      CT.enrollPath(i, new_paths[_i]);
    }

    if (!new_paths[num_refine_agents - 1].empty() && new_cost <= old_cost) {
      // success
      for (auto _i = 0; _i < num_refine_agents; ++_i) {
        const auto i = order[k * num_refine_agents + _i];
        solution_paths[i] = new_paths[_i];
      }
      cost = cost - old_cost + new_cost;
    } else {
      // failure
      for (auto _i = 0; _i < num_refine_agents; ++_i) {
        const auto i = order[k * num_refine_agents + _i];
        if (!new_paths[_i].empty()) CT.clearPath(i, new_paths[_i]);
        CT.enrollPath(i, solution_paths[i]);
      }
    }
  }

  solver_info(cost < cost_before ? 3 : 4, "cost: ", cost_before, " -> ", cost);
}
