/*
 * implementation of PIBT
 *
 * references:
 * Priority Inheritance with Backtracking for Iterative Multi-agent Path
 * Finding. Keisuke Okumura, Manao Machida, Xavier Défago & Yasumasa Tamura.
 * Artificial Intelligence (AIJ). 2022.
 */
#pragma once
#include "dist_table.hpp"
#include "graph.hpp"
#include "instance.hpp"
#include "policy.hpp"
#include "utils.hpp"

struct PIBT {
  const Instance *ins;
  std::mt19937 MT;
  std::uniform_real_distribution<float> rrd;  // random, real distribution

  // solver utils
  const int N;  // number of agents
  const int V_size;
  DistTable *D;

  // specific to PIBT
  std::vector<int> occupied_now;   // for quick collision checking
  std::vector<int> occupied_next;  // for quick collision checking

  AgentPolicy policy;

  // hyper parameters
  static bool SWAP;

  PIBT(const Instance *_ins, DistTable *_D, int seed = 0);
  ~PIBT();

  bool set_new_config(const Config &Q_from, Config &Q_to,
                      const std::vector<int> &order,
                      const std::set<int> &default_policy_agents);
  bool funcPIBT(const int i, const Config &Q_from, Config &Q_to,
                const std::set<int> &default_policy_agents);
  int is_swap_required_and_possible(const int ai, const Config &Q_from,
                                    Config &Q_to, Vertex *v_i_target);
  bool is_swap_required(const int pusher, const int puller,
                        Vertex *v_pusher_origin, Vertex *v_puller_origin);
  bool is_swap_possible(Vertex *v_pusher_origin, Vertex *v_puller_origin);
};
