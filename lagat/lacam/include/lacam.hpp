/*
 * Implementation of LaCAM*
 *
 * references:
 * LaCAM: Search-Based Algorithm for Quick Multi-Agent Pathfinding.
 * Keisuke Okumura.
 * Proc. AAAI Conf. on Artificial Intelligence (AAAI). 2023.
 *
 * Improving LaCAM for Scalable Eventually Optimal Multi-Agent Pathfinding.
 * Keisuke Okumura.
 * Proc. Int. Joint Conf. on Artificial Intelligence (IJCAI). 2023.
 *
 * Engineering LaCAM*: Towards Real-Time, Large-Scale, and Near-Optimal
 * Multi-Agent Pathfinding. Keisuke Okumura. Proc. Int. Conf. on Autonomous
 * Agents and Multiagent Systems. 2024.
 */
#pragma once

#include "dist_table.hpp"
#include "graph.hpp"
#include "instance.hpp"
#include "pibt.hpp"
#include "utils.hpp"

// low-level search node
struct LNode {
  std::vector<int> who;
  Vertices where;
  const int depth;
  LNode();
  LNode(LNode *parent, int i, Vertex *v);  // who and where
  ~LNode();
};

struct HNode;
struct CompareHNodePointers {  // for determinism
  bool operator()(const HNode *lhs, const HNode *rhs) const;
};

// high-level search node
struct HNode {
  const Config Q;
  HNode *parent;
  std::set<HNode *, CompareHNodePointers> neighbors;
  int depth;

  // cost
  int g;
  int h;
  int f;

  static bool TURN_OFF_ADJ;

  std::vector<float> priorities;
  std::vector<int> order;
  std::queue<LNode *> search_tree;
  std::set<int> default_policy_agents;
  std::vector<std::vector<int>> adj_agents;

  HNode(Config _C, DistTable *D, const Graph *G, std::vector<int> &occ,
        HNode *_parent = nullptr, int _g = 0, int _h = 0);
  ~HNode();
  void reset_constraints();
};
using HNodes = std::vector<HNode *>;

struct LaCAM {
  const Instance *ins;
  DistTable *D;
  const Deadline *deadline;
  const int seed;
  std::mt19937 MT;
  std::uniform_real_distribution<float> rrd;  // random, real distribution
  const int verbose;

  // solver utils
  PIBT pibt;
  HNode *H_goal;
  std::deque<HNode *> OPEN;
  int loop_cnt;

  // Hyperparameterss
  static bool STAR;
  static bool DEADLOCK_DETECTION;
  static int DEADLOCK_DEPTH;
  static bool PIBT_ONLY;
  static bool DISABLE_MODEL_DURING_STAR;
  static bool ENABLE_MAX_TIMESTEPS;
  static int MAX_TIMESTEPS;

  // known to be effective in the refinement
  static float RANDOM_INSERT_PROB1;
  static float RANDOM_INSERT_PROB2;

  LaCAM(const Instance *_ins, DistTable *_D, int _verbose = 0,
        const Deadline *_deadline = nullptr, int _seed = 0);
  ~LaCAM();
  Solution solve();
  bool set_new_config(HNode *S, LNode *M, Config &Q_to);

  void rewrite(HNode *H_from, HNode *H_to);
  int get_g_val(HNode *H_parent, const Config &Q_to);
  int get_h_val(const Config &Q);
  int get_edge_cost(const Config &Q1, const Config &Q2);

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
