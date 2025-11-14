/*
 * fast collision checking, used in SUO and refinner
 */
#pragma once

#include "graph.hpp"
#include "instance.hpp"
#include "utils.hpp"

struct CollisionTable {
  // vertex, time, agents
  std::vector<std::vector<std::vector<int>>> body;
  std::vector<std::vector<int>> body_last;
  int collision_cnt;
  const int N;
  const bool no_use_collision_cnt;

  CollisionTable(const Instance *ins, bool _no_use_collision_cnt = false);
  ~CollisionTable();

  int getCollisionCost(const Vertex *v_from, const Vertex *v_to,
                       const int t_from);
  void enrollPath(const int i, Path &path);
  void clearPath(const int i, Path &path);
};
