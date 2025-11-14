#pragma once

#include "dist_table.hpp"
#include "graph.hpp"
#include "instance.hpp"
#include "lacam.hpp"
#include "lns.hpp"
#include "planner.hpp"
#include "plns.hpp"
#include "policy.hpp"
#include "post_processing.hpp"
#include "utils.hpp"

using namespace Common;

Solution solve(const Instance &ins, const int verbose = 0,
               const Deadline *deadline = nullptr, int seed = 0);
