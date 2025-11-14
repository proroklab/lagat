/*
 * utility functions
 */
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <regex>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils.hpp"

using Time = std::chrono::steady_clock;

// time manager
struct Deadline {
  const Time::time_point t_s;
  const double time_limit_ms;

  Deadline(double _time_limit_ms = 0);
  double elapsed_ms() const;
  double elapsed_ns() const;
};

double elapsed_ms(const Deadline *deadline);
double elapsed_ns(const Deadline *deadline);
double elapsed_ms(const Deadline &deadline);
bool is_expired(const Deadline *deadline);
bool is_expired(const Deadline &deadline);

float get_random_float(std::mt19937 &MT, float from = 0, float to = 1);
float get_random_float(std::mt19937 *MT, float from = 0, float to = 1);
int get_random_int(std::mt19937 &MT, int from = 0, int to = 1);
int get_random_int(std::mt19937 *MT, int from = 0, int to = 1);

template <typename Head, typename... Tail>
void _info(const int level, const int verbose, Head &&head, Tail &&...tail);

void _info(const int level, const int verbose);

template <typename Head, typename... Tail>
void _info(const int level, const int verbose, Head &&head, Tail &&...tail)
{
  if (verbose < level) return;
  std::cout << head;
  _info(level, verbose, std::forward<Tail>(tail)...);
}

std::ostream &operator<<(std::ostream &os, const std::vector<int> &arr);
std::ostream &operator<<(std::ostream &os, const std::list<int> &arr);
std::ostream &operator<<(std::ostream &os, const std::set<int> &arr);

namespace Common
{
  extern Deadline *DEADLINE;
  extern int VERBOSE;
  extern int MODEL_LOAD_MS;

  void set_print_options(Deadline &_deadline, int _verbose = 0);

  // global
  template <typename... Body>
  void info(const int level, Body &&...body)
  {
    if (VERBOSE < level) return;
    std::cout << "elapsed:" << std::setw(6) << elapsed_ms(DEADLINE) << "ms  ";
    _info(level, VERBOSE, (body)...);
  }
};  // namespace Common
