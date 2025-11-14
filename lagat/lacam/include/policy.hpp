#pragma once
#include <torch/script.h>

#include "dist_table.hpp"
#include "global_guide.hpp"
#include "graph.hpp"
#include "instance.hpp"
#include "rng.hpp"
#include "utils.hpp"

using ActionCost = std::tuple<int, float, float>;
using Preference = std::vector<std::pair<Vertex *, ActionCost>>;
using Preferences = std::vector<Preference>;

constexpr int NO_AGENT = -1;

struct AgentPolicy {
  enum SamplingStrategy {
    Deterministic,
    Probablistic,
    Tiebreaking,
  };

  const Instance *ins;
  std::mt19937 MT;
  std::uniform_real_distribution<float> rrd;  // random, real distribution

  // solver utils
  const int N;  // number of agents
  const int V_size;
  DistTable *D;
  std::vector<int> occupied_now;  // for quick location check
  bool use_model;

  // RNG
  RandomNumberGenerator rng = RandomNumberGenerator();

  // inference
  const int fov_size;
  std::vector<torch::jit::IValue> inputs;
  torch::Device device;
  std::unordered_map<Config, torch::Tensor, ConfigHasher> known_config_table;

  // guidance
  GlobalGuide global_guide;

  // main
  Preferences preferences;
  static torch::jit::script::Module *model;

  // hyper parameters
  static std::string MODEL_FILEPATH;
  static int OBSERVATION_RAD;
  static bool USE_COMMUNICATION_RADIUS;
  static int COMMUNICATION_RADIUS;
  static SamplingStrategy SAMPLING_STRATEGY;
  static float SAMPLING_TEMPERTURE;

  AgentPolicy(const Instance *_ins, DistTable *_D, int seed = 0);
  ~AgentPolicy();

  void set_preferences(const Config &Q_from, const std::set<int> &A = {});
  void set_preferences_naive(const Config &Q_from, const std::set<int> &A = {});
  void set_preferences_learned(const Config &Q_from);

  // for model inference
  void set_features(const Config &Q);

  ActionCost get_action_cost(const int i, const Vertex *u);
  Vertex *get(const int i, const int k);
};
