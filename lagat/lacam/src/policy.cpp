#include "../include/policy.hpp"

#include <torch/nn/functional.h>
#include <torch/torch.h>

using namespace Common;

std::string AgentPolicy::MODEL_FILEPATH = "";
int AgentPolicy::OBSERVATION_RAD = 5;
bool AgentPolicy::USE_COMMUNICATION_RADIUS = false;
int AgentPolicy::COMMUNICATION_RADIUS = 7;
float AgentPolicy::SAMPLING_TEMPERTURE = 1.0;
AgentPolicy::SamplingStrategy AgentPolicy::SAMPLING_STRATEGY =
    SamplingStrategy::Deterministic;
constexpr float PI = 3.14;
torch::jit::script::Module *AgentPolicy::model;

AgentPolicy::AgentPolicy(const Instance* _ins, DistTable* _D, int seed)
    : ins(_ins),
      MT(seed),
      rrd(0, 1),
      N(ins->N),
      V_size(ins->G->size()),
      D(_D),
      occupied_now(V_size, NO_AGENT),
      use_model(std::filesystem::exists(MODEL_FILEPATH)),
      fov_size((OBSERVATION_RAD + 1) * 2 + 1),
      inputs(2),
      device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      global_guide(ins, D, Common::DEADLINE, seed),
      preferences(N, std::vector<std::pair<Vertex*, ActionCost>>(5))
{
  if (use_model) {
  //   auto timer = Deadline();
    torch::jit::getProfilingMode() = false;
  //   torch::manual_seed(seed);
  //   info(1, "use learned model for preference construction, ",
  //        "cuda availability:", torch::cuda::is_available());
  //   model = torch::jit::load(MODEL_FILEPATH);
  //   model.to(device);
  //   model.eval();
  //   info(1, "model loaded");
    set_preferences_learned(ins->starts);
    info(1, "finish first inference");
    MODEL_LOAD_MS = 0;
  }

  // space utilization optimization
  global_guide.construct();
}

AgentPolicy::~AgentPolicy() {}

void AgentPolicy::set_preferences(const Config& Q_from,
                                  const std::set<int>& default_policy_agents)
{
  if (use_model) {
    set_preferences_learned(Q_from);
    if (!default_policy_agents.empty())
      set_preferences_naive(Q_from, default_policy_agents);
  } else {
    set_preferences_naive(Q_from);
  }
}

void AgentPolicy::set_preferences_naive(const Config& Q_from,
                                        const std::set<int>& A)
{
  auto get_cost = [&](const int i, const Vertex* u) {
    auto gg = global_guide.get(i, Q_from[i], u);
    return std::make_tuple(gg, D->get(i, u), rrd(MT));
  };

  auto set = [&](const int i) {
    const auto K = Q_from[i]->neighbor.size();

    // set candidate actions
    for (size_t k = 0; k <= K; ++k) {
      auto u = Q_from[i]->actions[k];
      preferences[i][k] = std::make_pair(u, get_cost(i, u));
    }

    // sort, note: K + 1 is sufficient
    std::sort(
        preferences[i].begin(), preferences[i].begin() + K + 1,
        [&](auto&& a, auto&& b) { return std::get<1>(a) < std::get<1>(b); });
  };

  if (A.empty()) {
    for (int i = 0; i < N; ++i) set(i);
  } else {
    for (auto i : A) set(i);
  }
}

int get_policy_action_index_from_vertex(Vertex* v_from, Vertex* v_to)
{
  if (v_from->x == v_to->x && v_from->y > v_to->y) return 1;  // north
  if (v_from->x == v_to->x && v_from->y < v_to->y) return 2;  // south
  if (v_from->x > v_to->x && v_from->y == v_to->y) return 3;  // west
  if (v_from->x < v_to->x && v_from->y == v_to->y) return 4;  // east
  return 0;                                                   // stay
}

static int inference_cnt = 0;

void AgentPolicy::set_preferences_learned(const Config& Q)
{
  c10::InferenceMode guard(true);

  auto itr = known_config_table.find(Q);
  if (itr == known_config_table.end()) {
    // model inference
    set_features(Q);
    auto&& actions = model->forward(inputs).toTensor();
    ++inference_cnt;
    // if (SAMPLING_STRATEGY == SamplingStrategy::Probablistic) {
    //   static auto opt = torch::nn::functional::GumbelSoftmaxFuncOptions()
    //                         .tau(SAMPLING_TEMPERTURE)
    //                         .hard(false);
    //   actions = torch::nn::functional::gumbel_softmax(actions, opt);
    // }
    actions = actions.to(torch::kCPU);
    itr = std::get<0>(known_config_table.emplace(Q, actions));
  }
  auto&& actions = itr->second;

  auto get_cost = [&](const int i, Vertex* v_from, Vertex* v_to) {
    auto v_idx = get_policy_action_index_from_vertex(v_from, v_to);
    auto acc = actions.accessor<float, 2>();
    auto v_val = -acc[i][v_idx];
    if (SAMPLING_STRATEGY == SamplingStrategy::Probablistic) {
      v_val = v_val / SAMPLING_TEMPERTURE - rng.get();
    }
    auto gg = global_guide.get(i, v_from, v_to);

    if (SAMPLING_STRATEGY == SamplingStrategy::Tiebreaking) {
      /* CS-PIBT paper  */
      return std::make_pair(v_to,
                            std::make_tuple(gg, (float)D->get(i, v_to), v_val));
    } else {
      /* deterministic or probabilistic */
      return std::make_pair(v_to, std::make_tuple(gg, v_val, rrd(MT)));
    }
  };

  // set preference
  for (int i = 0; i < N; ++i) {
    // set candidate actions
    const auto u = Q[i];
    const auto K = u->neighbor.size();
    for (size_t k = 0; k <= K; ++k) {
      preferences[i][k] = get_cost(i, u, u->actions[k]);
    }

    // sort, note: K + 1 is sufficient
    std::sort(
        preferences[i].begin(), preferences[i].begin() + K + 1,
        [&](auto&& a, auto&& b) { return std::get<1>(a) < std::get<1>(b); });
  }
}

void AgentPolicy::set_features(const Config& Q)
{
  c10::InferenceMode guard(true);

  const float d_invalid = 2 * OBSERVATION_RAD;

  // for edge construction
  for (int i = 0; i < N; ++i) occupied_now[Q[i]->id] = i;

  // TODO: memory allocation might be expensive, need to investigate
  static torch::Tensor node_feature =
      torch::zeros({N, 4, fov_size, fov_size}, torch::kFloat32);
  static auto X = node_feature.accessor<float, 4>();

  int edge_ptr = 0;
  std::vector<std::pair<int, int>> vec_edge_index;
  std::vector<std::tuple<int, int, int>> vec_edge_attr;
  std::vector<std::tuple<int, int, int, int>> mean_vec_edge_attr(
      N, std::make_tuple(0, 0, 0, 0));
  auto add_edge = [&](int src, int dst, float x_rel = 0, float y_rel = 0) {
    if (src == dst) return;
    vec_edge_index.emplace_back(src, dst);
    vec_edge_attr.emplace_back(y_rel, x_rel, std::abs(x_rel) + std::abs(y_rel));
    mean_vec_edge_attr[dst] =
        std::make_tuple(std::get<0>(mean_vec_edge_attr[dst]) + y_rel,
                        std::get<1>(mean_vec_edge_attr[dst]) + x_rel,
                        std::get<2>(mean_vec_edge_attr[dst]) + std::abs(x_rel) +
                            std::abs(y_rel),
                        std::get<3>(mean_vec_edge_attr[dst]) + 1);
    ++edge_ptr;
  };

  auto&& W = ins->G->width;
  auto&& H = ins->G->height;

  // feature and edge_index construction
  for (int i = 0; i < N; ++i) {
    const auto v_i = Q[i];
    const auto d_base = D->get(i, v_i);

    // local, global
    for (auto y_l = 0; y_l < fov_size; ++y_l) {
      const auto y_g = y_l + v_i->y - OBSERVATION_RAD - 1;
      for (auto x_l = 0; x_l < fov_size; ++x_l) {
        const auto x_g = x_l + v_i->x - OBSERVATION_RAD - 1;

        // set default value
        X[i][0][y_l][x_l] = 0;     // without obstacle
        X[i][1][y_l][x_l] = 0;     // no agent
        X[i][2][y_l][x_l] = 0;     // no goal
        X[i][3][y_l][x_l] = 1.0f;  // cost-to-go

        if (x_l == 0 || x_l == fov_size - 1 || y_l == 0 ||
            y_l == fov_size - 1) {
          // padding
          X[i][3][y_l][x_l] = 0;
        } else if (0 <= x_g && x_g < W && 0 <= y_g && y_g < H) {
          const auto u = ins->G->U[W * y_g + x_g];
          if (u != nullptr) {
            // no obstacle
            X[i][0][y_l][x_l] = 0;

            // check neighbor agent
            const auto j = occupied_now[u->id];
            if (j != NO_AGENT) {
              const auto pos_diff_x = v_i->x - u->x;
              const auto pos_diff_y = v_i->y - u->y;
              if (!USE_COMMUNICATION_RADIUS ||
                  (pos_diff_x * pos_diff_x + pos_diff_y * pos_diff_y <=
                   COMMUNICATION_RADIUS * COMMUNICATION_RADIUS)) {
                add_edge(i, j, pos_diff_x, pos_diff_y);
              }
              // other agent
              X[i][1][y_l][x_l] = 1;
            }

            // cost-to-go
            X[i][3][y_l][x_l] =
                std::max(std::min((D->get(i, u) - d_base) / d_invalid, 1.0f),
                         -1.0f);
          } else {
            // obstacle
            X[i][0][y_l][x_l] = 1;
          }
        } else if (((x_g == -1 || x_g == W) && -1 <= y_g && y_g <= H) ||
                   ((y_g == -1 || y_g == H) && -1 <= x_g && x_g <= W)) {
          // boundary
          X[i][0][y_l][x_l] = 1;
        }
      }
    }

    if (USE_COMMUNICATION_RADIUS) {
      // Using comm radius to setup edges
      for (auto y_l = 0; y_l < 2 * (COMMUNICATION_RADIUS - OBSERVATION_RAD); ++y_l) {
        auto y_g = y_l + v_i->y - COMMUNICATION_RADIUS - 1;
        if (y_l >= COMMUNICATION_RADIUS - OBSERVATION_RAD) {
          y_g += 2 * OBSERVATION_RAD + 1;
        }
        for (auto x_l = 0;
             x_l < 2 * (COMMUNICATION_RADIUS - OBSERVATION_RAD); ++x_l) {
          auto x_g = x_l + v_i->x - COMMUNICATION_RADIUS - 1;
          if (x_l >= COMMUNICATION_RADIUS - OBSERVATION_RAD) {
            x_g += 2 * OBSERVATION_RAD + 1;
          }

          if (0 <= x_g && x_g < W && 0 <= y_g && y_g < H) {
            const auto u = ins->G->U[W * y_g + x_g];
            if (u != nullptr) {
              const auto j = occupied_now[u->id];
              if (j != NO_AGENT) {
                const auto pos_diff_x = v_i->x - u->x;
                const auto pos_diff_y = v_i->y - u->y;
                if (pos_diff_x * pos_diff_x + pos_diff_y * pos_diff_y <=
                    COMMUNICATION_RADIUS * COMMUNICATION_RADIUS) {
                  add_edge(i, j, pos_diff_x, pos_diff_y);
                }
              }
            }
          }
        }
      }
    }

    // set target
    auto&& r = OBSERVATION_RAD + 1;
    const auto d_x = ins->goals[i]->x - v_i->x;
    const auto d_y = ins->goals[i]->y - v_i->y;
    auto p_x = d_x;
    auto p_y = d_y;
    if (std::abs(d_x) > r || std::abs(d_y) > r) {
      auto angle = atan2(d_y, d_x);
      if ((angle >= PI / 4 && angle <= PI * 3 / 4) ||
          (angle >= -PI * 3 / 4 && angle <= -PI / 4)) {
        p_x = (d_y == 0) ? r * (d_x < 0 ? -1 : 1)
                         : std::round(d_x * ((float)r / d_y));
        p_y = r * (d_y < 0 ? -1 : 1);
      } else {
        p_x = r * (d_x < 0 ? -1 : 1);
        p_y = (d_x == 0) ? r * (d_y < 0 ? -1 : 1)
                         : std::round(d_y * ((float)r / d_x));
      }
    }
    X[i][2][p_y + r][p_x + r] = 1;
  }

  for (int i = 0; i < N; ++i) {
    auto&& edge_attr_tuple_i = mean_vec_edge_attr[i];
    auto num_edges = std::get<3>(edge_attr_tuple_i);
    if (num_edges == 0) {
      continue;
    }
    vec_edge_index.emplace_back(i, i);
    vec_edge_attr.emplace_back(std::get<0>(edge_attr_tuple_i) / num_edges,
                               std::get<1>(edge_attr_tuple_i) / num_edges,
                               std::get<2>(edge_attr_tuple_i) / num_edges);
    ++edge_ptr;
  }

  torch::Tensor edge_index = torch::empty({2, edge_ptr}, torch::kInt64);
  torch::Tensor edge_attr = torch::empty({edge_ptr, 3}, torch::kFloat32);
  auto edge_index_acc = edge_index.accessor<int64_t, 2>();
  auto edge_attr_acc = edge_attr.accessor<float, 2>();

  for (int64_t i = 0; i < edge_ptr; ++i) {
    edge_index_acc[0][i] = vec_edge_index[i].first;
    edge_index_acc[1][i] = vec_edge_index[i].second;
    edge_attr_acc[i][0] = std::get<0>(vec_edge_attr[i]);
    edge_attr_acc[i][1] = std::get<1>(vec_edge_attr[i]);
    edge_attr_acc[i][2] = std::get<2>(vec_edge_attr[i]);
  }

  auto kwargs = torch::Dict<std::string, torch::Tensor>();
  kwargs.insert("edge_index", edge_index.to(device));
  kwargs.insert("edge_attr", edge_attr.to(device));

  // model inference
  inputs[0] = node_feature.to(device);
  inputs[1] = kwargs;

  // cleanup
  for (int i = 0; i < N; ++i) occupied_now[Q[i]->id] = NO_AGENT;
}

Vertex* AgentPolicy::get(const int i, const int k)
{
  return std::get<0>(preferences[i][k]);
}
