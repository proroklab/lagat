#include "../include/global_guide.hpp"

bool GlobalGuide::ON = true;
int GlobalGuide::COST_MARGIN = 2;

GlobalGuide::GlobalGuide(const Instance *_ins, DistTable *_D,
                         const Deadline *_deadline, const int seed)
    : ins(_ins),
      deadline(_deadline != nullptr ? _deadline->time_limit_ms * 0.5 : 500),
      MT(seed),
      N(ins->N),
      V_size(ins->G->size()),
      T(get_makespan_lower_bound(*ins, *_D) + COST_MARGIN),
      D(_D),
      paths(N)
{
}

GlobalGuide::~GlobalGuide() {}

void GlobalGuide::construct()
{
  if (!ON) return;
  if (paths[0].empty()) {
    Common::info(3, "[global guidance] constructing");
    run_suo();
    Common::info(3, "[global guidance] done");
  }
}

void GlobalGuide::run_suo()
{
  // the implementation follows the lacam3 paper

  // define path finding utilities
  // vertex, cost-to-come, cost-to-go, collision, parent
  using Node = std::tuple<Vertex *, int, int, int, Vertex *>;
  auto cmp = [&](Node &a, Node &b) {
    // collision
    if (std::get<3>(a) != std::get<3>(b))
      return std::get<3>(a) > std::get<3>(b);
    auto f_a = std::get<1>(a) + std::get<2>(a);
    auto f_b = std::get<1>(b) + std::get<2>(b);
    if (f_a != f_b) return f_a > f_b;
    return std::get<0>(a)->id < std::get<0>(b)->id;
  };
  auto CLOSED = std::vector<Vertex *>(V_size, nullptr);  // parent

  // collision data
  CollisionTable CT(ins);

  // metrics
  auto collision_cnt_last = 0;
  auto paths_prev = std::vector<Path>();

  // main loop
  auto order = std::vector<int>(N, 0);
  std::iota(order.begin(), order.end(), 0);
  auto loop = 0;
  while (loop < 2 || CT.collision_cnt < collision_cnt_last) {
    ++loop;
    collision_cnt_last = CT.collision_cnt;

    // randomize planning order
    std::shuffle(order.begin(), order.end(), MT);

    // single-agent path finding for agent-i
    for (int _i = 0; _i < N; ++_i) {
      if (is_expired(deadline)) break;

      const auto i = order[_i];
      const auto cost_ub = D->get(i, ins->starts[i]) + COST_MARGIN;

      // clear cache
      CT.clearPath(i, paths[i]);

      // setup A*
      auto OPEN =
          std::priority_queue<Node, std::vector<Node>, decltype(cmp)>(cmp);
      // used with CLOSED, vertex-id list
      const auto s_i = ins->starts[i];
      OPEN.push(std::make_tuple(s_i, 0, D->get(i, s_i), 0, nullptr));
      auto USED = std::vector<int>();

      // A*
      auto solved = false;
      while (!OPEN.empty() && !is_expired(deadline)) {
        // pop
        auto node = OPEN.top();
        OPEN.pop();

        // check CLOSED list
        const auto v = std::get<0>(node);
        const auto g_v = std::get<1>(node);  // cost-to-come
        const auto c_v = std::get<3>(node);  // collision
        if (CLOSED[v->id] != nullptr) continue;
        CLOSED[v->id] = std::get<4>(node);  // parent
        USED.push_back(v->id);

        // check goal condition
        if (v == ins->goals[i]) {
          solved = true;
          break;
        }

        // expand
        for (auto u : v->neighbor) {
          auto d_u = D->get(i, u);
          if (u != s_i && CLOSED[u->id] == nullptr &&
              d_u + g_v + 1 <= cost_ub) {
            // insert new node
            OPEN.push(std::make_tuple(u, g_v + 1, d_u,
                                      CT.getCollisionCost(v, u, g_v) + c_v, v));
          }
        }
      }

      // backtrack
      if (solved) {
        paths[i].clear();
        auto v = ins->goals[i];
        while (v != nullptr) {
          paths[i].push_back(v);
          v = CLOSED[v->id];
        }
        std::reverse(paths[i].begin(), paths[i].end());
      }

      // register to CT & update collision count
      CT.enrollPath(i, paths[i]);

      // memory management
      for (auto k : USED) CLOSED[k] = nullptr;
    }

    paths_prev = paths;

    if (CT.collision_cnt == 0) break;
    if (is_expired(deadline)) break;
  }

  paths = paths_prev;
}

int GlobalGuide::get(const int i, const Vertex *v_from, const Vertex *v_to)
{
  if (!ON) return 0;
  auto &&path = paths[i];
  if (path.empty() || v_from == path.back()) return 0;

  auto itr = std::find(path.begin(), path.end(), v_from);
  if (itr != path.end() && *(itr + 1) == v_to) return -1;
  return 0;
}
