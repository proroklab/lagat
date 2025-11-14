#include "../include/lacam.hpp"

bool HNode::TURN_OFF_ADJ = false;
bool LaCAM::STAR = false;
bool LaCAM::DEADLOCK_DETECTION = true;
int LaCAM::DEADLOCK_DEPTH = 3;
bool LaCAM::PIBT_ONLY = false;
float LaCAM::RANDOM_INSERT_PROB1 = 0.000;  // default 0.001; disable for this project
float LaCAM::RANDOM_INSERT_PROB2 = 0.001;
bool LaCAM::DISABLE_MODEL_DURING_STAR = true;
bool LaCAM::ENABLE_MAX_TIMESTEPS = false;
int LaCAM::MAX_TIMESTEPS = 256;


bool CompareHNodePointers::operator()(const HNode *l, const HNode *r) const
{
  const auto N = l->Q.size();
  for (size_t i = 0; i < N; ++i) {
    if (l->Q[i] != r->Q[i]) return l->Q[i]->id < r->Q[i]->id;
  }
  return false;
}

HNode::HNode(Config _Q, DistTable *D, const Graph *G, std::vector<int> &occ, HNode *_parent, int _g, int _h)
    : Q(_Q),
      parent(_parent),
      neighbors(),
      depth(parent == nullptr ? 0 : parent->depth + 1),
      g(_g),
      h(_h),
      priorities(Q.size()),
      order(Q.size(), 0),
      search_tree(),
      adj_agents(Q.size())
{
  if (parent != nullptr) parent->neighbors.insert(this);
  search_tree.push(new LNode());
  const int N = Q.size();

  // for neighbor agents

  for (auto i = 0; i < N; ++i) {
    // set priorities
    if (parent == nullptr) {
      // initialize
      priorities[i] = (float)D->get(i, Q[i]) / 10000;
    } else {
      // dynamic priorities, akin to PIBT
      if (D->get(i, Q[i]) != 0) {
        priorities[i] = parent->priorities[i] + 1;
      } else {
        priorities[i] = parent->priorities[i] - (int)parent->priorities[i];
      }
    }

    // adj_agents
    if (!TURN_OFF_ADJ) occ[Q[i]->id] = i;
  }

  // set order
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](int i, int j) { return priorities[i] > priorities[j]; });

  // identify adjacent agents
  if (!TURN_OFF_ADJ) {
    for (auto i = 0; i < N; ++i) {
      for (auto u : Q[i]->neighbor) adj_agents[i].push_back(occ[u->id]);
    }

    // reset
    for (auto i = 0; i < N; ++i) occ[Q[i]->id] = -1;
  }
}

void HNode::reset_constraints()
{
  while (!search_tree.empty()) {
    delete search_tree.front();
    search_tree.pop();
  }
  search_tree.push(new LNode());
}

HNode::~HNode()
{
  while (!search_tree.empty()) {
    delete search_tree.front();
    search_tree.pop();
  }
}

LNode::LNode() : who(), where(), depth(0) {}

LNode::LNode(LNode *parent, int i, Vertex *v)
    : who(parent->who), where(parent->where), depth(parent->depth + 1)
{
  who.push_back(i);
  where.push_back(v);
}

LNode::~LNode(){};

LaCAM::LaCAM(const Instance *_ins, DistTable *_D, int _verbose,
             const Deadline *_deadline, int _seed)
    : ins(_ins),
      D(_D),
      deadline(_deadline),
      seed(_seed),
      MT(seed),
      rrd(0, 1),
      verbose(_verbose),
      pibt(ins, D, seed),
      H_goal(nullptr),
      OPEN(),
      loop_cnt(0)
{
}

LaCAM::~LaCAM() {}

Solution LaCAM::solve()
{
  solver_info(1, "LaCAM begins");

  // setup search
  auto OPEN = std::deque<HNode *>();
  auto EXPLORED = std::unordered_map<Config, HNode *, ConfigHasher>();
  HNodes GC_HNodes;

  // insert initial node
  auto occ = std::vector<int>(D->K, -1);
  auto H_init = new HNode(ins->starts, D, ins->G, occ);
  OPEN.push_front(H_init);
  if (!PIBT_ONLY) EXPLORED[H_init->Q] = H_init;
  GC_HNodes.push_back(H_init);

  // search loop
  solver_info(2, "search iteration begins");
  while (!OPEN.empty() && !is_expired(deadline)) {
    if (ENABLE_MAX_TIMESTEPS && loop_cnt >= MAX_TIMESTEPS) {
      break;
    }
    ++loop_cnt;

    // random insert
    if (H_goal != nullptr) {
      auto r = rrd(MT);
      if (r < RANDOM_INSERT_PROB2 / 2) {
        OPEN.push_front(H_init);
      } else if (r < RANDOM_INSERT_PROB2) {
        auto H = OPEN[get_random_int(MT, 0, OPEN.size() - 1)];
        OPEN.push_front(H);
      }
    }

    // do not pop here!
    auto H = OPEN.front();  // high-level node

    // check uppwer bounds
    if (H_goal != nullptr && H->g >= H_goal->g) {
      OPEN.pop_front();
      solver_info(4, "prune, g=", H->g, " >= ", H_goal->g);
      OPEN.push_front(H_init);
      continue;
    }

    // check goal condition
    if (H_goal == nullptr && is_same_config(H->Q, ins->goals)) {
      H_goal = H;
      solver_info(2, "found solution, g=", H->g);
      if (!STAR) break;
      if (pibt.policy.use_model && DISABLE_MODEL_DURING_STAR) {
        pibt.policy.use_model = false;
        solver_info(2, "turn off model inference");
        HNode::TURN_OFF_ADJ = true;
      }
      continue;
    }

    // extract constraints
    if (H->search_tree.empty()) {
      OPEN.pop_front();
      continue;
    }
    auto L = H->search_tree.front();
    H->search_tree.pop();

    // low level search
    if (L->depth < (int)H->Q.size()) {
      const auto i = H->order[L->depth];
      auto &&C = H->Q[i]->actions;
      std::shuffle(C.begin(), C.end(), MT);  // randomize
      for (auto u : C) H->search_tree.push(new LNode(L, i, u));
    }

    // create successors at the high-level search
    auto Q_to = Config(ins->N, nullptr);
    auto res = set_new_config(H, L, Q_to);
    delete L;
    if (!res) continue;

    // check explored list
    auto iter = EXPLORED.find(Q_to);
    if (iter != EXPLORED.end()) {
      // known configuration
      auto H_known = iter->second;
      rewrite(H, H_known);
      if (rrd(MT) >= RANDOM_INSERT_PROB1) {
        OPEN.push_front(H_known);  // usual
      } else {
        solver_info(3, "random restart");
        OPEN.push_front(H_init);  // sometimes
      }
    } else {
      // new one -> insert
      auto H_new = new HNode(Q_to, D, ins->G, occ, H, get_g_val(H, Q_to), get_h_val(Q_to));
      OPEN.push_front(H_new);
      if (!PIBT_ONLY) EXPLORED[H_new->Q] = H_new;
      GC_HNodes.push_back(H_new);

      // deadlock detection
      if (!PIBT_ONLY && DEADLOCK_DETECTION && pibt.policy.use_model) {
        auto H_ans = H->parent;  // ancestor
        for (int d = 0; d < DEADLOCK_DEPTH; ++d) {
          if (H_ans == nullptr) break;
          auto &&B = H_ans->default_policy_agents;
          const auto K = B.size();

          for (size_t i = 0; i < ins->N; ++i) {
            if (H_new->Q[i] != ins->goals[i] && H_new->Q[i] == H_ans->Q[i]) {
              // check surrounding agents
              if (H_new->adj_agents[i] == H_ans->adj_agents[i]) B.insert(i);
            }
          }

          // updated
          if (K != B.size()) {
            H_ans->reset_constraints();
            OPEN.push_front(H_ans);
            solver_info(4, "deadlock detection for ", B.size(), " agents");
          }
          H_ans = H_ans->parent;
        }
      }
    }
  }

  // backtrack
  Solution solution;
  {
    auto H = H_goal;
    while (H != nullptr) {
      solution.push_back(H->Q);
      H = H->parent;
    }
    std::reverse(solution.begin(), solution.end());
  }

  // solution
  if (solution.empty()) {
    if (OPEN.empty()) {
      solver_info(2, "fin. unsolvable instance");
    } else {
      solver_info(2, "fin. reach time limit");
    }
  } else {
    if (OPEN.empty()) {
      solver_info(2, "fin. optimal solution, g=", H_goal->g,
                  ", depth=", H_goal->depth);
    } else {
      solver_info(2, "fin. suboptimal solution, g=", H_goal->g,
                  ", depth=", H_goal->depth);
    }
  }

  // end processing
  for (auto &&H : GC_HNodes) delete H;  // memory management

  return solution;
}

bool LaCAM::set_new_config(HNode *H, LNode *L, Config &Q_to)
{
  for (auto d = 0; d < L->depth; ++d) Q_to[L->who[d]] = L->where[d];
  return pibt.set_new_config(H->Q, Q_to, H->order, H->default_policy_agents);
}

void LaCAM::rewrite(HNode *H_from, HNode *H_to)
{
  if (!STAR) return;

  // update neighbors
  H_from->neighbors.insert(H_to);

  // Dijkstra
  std::queue<HNode *> Q({H_from});  // queue is sufficient
  while (!Q.empty()) {
    auto n_from = Q.front();
    Q.pop();
    for (auto n_to : n_from->neighbors) {
      auto g_val = n_from->g + get_edge_cost(n_from->Q, n_to->Q);
      if (g_val < n_to->g) {
        if (n_to == H_goal) {
          solver_info(2, "cost update: g=", H_goal->g, " -> ", g_val,
                      ", depth=", H_goal->depth, " -> ", n_from->depth + 1);
        }
        n_to->g = g_val;
        n_to->f = n_to->g + n_to->h;
        n_to->parent = n_from;
        n_to->depth = n_from->depth + 1;
        Q.push(n_to);
        if (H_goal != nullptr && n_to->f < H_goal->f) {
          OPEN.push_front(n_to);
          solver_info(4, "reinsert: g=", n_to->g, " < ", H_goal->g);
        }
      }
    }
  }
}

int LaCAM::get_g_val(HNode *H_parent, const Config &Q_to)
{
  return H_parent->g + get_edge_cost(H_parent->Q, Q_to);
}

int LaCAM::get_h_val(const Config &Q)
{
  auto c = 0;
  for (size_t i = 0; i < ins->N; ++i) c += D->get(i, Q[i]);
  return c;
}

int LaCAM::get_edge_cost(const Config &Q1, const Config &Q2)
{
  auto cost = 0;
  for (size_t i = 0; i < ins->N; ++i) {
    if (Q1[i] != ins->goals[i] || Q2[i] != ins->goals[i]) {
      cost += 1;
    }
  }
  return cost;
}
