#include <argparse/argparse.hpp>
#include <planner.hpp>
#include <torch/torch.h>

#include "policy.hpp"
#include "utils.hpp"


extern "C" {
void *load_model(const char *model_path);
int free_model(void *model);
int run_lacam(int argc, char *argv[], void *model);
}

void *load_model(const char *model_path)
{
    auto use_model = std::filesystem::exists(model_path);
    if (!use_model) {
        info(1, "model file not found: ", model_path);
        return nullptr;
    }

    torch::jit::getProfilingMode() = false;

    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    info(1, "use learned model for preference construction, ",
            "cuda availability:", torch::cuda::is_available());
    torch::jit::script::Module *model = new torch::jit::script::Module;
    (*model) = torch::jit::load(model_path);
    model->to(device);
    model->eval();
    info(1, "model loaded");

    return static_cast<void *>(model);
}

int free_model(void *model)
{
    auto *m = static_cast<torch::jit::script::Module *>(model);
    delete m;
    return 0;
}

int run_lacam(int argc, char *argv[], void *model)
{
  // arguments parser
  auto program = argparse::ArgumentParser("lacam", "0.1.0");
  program.add_argument("-m", "--map").help("map file").required();
  program.add_argument("-i", "--scen").help("scenario file").default_value("");
  program.add_argument("-N", "--num")
      .help("number of agents")
      .scan<'d', int>()
      .required();
  program.add_argument("-s", "--seed")
      .help("seed")
      .scan<'d', int>()
      .default_value(0);
  program.add_argument("-v", "--verbose")
      .help("verbose")
      .scan<'d', int>()
      .default_value(0);
  program.add_argument("-t", "--time_limit_sec")
      .help("time limit sec")
      .scan<'g', float>()
      .default_value(3.0f);
  program.add_argument("-o", "--output")
      .help("output file")
      .default_value("./build/result.txt");
  program.add_argument("-l", "--log_short")
      .default_value(false)
      .implicit_value(true);
  // hyper parameters
  program.add_argument("--model").help("model file (.pt)").default_value("");
  program.add_argument("--sampling")
      .default_value("deterministic")
      .help("deterministic, probabilistic, tiebreak-only");
  program.add_argument("--tau").scan<'g', float>().default_value(1.0f);
  program.add_argument("--no_deadlock_detection", "--no_dd")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--deadlock_depth").scan<'d', int>().default_value(3);
  program.add_argument("--no_pibt_swap")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--gg_margin").scan<'d', int>().default_value(10);
  program.add_argument("--gg", "--global_guidance")
      .help("enable global guidance")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--pibt_only")
      .help("use PIBT only without LaCAM")
      .default_value(false)
      .implicit_value(true);

  program.add_argument("--enable_communication_radius", "--enable_comm_radius")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--communication_radius", "--comm_radius")
      .scan<'d', int>()
      .default_value(7);

  // anytime refinement
  program.add_argument("--star")
      .help("use anytime refinement by tree rewiring")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--enable_model_during_star")
      .help("use model inference during anytime refinement")
      .default_value(false)
      .implicit_value(true);

  // LNS refinement
  program.add_argument("--lns")
      .help("use LNS refinement after LaCAM")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--plns_num_refiners").scan<'d', int>().default_value(4);

  // Max Timesteps
  program.add_argument("--enable_max_timesteps")
      .help("limit number of timesteps/loop counts")
      .default_value(false)
      .implicit_value(true);

  program.add_argument("--max_timesteps")
      .help("timestep/loop count limit")
      .scan<'d', int>()
      .default_value(256);

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  // setup instance
  const auto verbose = program.get<int>("verbose");
  const auto time_limit_sec = program.get<float>("time_limit_sec");
  const auto scen_name = program.get<std::string>("scen");
  const auto seed = program.get<int>("seed");
  const auto map_name = program.get<std::string>("map");
  const auto output_name = program.get<std::string>("output");
  const auto log_short = program.get<bool>("log_short");
  const auto N = program.get<int>("num");
  const auto ins = scen_name.size() > 0 ? Instance(scen_name, map_name, N)
                                        : Instance(map_name, N, seed);
  if (!ins.is_valid(1)) return 1;

  // hyperparameters

  // policy
  AgentPolicy::MODEL_FILEPATH = program.get<std::string>("model");
  const auto sampling_strategy = program.get<std::string>("sampling");
  if (!sampling_strategy.rfind("det", 0)) {
    AgentPolicy::SAMPLING_STRATEGY = AgentPolicy::Deterministic;
  } else if (!sampling_strategy.rfind("prob", 0)) {
    AgentPolicy::SAMPLING_STRATEGY = AgentPolicy::Probablistic;
  } else if (!sampling_strategy.rfind("tie", 0)) {
    AgentPolicy::SAMPLING_STRATEGY = AgentPolicy::Tiebreaking;
  }
  AgentPolicy::SAMPLING_TEMPERTURE = program.get<float>("tau");
  AgentPolicy::USE_COMMUNICATION_RADIUS = 
      program.get<bool>("enable_communication_radius");
  AgentPolicy::COMMUNICATION_RADIUS =
      program.get<int>("communication_radius");

  if (model == nullptr) {
    model = load_model(AgentPolicy::MODEL_FILEPATH.c_str());
  }

  if (model != nullptr) {
    AgentPolicy::model = static_cast<torch::jit::script::Module *>(model);
  } else {
    info(1, "no model loaded, using naive policy");
  }

  // LaCAM
  LaCAM::DEADLOCK_DETECTION = !program.get<bool>("no_deadlock_detection");
  LaCAM::DEADLOCK_DEPTH = program.get<int>("deadlock_depth");
  LaCAM::PIBT_ONLY = program.get<bool>("pibt_only");
  LaCAM::STAR = program.get<bool>("star");
  LaCAM::DISABLE_MODEL_DURING_STAR =
      !program.get<bool>("enable_model_during_star");

  LaCAM::ENABLE_MAX_TIMESTEPS = program.get<bool>("enable_max_timesteps");
  LaCAM::MAX_TIMESTEPS = program.get<int>("max_timesteps");

  // PIBT
  PIBT::SWAP = !program.get<bool>("no_pibt_swap");

  // global guide
  GlobalGuide::ON = program.get<bool>("gg");
  GlobalGuide::COST_MARGIN = program.get<int>("gg_margin");

  // lns, plns
  LNS::ON = program.get<bool>("lns");
  PLNS::NUM_REFINERS = program.get<int>("plns_num_refiners");

  // solve
  auto deadline = Deadline(time_limit_sec * 1000);
  set_print_options(deadline, verbose);
  const auto solution = solve(ins, verbose - 1, &deadline, seed);
  const auto comp_time_ms = deadline.elapsed_ms();

  // failure
  if (solution.empty()) info(1, "failed to solve");

  // check feasibility
  if (!is_feasible_solution(ins, solution)) {
    info(0, "invalid solution");
    return 1;
  }

  // post processing
  print_stats(ins, solution, comp_time_ms);
  make_log(ins, solution, output_name, comp_time_ms, map_name, seed, log_short);
  return 0;
}

int main(int argc, char *argv[])
{
    return run_lacam(argc, argv, nullptr);
}