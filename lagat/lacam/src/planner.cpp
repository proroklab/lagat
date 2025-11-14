#include "../include/planner.hpp"

Solution solve(const Instance &ins, int verbose, const Deadline *deadline,
               int seed)
{
  // distance table
  auto D = DistTable(ins);
  info(1, "set distance table");

  // lacam
  auto lacam = LaCAM(&ins, &D, verbose, deadline, seed);
  info(1, "start lacam");
  auto solution = lacam.solve();
  if (solution.empty() || !LNS::ON) return solution;

  // lns refinement
  info(1, "use lns");
  auto refiner = PLNS(&ins, &D, solution, deadline, verbose);
  return refiner.refine();
}
