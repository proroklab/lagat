"""
This script serves as a wrapper to run the SSIL-based MAPF simulator on custom-defined
maps and scenarios. 

The script is run from the command line and requires arguments to specify the
input files and output directory.

Usage:
    python custom_run_generator.py --input-yaml <path> --bds-file <path> [OPTIONS]

Example:
    python custom_run_generator.py \\
        --input-yaml path/to/your/input.yaml \\
        --bds-file path/to/your/bds_data.npz \\
        --output-dir results/my_custom_run \\
        --shieldType Real-Time-LaCAM \\
        --maxSteps 500 \\
        --useGPU

Arguments:
    --input-yaml (required): Path to the YAML file with map and agent definitions.
    --bds-file (required):   Path to the .npz file with pre-computed BDS heuristics.
    --output-dir (optional): Directory to save generated files and output.csv.
                             Defaults to 'custom_mapf_runs'.
    --shieldType (optional): The collision shield to use.
                             Choices: ['CS-PIBT', 'CS-Freeze', 'LaCAM', 'Real-Time-LaCAM'].
                             Default: 'CS-PIBT'.
    --useGPU (optional):     Flag to enable GPU usage. Default is False.
    --seed (optional):       Random seed for the simulation. Default: 0.
    --maxSteps (optional):   Max steps for simulation (e.g., '1000' or '3x').
                             Default: "1000".
    --lacamLookahead (optional): LaCAM node expansion limit. Required for LaCAM shields.
                                 Default: 0.
    --timeLimit (optional):  Time limit in seconds. Default: 60.
"""
import yaml
import numpy as np
import os
import subprocess
import argparse
import sys
import pandas as pd

MIN_AGENTS_REQUIRED = 6

def create_map_npz(map_data, output_path):
    """Creates a .npz file from map data."""
    dims = map_data['dimensions']
    obstacles = map_data.get('obstacles', [])
    grid = np.zeros(dims, dtype=np.uint8)
    for obs in obstacles:
        grid[obs[0], obs[1]] = 1
    np.savez_compressed(output_path, **{'custom_map.map': grid})
    print(f"Map saved to {output_path}")

def create_scen_file(agents_data, map_name, scen_num, output_dir):
    """Creates a .scen file with a specific format."""
    output_path = os.path.join(output_dir, f"{map_name}-random-{scen_num}.scen")
    with open(output_path, 'w') as f:
        f.write("version 1\n")
        for i, agent in enumerate(agents_data):
            start = agent['start']
            goal = agent['goal']
            f.write(f"{i}\t{map_name}.map\t0\t0\t{start[1]}\t{start[0]}\t{goal[1]}\t{goal[0]}\t0.0\n")
    print(f"Scenario file saved to {output_path}")
    return output_path

def create_yaml_output(paths_file, csv_file, yaml_file, agent_names):
    """Creates the final YAML output from the simulation results."""
    # Load the raw path data
    solution_path = np.load(paths_file) # Shape: (Timesteps, Agents, 2) -> (t, agent, [y, x])
    
    # Load the statistics from the CSV
    stats_df = pd.read_csv(csv_file)
    last_run_stats = stats_df.iloc[-1]
    
    # Build the schedule dictionary
    schedule = {}
    for i in range(solution_path.shape[1]): # Iterate through agents
        agent_name = agent_names[i]
        trajectory = []
        for t in range(solution_path.shape[0]): # Iterate through timesteps
            pos = solution_path[t, i] 
            trajectory.append({'x': int(pos[0]), 'y': int(pos[1]), 't': t})
        schedule[agent_name] = trajectory

    # Build the final output dictionary
    output_data = {
        'statistics': {
            'cost': int(last_run_stats['total_cost_true']),
            'makespan': solution_path.shape[0] - 1,
            'runtime': float(last_run_stats['runtime']),
        },
        'schedule': schedule
    }
    
    # Write to YAML file
    with open(yaml_file, 'w') as f:
        yaml.dump(output_data, f, default_flow_style=None, sort_keys=False)
    print(f"Final trajectory YAML saved to {yaml_file}")


def run_simulation(map_npz_path, map_name, scen_path, bd_npz_path, agent_num, output_csv, sim_args):
    """Constructs and runs the simulation command after performing validation."""
    model_path = "model/ssil_model.pt"
    custom_bd_path = "custom_mapf_runs/custom_bds.npz"

    # --- BDS File Validation ---
    original_bd_data_all = np.load(bd_npz_path)
    original_bd_data = original_bd_data_all[original_bd_data_all.files[0]]
    if original_bd_data.shape[0] < agent_num:
        print(f"Error: The provided BDS file does not have enough pre-computed data.", file=sys.stderr)
        print(f"       Scenario requires data for {agent_num} agents, but the BDS file only contains data for {original_bd_data.shape[0]}.", file=sys.stderr)
        sys.exit(1)

    scen_basename = os.path.basename(scen_path)
    scen_num = scen_basename.split('-')[-1].split('.')[0]
    bd_key = f"{map_name}-random-{scen_num}"
    
    final_bd_data = original_bd_data[:agent_num]
    np.savez_compressed(custom_bd_path, **{bd_key: final_bd_data})
    print(f"Custom BD file created with key: '{bd_key}' using data for {agent_num} agents.")

    command = [
        "python", "-m", "main_pys.simulator",
        "--mapNpzFile", map_npz_path, "--mapName", map_name,
        "--scenFile", scen_path, "--bdNpzFile", custom_bd_path,
        "--modelPath", model_path, "--outputCSVFile", output_csv,
        "--agentNum", str(agent_num), "--maxSteps", sim_args.maxSteps,
        "--shieldType", sim_args.shieldType, "--useGPU", str(sim_args.useGPU),
        "--seed", str(sim_args.seed), "--lacamLookahead", str(sim_args.lacamLookahead),
        "--timeLimit", str(sim_args.timeLimit)
    ]
    
    # Add argument to save the path file
    temp_paths_file = os.path.join(os.path.dirname(output_csv), "temp_paths.npy")
    command.extend(["--outputPathsFile", temp_paths_file])
    
    print("\nExecuting command:")
    print(' '.join(command))
    subprocess.run(command, check=True)
    return temp_paths_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate files and run MAPF simulation for custom inputs.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- File Arguments ---
    parser.add_argument("--input-yaml", required=True, help="Path to the input YAML file describing the map and agents.")
    parser.add_argument("--bds-file", required=True, help="Path to the pre-computed backward dijkstra .npz file.")
    parser.add_argument("--output-dir", default="custom_mapf_runs", help="Directory to save generated files and final output CSV.")
    
    # --- Simulation Arguments ---
    parser.add_argument('--shieldType', type=str, default='CS-PIBT',
                        choices=['CS-PIBT', 'CS-Freeze', 'LaCAM', 'Real-Time-LaCAM'],
                        help="The collision shield to use. (Default: CS-PIBT)")
    parser.add_argument('--useGPU', action='store_true', help="Flag to enable GPU usage. (Default: Disabled)")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for the simulation. (Default: 0)")
    parser.add_argument('--maxSteps', type=str, default="1000",
                        help="Max simulation steps (e.g., '1000' or '3x' for a multiple of the longest path). (Default: 1000)")
    parser.add_argument('--lacamLookahead', type=int, default=0,
                        help="LaCAM node expansion limit. Required for LaCAM shields. (Default: 0)")
    parser.add_argument('--timeLimit', type=int, default=60, help="Time limit in seconds. (Default: 60)")

    args = parser.parse_args()

    # --- Argument Validation ---
    if args.shieldType == "LaCAM" and args.lacamLookahead == 0:
        print("Error: --lacamLookahead must be set to a positive integer when using shieldType 'LaCAM'.", file=sys.stderr)
        sys.exit(1)
    if args.shieldType == "Real-Time-LaCAM" and args.lacamLookahead not in [0, 1]:
        print("Warning: Real-Time-LaCAM only works with --lacamLookahead 1. Forcing it to 1.", file=sys.stderr)
        args.lacamLookahead = 1

    with open(args.input_yaml, 'r') as f:
        data = yaml.safe_load(f)
    agents_data = data['agents']
    if len(agents_data) < MIN_AGENTS_REQUIRED:
        print(f"Error: The SSIL model requires at least {MIN_AGENTS_REQUIRED} agents to run.", file=sys.stderr)
        print(f"       Your scenario only has {len(agents_data)} agent(s). Please add more agents to '{args.input_yaml}'.", file=sys.stderr)
        sys.exit(1)
        
    map_data = data['map']
    map_name = "custom_map"
    scen_num = 1
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    map_npz_path = os.path.join(args.output_dir, "custom_map.npz")
    output_csv_path = os.path.join(args.output_dir, "output.csv")
    final_yaml_path = os.path.join(args.output_dir, "output.yaml")
    
    create_map_npz(map_data, map_npz_path)
    scen_path = create_scen_file(agents_data, map_name, scen_num, args.output_dir)
    
    # Run simulation and get the path to the temporary paths file
    temp_paths_file = run_simulation(map_npz_path, map_name, scen_path, args.bds_file, len(agents_data), output_csv_path, args)
    
    # Create the final YAML output
    agent_names = [agent['name'] for agent in agents_data]
    create_yaml_output(temp_paths_file, output_csv_path, final_yaml_path, agent_names)

    # Clean up the temporary file
    os.remove(temp_paths_file)
    print(f"Temporary path file {temp_paths_file} removed.")

    print(f"\nSimulation complete. Final output saved to {final_yaml_path}") 