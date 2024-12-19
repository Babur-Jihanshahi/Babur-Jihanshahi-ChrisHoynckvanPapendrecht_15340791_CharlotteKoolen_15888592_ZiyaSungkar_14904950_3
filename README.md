Babur-Jihanshahi-ChrisHoynckvanPapendrecht_15340791_CharlotteKoolen_15888592_ZiyaSungkar_14904950_3


# Solving the Traveling Salesman Problem using Simulated Annealing

This project demonstrates how to solve the Traveling Salesman Problem (TSP) using **Simulated Annealing** (SA). This parallelized implementation includes different cooling schedules (Linear, Exponential, and Logarithmic) and focuses on optimizing the total distance of a TSP route.

## Key Features
- **Simplifying Assumptions**:
  - Paths exist between all cities.
  - The triangle inequality holds for all paths.
  - The problem is symmetric (distance A → B = distance B → A).
- **Elementary Edit**: 
  - The algorithm uses 2-opt moves to update the route.
- **Cooling Schedules**:
  - Linear Cooling
  - Exponential Cooling
  - Logarithmic Cooling

## Implementation
The main functions include:
- `total_length`: Calculates the total distance of a TSP route.
- `coordinate_diff`: Computes the Euclidean distance between two points.
- `switch`: Swaps two connections in the route, and reverses the order between these edges. wraps the route around city 1.  
- `diff_dist`: Computes the change in distance when two cities are swapped.
- `linear_cooling`, `exponential_cooling`, `logarithmic_cooling`: Implement different cooling strategies.
- `accept`: Determines whether to accept a move based on the Metropolis criterion.
- `mainloop`: Runs the simulated annealing optimization for one iteration.

## How to Run
1. Choose (or place a new) TSP configuration file from the `TSP-Configurations` folder.
2. Run the script using:
   ```bash
   python Travelling.py
   ```
3. Configure parameters such as `T_0`, `T_min`, `iterations`, and cooling schedules in the `main` function.

## File Structure
- **data/**: Stores output results for each cooling schedule.
- **plots/**: Contains visualizations of route development.
- **TSP-Configurations/**: Configuration files for TSP instances.
- **Travelling.py**: Main script for running the simulated annealing algorithm.
- **visualize.py**: Visualizes routes and distance progress.

## Example Usage
Run the simulation with default parameters, or experiment with different cooling schedules and parameters like `T_0`, `T_min` and `iterations`.

