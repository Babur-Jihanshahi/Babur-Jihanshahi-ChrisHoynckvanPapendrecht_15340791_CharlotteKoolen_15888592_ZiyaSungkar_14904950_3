import numpy as np
import visualize
import os
from multiprocessing import Pool
import pandas as pd

def parse_optimal_tour(file_path):
    """
    Reads a .txt file formatted as a TSP tour and extracts the numbers under TOUR_SECTION into a list.

    Parameters:
        file_path (str): Path to the .txt file.
    
    Returns:
        list: A list of integers representing the tour.
    """
    numbers = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # Look for the start of the TOUR_SECTION
        in_tour_section = False
        for line in lines:
            stripped_line = line.strip()
            
            if stripped_line == "TOUR_SECTION":
                in_tour_section = True
                continue
            
            # Stop reading if the section ends with "-1"
            if stripped_line == "-1":
                break
            
            # If in TOUR_SECTION, collect numbers
            if in_tour_section:
                if stripped_line.isdigit():
                    numbers.append(int(stripped_line))
    
    return numbers

def parse_tsp_data(configuration):
    """
    Parse a TSP configuration file to extract city coordinates and identifiers.

    Args:
        configuration (str): Path to the TSP configuration file.

    Returns:
        tuple:
            cities (list): List of city identifiers.
            coordinates (list of tuples): List of (x, y) coordinates for each city.
    """
    coordinates = []
    is_node_section = False  
    cities = []

    with open(configuration, 'r') as file:
        for line in file:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                is_node_section = True
                continue
            if line.startswith("EOF"):
                break
            if is_node_section:
                parts = line.split()
                if len(parts) == 3:  # Expecting ID, X, Y
                    i, x, y = parts
                    coordinates.append((int(x), int(y)))
                    cities.append(int(i))

    return cities, coordinates


def total_length(cities, cities_cor):
    """
    Calculate the total distance of a TSP route.

    Args:
        cities (list): List of city identifiers in the order of the route.
        cities_cor (list of tuples): Coordinates for each city.

    Returns:
        float: Total distance of the route.
    """
    cur_x, cur_y = cities_cor[cities[0] - 1]
    tot_dist = 0
    for i in range(1, len(cities)):
        which_city = cities[i]
        x, y = cities_cor[which_city - 1]
        dist = np.sqrt((x - cur_x) ** 2 + (y - cur_y) ** 2)
        tot_dist += dist
        cur_x, cur_y = x, y
    return tot_dist


def coordinate_diff(cor1, cor2):
    """
    Args:
        cor1 and cor2 (tuples): First and second coordinates as (x, y).

    Returns:
        float: Euclidean distance between the two coordinates.
    """
    return np.sqrt((cor1[0] - cor2[0]) ** 2 + (cor1[1] - cor2[1]) ** 2)




def diff_dist(switch1, switch2, cities_cor, cities):
    """
    Calculate the change in total distance if two cities in the route are swapped.

    Args:
        switch1 (int): Index of the first city to swap.
        switch2 (int): Index of the second city to swap.
        cities_cor (list of tuples): Coordinates for each city.
        cities (list): Current route of cities.

    Returns:
        tuple:
            tot_diff_old (float): Total distance of the route before the swap.
            tot_diff_new (float): Total distance of the route after the swap.
    """
    n = len(cities) - 1
    neighbours1 = []
    neighbours2 = []

    idx1 = cities[switch1] - 1
    idx2 = cities[switch2] - 1

    # do minus one to get from city name to index -> index starts at 0 and cityname at 1

    # this can't be the case anymore as index 0 can't ever be picked
    if switch1 != 0:
        old_prev_cor1 = cities_cor[cities[switch1 - 1] - 1]
        neighbours1.append(cities[switch1 - 1])
    # do -2 as the last city in the list is the same as the first one
    else:
        old_prev_cor1 = cities_cor[cities[n - 1] - 1]
        neighbours1.append(cities[n - 1])

    # this can't be the case anymore because index 0 can't be picked 
    if switch2 != 0:
        old_prev_cor2 = cities_cor[cities[switch2 - 1] - 1]
        neighbours2.append(cities[switch2 - 1])
    # do -2 as the last city in the list is the same as the first one
    else:
        old_prev_cor2 = cities_cor[cities[n - 1] - 1]
        neighbours2.append(cities[n - 1])

    # determine city after current one, no bounds are needed as the last city is the same as the first in the list

    if switch1 >= n or switch2 >= n:
        print("invalid operation, indexes chosen is too high")
    old_next_cor1 = cities_cor[cities[switch1 + 1] - 1]
    neighbours1.append(cities[switch1 + 1])
    old_next_cor2 = cities_cor[cities[switch2 + 1] - 1]
    neighbours2.append(cities[switch2 + 1])

    # calculate old distances
    old_diff_1 = coordinate_diff(old_prev_cor1, cities_cor[idx1]) + coordinate_diff(cities_cor[idx1], old_next_cor1)
    old_diff_2 = coordinate_diff(old_prev_cor2, cities_cor[idx2]) + coordinate_diff(cities_cor[idx2], old_next_cor2)

    tot_diff_old = old_diff_1 + old_diff_2

    # calculate new differences
    new_diff_1 = coordinate_diff(old_prev_cor1, cities_cor[idx2]) + coordinate_diff(cities_cor[idx2], old_next_cor1)
    new_diff_2 = coordinate_diff(old_prev_cor2, cities_cor[idx1]) + coordinate_diff(cities_cor[idx1], old_next_cor2)

    tot_diff_new = new_diff_1 + new_diff_2

    return tot_diff_old, tot_diff_new


def pick_cities(length, seed=None):
    """
    Picks two random indexes from a list such that they are not adjacent.
    
    Args:
        length (int): Length of the list.
    
    Returns:
        tuple: Two indexes that are not next to each other.
    """

    if seed is not None:
        np.random.seed(seed)

    # Randomly pick the first index
    idx1 = np.random.randint(1, length)

    # Exclude adjacent indexes
    # this will never be happen as first index is currently fixed
    if idx1 == 0: 
        previdx = length-1
    else:
        previdx = idx1 -1
    if idx1 == length-1:
        nextidx = 0
    else:
        nextidx = idx1+1
    
    all_indices = np.arange(1, length)
    possible_indexes = all_indices[(all_indices != previdx) & (all_indices != idx1) & (all_indices != nextidx)]

    # Randomly pick the second index from the remaining options
    idx2 = np.random.choice(possible_indexes)

    assert idx1 != 0, "idx1 should not be 0"
    assert idx2 != 0, "idx2 should not be 0"

    return idx1, idx2


def switch(cor1, cor2, cits):
    """
    Swap two cities in the route and update the cyclic route.

    Args:
        cor1 (int): Index of the first city to swap.
        cor2 (int): Index of the second city to swap.
        cits (list): Current route of cities.

    Returns:
        list: Updated route after the swap.
    """
    switch1 = cits[cor1]
    cits[cor1] = cits[cor2]

    # last index and first index has the same value
    if cor1 == 0:
        cits[-1] = cits[cor2]
    cits[cor2] = switch1

    # last index and first index has the same value
    if cor2 == 0:
        cits[-1] = switch1

    return cits 

def linear_cooling(T_0, T_min, k):
    return T_0 - k*(T_0 - T_min)/ITERATIONS

def exponential_cooling(T_0, T_min, alpha, t):
    """
    Exponential cooling schedule.

    Parameters:
    T_0: Initial temperature.
    T_min: Minimum temperature (stopping criterion).
    alpha: Cooling rate (0 < alpha < 1).
    t: Current iteration number.

    Returns:
    Updated temperature.
    """
    T = T_0 * (alpha ** t)
    return max(T, T_min)  # Temperature not below T_min

def accept(dist_i, dist_j, T_k, seed):
    if dist_j <= dist_i:
        return True
    prob_accept = np.exp(-(dist_j - dist_i)/T_k)
    np.random.seed(seed)
    if np.random.rand() < prob_accept:
        # print(f"accepted with probabiltiy {prob_accept}, difference in distance {dist_j - dist_i}\n Tk value {T_k}")
        return True
    return False


def mainloop(parameters):
    """
    Perform a single optimization loop for the TSP problem.

    Args:
        cities (list): List of city identifiers for the route.
        cities_cor (list of tuples): Coordinates for each city.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple:
            all_dists (list): Distances at each iteration.
            best_route (list): Best route found during the loop.
            best_dist (float): Shortest distance found during the loop.
    """

    cities, cities_cor, T_0, T_min, iteration, seed = parameters
    total_dist = total_length(cities, cities_cor)
    all_dists = []
    best_dist = total_dist
    best_route = cities
    alpha = 0.5 # should be between 0 < alpha < 1 for exponential cooling

    for l in range(ITERATIONS):
        if EXPONENTIAL_COOLING:
            T_k = exponential_cooling(T_0, T_min, alpha, l)
        elif LINEAR_COOLING:
            T_k = linear_cooling(T_0, T_min, l)
        elif LOGARITHMIC_COOLING:
            print("add logarithmic function") # add function here

        all_dists.append(total_dist)
        seed += 1
        city1, city2 = pick_cities(len(cities) - 1, seed)

        # Ensure indices are valid and cities are not neighbors
        assert city1 < len(cities) - 1 and city2 < len(cities) - 1, "Index of city is too big"
        assert abs(city1 - city2) > 1, "Neighbouring cities are swapped"
        
        old_dist, new_dist = diff_dist(city1, city2, cities_cor, cities)
        
        # Perform the swap if it improves the distance
        # insert cooling scheme if distance is worse. 
        seed+=1
        if accept(old_dist, new_dist, T_k, seed):
            cities = switch(city1, city2, cities)
            total_dist += new_dist - old_dist

            # save the best found route so far
            if total_dist <= best_dist:
                best_dist = total_dist
                best_route = cities[:]
    
    return all_dists, best_route, best_dist, iteration


def multiple_iterations(shuffle_cities, cities_cor, num_runs, T_0, T_min, seed):
    """
    Run multiple iterations of a TSP optimization to find the best route.

    Args:
        cities (list): List of city identifiers for the TSP route.
        cities_cor (list of tuples): Coordinates for each city.
        num_runs (int): Number of iterations to perform.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (best distance, best route, distances from all runs).
    """
    # overall_best_route = []
    # overall_best_dist = np.inf

    # all_dists_from_runs = []
    # for i in range(num_runs):
    #     print(f"Starting iteration {i}")
    #     # ensure reproducibility
    #     seed += 1
        
    #     np.random.seed(seed)
    #     np.random.shuffle(shuffle_cities)

    #     cities = [1] + shuffle_cities + [1]

    #     # do a run, compute all distances in an iteration
    #     all_dists, best_route, best_dist = mainloop(cities, cities_cor, T_0, T_min, seed)
    #     all_dists_from_runs.append(all_dists)

    #     # update overall best distance if a new low is computed
    #     if best_dist < overall_best_dist:
    #         overall_best_dist = best_dist
    #         overall_best_route = best_route[:]

    #     print(f"finished iteration {i}, found route with distance {best_dist}")

    # distt = total_length(overall_best_route, cities_cor)
    # print(f"found route with distance: {overall_best_dist}, actual dist {distt} \n route: {overall_best_route}")

    # return overall_best_dist, overall_best_route, all_dists_from_runs
    overall_best_route = []
    overall_best_dist = np.inf
    pars = []
    
    all_dists_from_runs = []
    for i in range(num_runs):
        print(f"Starting iteration {i}")
        # ensure reproducibility
        seed += 1
        
        np.random.seed(seed)
        np.random.shuffle(shuffle_cities)

        cities = [1] + shuffle_cities + [1]

        parameters = (cities, cities_cor, T_0, T_min, i, seed)
        pars.append(parameters)

        
    
    with Pool(PROCESSES) as pool:
        assert PROCESSES < os.cpu_count(), "Lower the number of processes (PROCESSES)"
        print(f"Starting parallel execution for linear convergence")
        for res in pool.imap_unordered(mainloop, pars):
            all_dists, best_route, best_dist, iteration = res
            all_dists_from_runs.append(all_dists)

            # update overall best distance if a new low is computed
            if best_dist < overall_best_dist:
                overall_best_dist = best_dist
                overall_best_route = best_route[:]
        
            
            print(f"finished iteration {iteration}, found route with distance {best_dist}")

    distt = total_length(overall_best_route, cities_cor)
    print(f"found route with distance: {overall_best_dist}, actual dist {distt} \n route: {overall_best_route}")

    return overall_best_dist, overall_best_route, all_dists_from_runs

#ITERATIONS = 5000000
ITERATIONS = 50000 # lowered this to test the results for visualization
PROCESSES=2 # adjust this to more
EXPONENTIAL_COOLING = True
LINEAR_COOLING = False
LOGARITHMIC_COOLING = False

if EXPONENTIAL_COOLING:
    cooling_strategy = "Exponential"
elif LINEAR_COOLING:
    cooling_strategy = "Linear"
elif LOGARITHMIC_COOLING:
    cooling_strategy = "Logarithmic"

def main():
    cities, cities_cor = parse_tsp_data("TSP-Configurations/a280.tsp.txt")
    opt_tour = parse_optimal_tour("TSP-Configurations/a280.opt.tour.txt")
    opt_tour.append(1)

    # adjust number of runs to something else
    num_runs = 5
    orig_seed = 33
    shuffle_cities = cities[1:]

    # vary with these values to get different stepsizes
    T_0_values = [100, 90]  # change T_0 values later
    T_min_values = [1.2, 0.1]  # change T_min values later
    all_results = []

    for T_0 in T_0_values:
        for T_min in T_min_values:
            print(f"Running with T_0 = {T_0}, T_min = {T_min}")
            best_overall_dist, best_overall_route, distances = multiple_iterations(
                shuffle_cities, cities_cor, num_runs, T_0, T_min, orig_seed
            )
            
            # Add only the best distance and label to results
            all_results.append({
                "label": f"T_0={T_0}, T_min={T_min}",
                "best_distance": best_overall_dist,
                "distances": distances
            })

            # Convert to DataFrame
            df = pd.DataFrame(all_results)
            df = df[['label', 'best_distance']]
            csv_filename = f"best_dist_{cooling_strategy}.csv"
            df.to_csv(csv_filename, index=False)

            # Print the best distance for each run
            for result in all_results:
                label = result["label"]
                best_distance = result["best_distance"]
                print(f"For {label}: Best Distance = {best_distance}")

    # Visualize all results
    visualize.visualize_developing_multiple_lines(all_results)
    visualize.visualize_route(best_overall_route, opt_tour, cities_cor)

if __name__ =="__main__":
    main()
    