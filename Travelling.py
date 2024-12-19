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

def find_temperature_parameters(cities_cor, cities, num_samples=10000):
    differences = []
    min_difference = float('inf')
    max_difference = float('-inf')

    # sample random moves:
    for i in range(num_samples):
        city1, city2 = pick_connection(len(cities) - 1, seed=i)
        old_dist, new_dist = diff_dist(city1, city2, cities_cor, cities)
        diff = new_dist - old_dist
        if diff > 0:
            differences.append(diff)
            min_difference = min(min_difference, diff)
            max_difference = max(max_difference, diff)

    if not differences:
        return find_temperature_parameters(cities_cor, cities, num_samples * 2)
    
    avg_difference = np.mean(differences)
    T_0 = -avg_difference / np.log(0.8)
    T_min = -min_difference / np.log(0.01)
    print(f"Suggested parameters based on {num_samples} samples:")
    print(f'T_0: {T_0:.2f} (will accept moves that increases distance by {avg_difference:.2f} with 80% probability)')
    print(f"T_min: {T_min:.2f} (will accept moves that increase distance by {min_difference:.2f} with 1% probability)")
    return T_0, T_min

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



    idx1 = cities[switch1] - 1
    idx2 = cities[switch2] - 1

    # do minus one to get from city name to index -> index starts at 0 and cityname at 1

    # this can't be the case anymore as index 0 can't ever be picked
    # if switch1 != 0:
    old_prev_cor1 = cities_cor[cities[switch1] - 1]
    # if switch2 != 0:
    old_prev_cor2 = cities_cor[cities[switch2] - 1]
    # do -2 as the last city in the list is the same as the first one
    # else:
    #     old_prev_cor2 = cities_cor[cities[n - 1] - 1]
    # else:
    #     old_prev_cor2 = cities_cor[cities[n - 1] - 1]
    # determine city after current one, no bounds are needed as the last city is the same as the first in the list

    if switch1 >= n or switch2 >= n:
        print("invalid operation, indexes chosen is too high")
    old_next_cor1 = cities_cor[cities[switch1 + 1] - 1]
    old_next_cor2 = cities_cor[cities[switch2 + 1] - 1]

    # calculate old distances
    old_diff_1 = coordinate_diff(old_prev_cor1, old_next_cor1) 
    old_diff_2 = coordinate_diff(old_prev_cor2, old_next_cor2)

    tot_diff_old = old_diff_1 + old_diff_2

    # calculate new differences
    new_diff_1 = coordinate_diff(old_prev_cor1, old_prev_cor2)
    new_diff_2 = coordinate_diff(old_next_cor2, old_next_cor1) 

    tot_diff_new = new_diff_1 + new_diff_2

    return tot_diff_old, tot_diff_new


def pick_connection(length, seed=None):
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
    idx1 = np.random.randint(0, length)
    idx1 = np.random.randint(0, length)

    # Exclude adjacent indexes
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

    # assert idx1 != 0, "idx1 should not be 0"
    # assert idx2 != 0, "idx2 should not be 0"

    return idx1, idx2


def switch(cor1, cor2, cits):
    """
    Swap two connections in the route and update the cyclic route.

    Args:
        cor1 (int): Index of the first connection to swap.
        cor2 (int): Index of the second connection to swap.
        cits (list): Current route of cities.

    Returns:
        list: Updated route after the swap.
    """
    citscopy = cits[:-1]

    if cor1 < cor2:
        citietjes = citscopy[:1] + citscopy[1:cor1+1] + citscopy[cor1+1: cor2+1][::-1] + citscopy[cor2+1:] 
        # print(f"normal swap, before swap: {citscopy}, after swap: {citietjes}, cor1 = {cor1}, cor2 = {cor2}")
    else: #swap with wraparound
        citietjes = citscopy[:1] + citscopy[cor1+1:][::-1] + citscopy[cor2+1 : cor1+1] +  citscopy[1:cor2+1][::-1]
        # print(f"swap with raparound, before swap: {citscopy}, after swap: {citietjes} cor1 = {cor1}, cor2 = {cor2}")

    citietjes = citietjes + [cits[0]]
    return citietjes

def linear_cooling(T_0, T_min, k, iters):
    value =  T_0 - k*(T_0 - T_min)/iters
    # if value < 3:
    #     return 3
    return value


def exponential_cooling(T_0, T_min, t, iters):
    """
    Exponential cooling schedule with calculated alpha.

    Parameters:
    T_0: Initial temperature.
    T_min: Minimum temperature (stopping criterion).
    t: Current iteration number.
    ITERATIONS: Total number of iterations.

    Returns:
    Updated temperature.
    """
    alpha = (T_min / T_0) ** (1 / iters)
    T = T_0 * (alpha ** t)

    return (T) 

def logarithmic_cooling(T_0, T_min, k, iters):
    """
    Logarithmic cooling schedule 

    Parameters:
        T_0 (float): Initial temperature
        T_min (float): Minimum temperature threshold
        k (int): current iteration number
        alpha (float): Cooling speed parameter (0 < alpha <= 1)

    Returns:
    float: Updated temperature based on logarithmic schedule  
    """
    # C = T_0 * np.log(2 + iters) / (T_0 - T_min)
    # T =  C / np.log(2+k)

    # C = (T_0 - T_min)/ np.log(2 + iters)
    # T = T_min + C / np.log(2 + k)

    
    alpha = (T_min * np.log(iters + 2)) / T_0
    # T = T_0 - alpha  #*np.log(k+1)
    T =(alpha * T_0) / np.log(k + 2)


    return (T)
    #return max(T, T_min) # not necessary anymore

def accept(dist_i, dist_j, T_k, seed):
    """
    accepts based upon the evaluation function, wiht lower distance
    with probability one, otherwise probability depends on temperature and difference.
    """
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
    cities, cities_cor, T_0, T_min, proc, seed, iterations = parameters
    
    chain_length = ITERATIONS/iterations
    # cities, cities_cor, T_0, T_min, iteration, seed = parameters
    total_dist = total_length(cities, cities_cor)
    all_dists = []
    variances = []
    best_dist = total_dist
    best_route = cities
    sub_dists = []
    # alpha = 0.5 # should be between 0 < alpha < 1 for exponential cooling
    counter= 1
    evals = 0
    T_k = T_0
    for l in range(ITERATIONS):
        if counter == 0: 
            if EXPONENTIAL_COOLING:
                T_k = exponential_cooling(T_0, T_min, evals, iterations)
            elif LINEAR_COOLING:
                T_k = linear_cooling(T_0, T_min, evals, iterations)
            elif LOGARITHMIC_COOLING:
                T_k = logarithmic_cooling(T_0, T_min, evals,iterations)
            evals+=1

        #for implementing markov chain
        counter = (counter + 1) % chain_length

        sub_dists.append(total_dist)
        if l % 100 == 0:
            all_dists.append(np.mean(sub_dists))
            variances.append(np.var(sub_dists))
            sub_dists = []

        # all_dists.append(total_dist)
        seed += 1
        city1, city2 = pick_connection(len(cities) - 1, seed)

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
    
    
    return all_dists, variances, best_route, best_dist, proc


def multiple_iterations(shuffle_cities, cities_cor, num_runs, T_0, T_min, iterations, seed):
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
    os.makedirs('data', exist_ok=True)
    overall_best_route = []
    overall_best_dist = np.inf
    pars = []
    
    all_dists_from_runs = []
    all_variances_from_runs = []
    for i in range(num_runs):
        print(f"Starting iteration {i}")
        # ensure reproducibility
        seed += 1
        
        np.random.seed(seed)
        np.random.shuffle(shuffle_cities)

        cities = [1] + shuffle_cities + [1]

        parameters = (cities, cities_cor, T_0, T_min, i, seed*51, iterations)
        pars.append(parameters)

        
    best_distances_runs = []
    best_routes_runs = []

    with Pool(PROCESSES) as pool:
        assert PROCESSES < os.cpu_count(), "Lower the number of processes (PROCESSES)"
        print(f"Starting parallel execution for {cooling_strategy} schedule")
        results = pool.map(mainloop, pars)
        for res in results:
            all_dists, variances, best_route, best_dist, proc = res
            all_dists_from_runs.append(list(zip(all_dists, variances)))
            all_variances_from_runs.append(variances)

            best_routes_runs.append(best_route)
            best_distances_runs.append(best_dist)

            # update overall best distance if a new low is computed
            if best_dist < overall_best_dist:
                overall_best_dist = best_dist
                overall_best_route = best_route[:]
        
            
            print(f"finished iteration {proc}, found route with distance {best_dist}")

    distt = total_length(overall_best_route, cities_cor)
    print(f"found route with distance: {overall_best_dist}, actual dist {distt}")

    return overall_best_dist, overall_best_route, all_dists_from_runs, best_distances_runs, best_routes_runs

ITERATIONS = 10000000
PROCESSES=10 # adjust this to more
EXPONENTIAL_COOLING = False
LINEAR_COOLING = True
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
    num_runs = 10
    orig_seed = 36
    shuffle_cities = cities[1:]

    initial_solution = [1] + list(shuffle_cities) + [1]
    base_T0, base_Tmin = find_temperature_parameters(cities_cor, initial_solution)


    T_0_values = [400, 40]
    T_min = 1
    iterations = [100, 10000, 10000000] 

    all_results = []
    best_overall_route_ofsettings = []
    best_overall_dist_ofsettings = 0
    for T_0 in T_0_values:
        for iter in iterations:
            orig_seed +=1
            print(f"Running with T_0 = {T_0}, T_min = {T_min}")
            best_overall_dist, best_overall_route, distances, best_distances_its, best_routes_its = multiple_iterations(
                shuffle_cities, cities_cor, num_runs, T_0, T_min, iter, orig_seed
            )
            
            # Add only the best distance and label to results
            all_results.append({
                "label": f"T_0={T_0}, T_min={T_min}, Iterations={iter}",
                "best_distance": best_overall_dist,
                "distances": distances,
                "best_distances_runs": best_distances_its,
                "best_routes_runs": best_routes_its
            })

            if len(best_overall_route_ofsettings) == 0:
                best_overall_route_ofsettings = best_overall_route.copy()
                best_overall_dist_ofsettings = best_overall_dist
            elif best_overall_dist < best_overall_dist_ofsettings:
                best_overall_route_ofsettings = best_overall_route.copy()
                best_overall_dist_ofsettings = best_overall_dist

            df2 = pd.DataFrame(distances).T
            df2.columns = [f"Run {i+1}" for i in range(len(distances))]
            # empty file
            csv_filename = f"data/{cooling_strategy}/distances_{T_0}_{iter}.csv"
            with open(csv_filename, 'w') as f:
                pass
            df2.to_csv(csv_filename, index=False)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    df = df[['label', 'best_distance', 'best_distances_runs']]
    csv_filename = f"data/best_dist_{cooling_strategy}.csv"
    df.to_csv(csv_filename, index=False)

    # Print the best distance for each run
    for result in all_results:
        label = result["label"]
        best_distance = result["best_distance"]
        print(f"For {label}: Best Distance = {best_distance}")

    # Visualize all results
    print(f"Opitmal tour is: {total_length(opt_tour, cities_cor)}, best found tour: {best_overall_dist_ofsettings}")
    visualize.visualize_developing_multiple_lines(all_results)
    visualize.visualize_route(best_overall_route_ofsettings, opt_tour, cities_cor)

if __name__ =="__main__":
    main()
    