import numpy as np
import visualize


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
    if switch1 != 0:
        old_prev_cor1 = cities_cor[cities[switch1 - 1] - 1]
        neighbours1.append(cities[switch1 - 1])
    # do -2 as the last city in the list is the same as the first one
    else:
        old_prev_cor1 = cities_cor[cities[n - 1] - 1]
        neighbours1.append(cities[n - 1])

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
    
    all_indices = np.arange(length)
    possible_indexes = all_indices[(all_indices != previdx) & (all_indices != idx1) & (all_indices != nextidx)]

    # Randomly pick the second index from the remaining options
    idx2 = np.random.choice(possible_indexes)

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


def mainloop(cities, cities_cor, seed):
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
    total_dist = total_length(cities, cities_cor)
    all_dists = []
    stagnating = 0
    best_dist = total_dist
    best_route = cities

    for _ in range(ITERATIONS):
        all_dists.append(total_dist)
        seed += 1
        city1, city2 = pick_cities(len(cities) - 1, seed)

        # Ensure indices are valid and cities are not neighbors
        assert city1 < len(cities) - 1 and city2 < len(cities) - 1, "Index of city is too big"
        assert abs(city1 - city2) > 1, "Neighbouring cities are swapped"
        
        old_dist, new_dist = diff_dist(city1, city2, cities_cor, cities)
        
        # Perform the swap if it improves the distance
        # insert cooling scheme if distance is worse. 
        if new_dist < old_dist:

            

            stagnating = 0
            # print(f"cities switch = {cities[city1], cities[city2]}")
            switch(city1, city2, cities)
            # print(cities)
            total_dist += new_dist - old_dist

            best_dist = total_dist
            best_route = cities
            # print(f"old distance: {old_dist}, new distance {new_dist}")
            # print(f"new total distance: {total_dist}")
        else:
            stagnating += 1

            # if no improvement is made for 10000 runs, return. 
            # if stagnating == 10000:
                # return cities, total_dist
    
    return all_dists, best_route, best_dist


def multiple_iteations(cities, cities_cor, num_runs, seed):
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
    overall_best_route = []
    overall_best_dist = np.inf

    all_dists_from_runs = []
    for i in range(num_runs):
        print(f"Starting iteration {i}")
        # ensure reproducibility
        seed += 1
        np.random.seed(seed)
        np.random.shuffle(cities)

        cities.append(cities[0])

        # do a run, compute all distances in an iteration
        all_dists, best_route, best_dist = mainloop(cities, cities_cor, seed)
        all_dists_from_runs.append(all_dists)

        # update overall best distance if a new low is computed
        if best_dist < overall_best_dist:
            overall_best_dist = best_dist
            overall_best_route = best_route[:]

        print(f"finished iteration {i}, found route with distance {best_dist}")

    distt = total_length(overall_best_route, cities_cor)
    print(f"found route with distance: {overall_best_dist}, actual dist {distt} \n route: {overall_best_route}")

    return overall_best_dist, overall_best_route, all_dists_from_runs

ITERATIONS = 100000
def main():
    cities, cities_cor = parse_tsp_data("TSP-Configurations/eil51.tsp.txt")

    # adjust number of runs to something else
    num_runs = 10
    orig_seed = 33

    beste_overall_dist, beste_overall_route, distances =  multiple_iteations(cities, cities_cor, num_runs, orig_seed)
    # cities = [1, 2, 3, 4, 5]
    # cities_cor = [(0, 4), (3, 5), (6, 2), (3,3), (2, 0)]
    visualize.visualize_route(beste_overall_route, cities_cor)
    visualize.visualize_developing(distances)

if __name__ =="__main__":
    main()
    
