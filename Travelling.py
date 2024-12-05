import numpy as np
import visualize

# Script to parse the data
def parse_tsp_data(configuration):
    coordinates = []
    is_node_section = False  
    cities = []

    with open(configuration, 'r') as file:
        for line in file:
            line = line.strip()
            if line=="NODE_COORD_SECTION":
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
    cur_x, cur_y = cities_cor[cities[0]-1]
    tot_dist = 0
    for i in range(1, len(cities)):
        which_city = cities[i]
        x, y = cities_cor[which_city -1]
        # print(f" first: {cur_x, cur_y} second: {x,y}")
        dist = np.sqrt((x-cur_x)**2 + (y-cur_y)**2)
        tot_dist += dist
        cur_x, cur_y = x,y
    return tot_dist


def coordinate_diff(cor1, cor2):
    return np.sqrt((cor1[0]-cor2[0])**2 + (cor1[1]-cor2[1])**2)

def diff_dist(switch1, switch2, cities_cor, cities):

    n = len(cities) -1
    neighbours1 = []
    neighbours2 = []

    idx1 = cities[switch1]-1
    idx2 = cities[switch2]-1

    # do minus one to get from city name to index -> index starts at 0 and cityname at 1
    if switch1 != 0:
        old_prev_cor1 = cities_cor[cities[switch1-1] -1]
        neighbours1.append(cities[switch1-1])
    # do -2 as the last city in the list is the same as the first one
    else: 
        old_prev_cor1 = cities_cor[cities[n-1] -1]
        neighbours1.append(cities[n-1])
    
    if switch2 != 0:
        old_prev_cor2 = cities_cor[cities[switch2-1] -1]
        neighbours2.append(cities[switch2-1])
    # do -2 as the last city in the list is the same as the first one
    else:
        old_prev_cor2 = cities_cor[cities[n-1] -1]
        neighbours2.append(cities[n-1])

    
    # determine city after current one, no bounds are needed as the last city is the same as the first in the list

    if switch1 >= n or switch2 >= n:
        print("invalid operation, indexes chosen is too high")
    old_next_cor1 = cities_cor[cities[switch1 +1] -1]
    neighbours1.append(cities[switch1 +1])
    old_next_cor2 = cities_cor[cities[switch2 +1] -1]
    neighbours2.append(cities[switch2+1])

    # print(f"hello, I amo index switch1, with value: {cities[switch1]} and neighbours {neighbours1}")
    # print(f"hello, I amo index switch1, with value: {cities[switch2]} and neighbours {neighbours2}")

    
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

    possible_indexes = np.array([i for i in range(length) if i != previdx and i != idx1 and i != nextidx])

    # Randomly pick the second index from the remaining options
    idx2 = np.random.choice(possible_indexes)

    return idx1, idx2


def switch(cor1, cor2, cits):

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
    total_dist = total_length(cities, cities_cor)
    print(f"initial total distance: {total_dist}")
    
    stagnating = 0

    for _ in range(100000000):
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
            print(f"cities switch = {cities[city1], cities[city2]}")
            switch(city1, city2, cities)
            print(cities)
            total_dist += new_dist - old_dist
            print(f"old distance: {old_dist}, new distance {new_dist}")
            print(f"new total distance: {total_dist}")
        else:
            stagnating += 1

            # if no improvement is made for 10000 runs, return. 
            if stagnating == 100000:
                return cities, total_dist
    
    return cities, total_dist





def main():
    cities, cities_cor = parse_tsp_data("TSP-Configurations/eil51.tsp.txt")

    # cities = [1, 2, 3, 4, 5]
    # cities_cor = [(0, 4), (3, 5), (6, 2), (3,3), (2, 0)]

    cities.append(cities[0])
    seed = 32
    found_route, total_dist = mainloop(cities, cities_cor, seed)
    distt = total_length(found_route, cities_cor)


    print(f"found route with distance: {total_dist}, actual dist {distt} \n route: {found_route}")

    visualize.visualize_route(found_route, cities_cor)

if __name__ =="__main__":
    main()
    
