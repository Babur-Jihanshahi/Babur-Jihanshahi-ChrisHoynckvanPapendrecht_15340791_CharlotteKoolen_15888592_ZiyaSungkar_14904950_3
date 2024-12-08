import matplotlib.pyplot as plt
import numpy as np


def visualize_route(cities, opt_tour, coordinates):
    """
    Visualize a route connecting cities based on their coordinates.
    
    Args:
        cities (list): A list of city names corresponding to the route.
        coordinates (list of tuples): A list of (x, y) coordinates for each city.
    """
    plt.figure(figsize=(5, 5))
    order = []
    opt_order = []
    for t, cit in enumerate(cities):
        order.append(coordinates[cit-1])
        opt_order.append(coordinates[opt_tour[t]-1])


    # Unzip the coordinates into x and y lists, prevent first city of being printed twice
    xcor, ycor = zip(*order)
    opt_xcor, opt_ycor = zip(*opt_order)
    partx, party = zip(*order[:-1])

    # Plot the cities as scatter points
    plt.scatter(partx, party, color='blue', zorder=2)

    # Connect the cities with lines in the given order
    
    plt.plot(xcor, ycor, color='green', linestyle='-', zorder=1)
    plt.plot(opt_xcor, opt_ycor, color='red', linestyle='--', alpha = 0.6, zorder=1)
    # Annotate each city with its number

    all_cities = cities[:-1]
    for i, city in enumerate(all_cities):
        plt.text(xcor[i], ycor[i], str(city), fontsize=4, ha='center', va='center', 
                 bbox=dict(boxstyle="circle,pad=0.3", edgecolor='black', facecolor='white'))

    # Add labels and a title
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Route Visualization")
    plt.grid(True)

    # Display the plot
    plt.show()

# def visualize_developing(all_distances):
#     sort_on_index = np.array(all_distances).T
#     means = np.mean(sort_on_index, axis=1)
#     variances = np.var(sort_on_index, axis=1)

#     plt.figure(figsize=(4,3))
#     num_iterations = len(sort_on_index)
#     iters = np.linspace(0, num_iterations, num_iterations)
#     plt.plot(iters, means, label="No Cooling Scheme")

#     stdevv =  np.sqrt(variances) 
#     lower_bound = np.array(means) - stdevv
#     upper_bound = np.array(means) + stdevv
#     plt.fill_between(iters, lower_bound, upper_bound, alpha=0.2)

#     # possibly use a logarithmic scale
#     # plt.xscale("log")
#     plt.xlim(-100, num_iterations)
#     plt.grid()
#     plt.legend()
#     plt.title("Mean distance of calculated route")
#     plt.ylabel("Distance")
#     plt.xlabel("Iteration")
#     plt.show()

def visualize_developing_multiple_lines(all_results):

    plt.figure(figsize=(8, 6))

    for result in all_results:
        distances = result["distances"]
        label = result["label"]
        sort_on_index = np.array(distances).T
        means = np.mean(sort_on_index, axis=1)
        variances = np.var(sort_on_index, axis=1)
        num_iterations = len(sort_on_index)
        iters = np.linspace(0, num_iterations, num_iterations)
        plt.plot(iters, means, label=label)
        stdevv = np.sqrt(variances)
        lower_bound = means - stdevv
        upper_bound = means + stdevv
        plt.fill_between(iters, lower_bound, upper_bound, alpha=0.2)

    plt.xlim(0, num_iterations)
    plt.grid()
    plt.legend()
    plt.title("Mean Distance of Calculated Routes")
    plt.ylabel("Distance")
    plt.xlabel("Iteration")
    plt.show()