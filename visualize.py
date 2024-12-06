import matplotlib.pyplot as plt
import numpy as np


def visualize_route(cities, coordinates):
    """
    Visualize a route connecting cities based on their coordinates.
    
    Args:
        cities (list): A list of city names corresponding to the route.
        coordinates (list of tuples): A list of (x, y) coordinates for each city.
    """
    plt.figure(figsize=(5, 5))
    order = []
    for cit in cities:
        order.append(coordinates[cit-1])


    # Unzip the coordinates into x and y lists, prevent first city of being printed twice
    xcor, ycor = zip(*order)
    partx, party = zip(*order[:-1])

    # Plot the cities as scatter points
    plt.scatter(partx, party, color='blue', zorder=2)

    # Connect the cities with lines in the given order
    
    plt.plot(xcor, ycor, color='green', linestyle='-', zorder=1)

    # Annotate each city with its number

    all_cities = cities[:-1]
    for i, city in enumerate(all_cities):
        plt.text(xcor[i], ycor[i], str(city), fontsize=8, ha='center', va='center', 
                 bbox=dict(boxstyle="circle,pad=0.3", edgecolor='black', facecolor='white'))

    # Add labels and a title
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Route Visualization")
    plt.grid(True)

    # Display the plot
    plt.show()

def visualize_developing(all_distances):
    sort_on_index = np.array(all_distances).T
    means = np.mean(sort_on_index, axis=1)
    variances = np.var(sort_on_index, axis=1)

    plt.figure(figsize=(4,3))
    num_iterations = len(sort_on_index)
    iters = np.linspace(0, num_iterations, num_iterations)
    plt.plot(iters, means, label="No Cooling Scheme")

    stdevv =  np.sqrt(variances) 
    lower_bound = np.array(means) - stdevv
    upper_bound = np.array(means) + stdevv
    plt.fill_between(iters, lower_bound, upper_bound, alpha=0.2)

    # possibly use a logarithmic scale
    # plt.xscale("log")
    plt.xlim(-100, num_iterations)
    plt.grid()
    plt.legend()
    plt.title("Mean distance of calculated route")
    plt.ylabel("Distance")
    plt.xlabel("Iteration")
    plt.show()