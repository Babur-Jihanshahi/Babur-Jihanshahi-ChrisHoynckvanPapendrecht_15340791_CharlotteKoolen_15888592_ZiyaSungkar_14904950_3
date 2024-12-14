import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
from matplotlib.ticker import FuncFormatter

def visualize_route(cities, opt_tour, coordinates):
    """
    Visualize a route connecting cities based on their coordinates.
    
    Args:
        cities (list): A list of city names corresponding to the route.
        coordinates (list of tuples): A list of (x, y) coordinates for each city.
    """
    plt.figure(figsize=(5, 5), dpi=300)
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
    plt.scatter(partx, party, color='black', zorder=2)

    # Connect the cities with lines in the given order
    
    plt.plot(xcor, ycor, color='blue', linestyle='-', zorder=1, alpha=0.6, label="Found Route")
    plt.plot(opt_xcor, opt_ycor, color='red', linestyle='-', alpha = 0.6, zorder=1, label="Optimum Route")
    # Annotate each city with its number

    all_cities = cities[:-1]
    for i, city in enumerate(all_cities):
        plt.text(xcor[i], ycor[i], str(city), fontsize=3, ha='center', va='center', 
                 bbox=dict(boxstyle="circle,pad=0.3", edgecolor='purple', facecolor='white'))

    # Add labels and a title
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("TSP Route")
    plt.legend()
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
        distances = [[item[0] for item in sublist] for sublist in result["distances"]]
        variances = [[item[1] for item in sublist] for sublist in result["distances"]]
        label = result["label"]
        sort_on_index = np.array(distances).T
        means = np.mean(sort_on_index, axis=1)
        variances = np.var(sort_on_index, axis=1)
        num_iterations = len(sort_on_index)

        # Group the means and variances
        group_size = 10
        num_iterations = len(means)
        num_groups = num_iterations // group_size
        grouped_means = means.reshape(-1, group_size).mean(axis=1)
        grouped_variances = variances.reshape(-1, group_size).mean(axis=1)

        # Create grouped x-axis
        grouped_iters = np.linspace(0, num_iterations, len(grouped_means))

        # Calculate standard deviation bounds for visualization
        grouped_stdev = np.sqrt(grouped_variances)
        lower_bound = grouped_means - grouped_stdev
        upper_bound = grouped_means + grouped_stdev

        # Plot the grouped data
        plt.plot(grouped_iters, grouped_means, label=label)
        plt.fill_between(grouped_iters, lower_bound, upper_bound, alpha=0.2)

    plt.xlim(0, num_iterations)
    plt.grid()
    plt.legend()
    plt.title("Mean Distance of Calculated Routes")
    plt.ylabel("Distance")
    plt.xlabel("Iteration")
    plt.show()

def parse_tuple(value):
    try:
        # Evaluate the string representation of the tuple
        return eval(value)
    except Exception as e:
        print(f"Error parsing value: {value}, error: {e}")
        return (np.nan, np.nan)  # Return NaN tuple if parsing fails

def visualize_runs(whichrun):
    group_size = 200
    labels= [r"$T_0: 40$, MC:$100000$", r"$T_0: 40$, MC:$1000$", r"$T_0: 40$, MC:$1$", r"$T_0: 400$, MC:$100000$", r"$T_0: 400$, MC:$1000$", r"$T_0: 400$, MC:$1$"]
    i = 0
    plt.figure(figsize=(4, 4.5), dpi=300)
    for tzero in [40 , 400]:
        for montecarlo in [100, 10000, 10000000]:
            data = pd.read_csv(f'data/{whichrun}/distances_{tzero}_{montecarlo}.csv')
            # data = pd.read_csv(f'data/{whichrun}/dummy.csv')
            data = data.applymap(parse_tuple)
            
            means_df = data.applymap(lambda x: x[0]).to_numpy()
            variances_df = data.applymap(lambda x: x[1]).to_numpy()

            means_df = means_df.T
            variances_df = variances_df.T

            grouped_columns_means = [col.reshape(-1, group_size).tolist() for col in means_df]
            mean_data_col = [[np.mean(group) for group in run] for run in grouped_columns_means]
            var_data_col = [[np.var(group) for group in run] for run in grouped_columns_means]
            # grouped_columns_vars = [col.reshape(-1, group_size).var() for col in means_df]
            column_vars = [col.reshape(-1, group_size).tolist() for col in variances_df]
            mean_column_vars = [[np.mean(group) + var_data_col[j][i] for i,group in enumerate(run)] for j,run in enumerate(column_vars)]
            print(len(mean_column_vars))

            mean_between_runs = np.mean(mean_data_col, axis=0)
            var_between_runs = np.mean(mean_column_vars, axis=0) + np.var(mean_data_col, axis=0)
            print(var_between_runs)
            
            # Create grouped x-axis
            grouped_iters = np.linspace(0, 10, len(mean_between_runs))

            # Calculate standard deviation bounds for visualization
            grouped_stdev = np.sqrt(var_between_runs)
            lower_bound = mean_between_runs - grouped_stdev
            upper_bound = mean_between_runs + grouped_stdev

            # Plot the grouped data
            plt.plot(grouped_iters, mean_between_runs, linewidth=1, label=labels[i], alpha=0.85)
            plt.fill_between(grouped_iters, lower_bound, upper_bound, alpha=0.33)
            i+=1


    def format_func(value, tick_number):
        return f"{int(value / 1000)}"

    # Apply the formatter
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_func))
    plt.xlim(0, 10)
    plt.grid()
    plt.legend(fontsize=8)
    plt.title(f"Convergence Behavriour {whichrun}")
    plt.ylabel(r"Distance $\times 10^3$")
    plt.xlabel(r"Iteration $\times 10^6$")
    plt.show()
            


if __name__ == "__main__":
    visualize_runs("Exponential")
    