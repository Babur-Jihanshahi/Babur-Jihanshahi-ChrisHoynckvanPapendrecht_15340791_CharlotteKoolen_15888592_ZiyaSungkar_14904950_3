import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

def preprocess_data(data):
    """
    Preprocess `best_distances_runs` to ensure all values are numeric -> no errors.
    """
    data['best_distances_runs'] = data['best_distances_runs'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    data = data.explode('best_distances_runs')
    data['best_distances_runs'] = pd.to_numeric(data['best_distances_runs'], errors='coerce')

    return data

def compare_between_strategies(strategies, output_file):
    """
    Perform pairwise t-tests to compare the mean of best distances between strategies for H_0^1.
    """
    with open(output_file, "a") as f:
        f.write("-" * 80 + "\n")
        f.write("Performing statistical tests between different cooling strategies for H_0^1 (Iterations=10,000,000)\n\n")

        # Iterate through each T_0 value
        all_t0_values = set()
        for data in strategies.values():

            # Filter for Iterations=10,000,000
            data['iterations'] = data['label'].apply(lambda x: int(x.split('Iterations=')[-1]))
            filtered_data = data[data['iterations'] == 10_000_000]
            all_t0_values.update(filtered_data['label'].apply(lambda x: x.split(',')[0]))

        all_t0_values = sorted(list(all_t0_values))

        for t0 in all_t0_values:
            strategy_names = list(strategies.keys())
            for i in range(len(strategy_names)):
                for j in range(i + 1, len(strategy_names)):
                    strat1 = strategy_names[i]
                    strat2 = strategy_names[j]

                    # Filter groups based on T_0 and Iterations=10,000,000
                    group1 = strategies[strat1]
                    group1 = group1[group1['iterations'] == 10_000_000]
                    group1 = group1[group1['label'].str.contains(f"^{t0},")]['best_distances_runs']

                    group2 = strategies[strat2]
                    group2 = group2[group2['iterations'] == 10_000_000]
                    group2 = group2[group2['label'].str.contains(f"^{t0},")]['best_distances_runs']

                    # Perform t-test
                    t_stat, p_value = ttest_ind(group1, group2, equal_var=False)

                    # Save results
                    hypothesis = f"H_0^1: E(X)_{strat1},T_0={t0} = E(X)_{strat2},T_0={t0}"
                    f.write(f"Group 1 ({strat1}, T_0={t0}): Mean = {group1.mean()}, Variance = {group1.var()}, Values = {group1.tolist()}\n")
                    f.write(f"Group 2 ({strat2}, T_0={t0}): Mean = {group2.mean()}, Variance = {group2.var()}, Values = {group2.tolist()}\n")
                    f.write(f"Hypothesis: {hypothesis}\n")
                    f.write(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4e}\n")
                    f.write(f"Reject H_0: {'True' if p_value < 0.05 else 'False'}\n\n")

def compare_within_strategy_chain_length(strategies, output_file):
    """
    Perform pairwise t-tests for hypotheses H_0^2 to compare chain lengths within a strategy.
    """
    with open(output_file, "a") as f:
        f.write("-" * 80 + "\n")
        f.write("Performing statistical tests for different chain lengths and T_0 values within one cooling strategy for H_0^2\n")

        for strategy_name, data in strategies.items():
            f.write(f"\nPerforming T-tests for strategy: {strategy_name}\n")

            # Extract unique `Iterations` and `T_0` values
            data['iterations'] = data['label'].apply(lambda x: int(x.split('Iterations=')[-1]))
            data['T_0'] = data['label'].apply(lambda x: x.split(',')[0].split('=')[1])
            
            unique_iterations = sorted(data['iterations'].unique())
            unique_T0_values = sorted(data['T_0'].unique())

            # Iterate over all pairs of Iterations and T_0
            for T0 in unique_T0_values:
                T0_data = data[data['T_0'] == T0]

                for i in range(len(unique_iterations) - 1):
                    for j in range(i + 1, len(unique_iterations)):
                        L1 = unique_iterations[i]
                        L2 = unique_iterations[j]

                        # Filter groups by chain length (by Iterations)
                        group_L1 = T0_data[T0_data['iterations'] == L1]['best_distances_runs']
                        group_L2 = T0_data[T0_data['iterations'] == L2]['best_distances_runs']

                        # Perform t-test
                        t_stat, p_value = ttest_ind(group_L1, group_L2, equal_var=False)

                        # Save results
                        hypothesis = f"H_0^2: E(X)_{strategy_name},T_0={T0},L={L1} = E(X)_{strategy_name},T_0={T0},L={L2}"
                        f.write(f"Group L1 ({strategy_name}, T_0={T0}, L={L1}): Mean = {group_L1.mean()}, Variance = {group_L1.var()}, Values = {group_L1.tolist()}\n")
                        f.write(f"Group L2 ({strategy_name}, T_0={T0}, L={L2}): Mean = {group_L2.mean()}, Variance = {group_L2.var()}, Values = {group_L2.tolist()}\n")
                        f.write(f"Hypothesis: {hypothesis}\n")
                        f.write(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4e}\n")
                        f.write(f"Reject H_0: {'True' if p_value < 0.05 else 'False'}\n\n")

def compare_within_strategy_T0_values(strategies, output_file):
    """
    Perform pairwise t-tests for hypotheses H_0^3 to compare T_0 values within a strategy,
    filtering for Iterations=10,000,000.
    """
    with open(output_file, "a") as f:
        f.write("-" * 80 + "\n")
        f.write("Performing statistical tests for different T_0 values within one cooling strategy for H_0^3\n")

        for strategy_name, data in strategies.items():
            f.write(f"\nPerforming T-tests for strategy: {strategy_name}\n")

            # Filter for Iterations=10,000,000
            data['iterations'] = data['label'].apply(lambda x: int(x.split('Iterations=')[-1]))
            filtered_data = data[data['iterations'] == 10_000_000]

            # # Extract unique T_0 values
            filtered_data = filtered_data.copy()
            filtered_data.loc[:, 'T_0'] = filtered_data['label'].apply(lambda x: x.split(',')[0].split('=')[1])
            t0_values = sorted(filtered_data['T_0'].unique())

            # Perform pairwise comparisons for all unique T_0 values
            for i in range(len(t0_values) - 1):
                for j in range(i + 1, len(t0_values)):
                    T0_1 = t0_values[i]
                    T0_2 = t0_values[j]

                    # Filter groups based on T_0 values
                    group_T0_1 = filtered_data[filtered_data['T_0'] == T0_1]['best_distances_runs']
                    group_T0_2 = filtered_data[filtered_data['T_0'] == T0_2]['best_distances_runs']

                    # Perform t-test
                    t_stat, p_value = ttest_ind(group_T0_1, group_T0_2, equal_var = False)

                    # Save results
                    hypothesis = f"H_0^3: E(X)_{strategy_name},T_0={T0_1} = E(X)_{strategy_name},T_0={T0_2}"
                    f.write(f"Group T_0={T0_1}: Mean = {group_T0_1.mean()}, Variance = {group_T0_1.var()}, Values = {group_T0_1.tolist()}\n")
                    f.write(f"Group T_0={T0_2}: Mean = {group_T0_2.mean()}, Variance = {group_T0_2.var()}, Values = {group_T0_2.tolist()}\n")
                    f.write(f"Hypothesis: {hypothesis}\n")
                    f.write(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4e}\n")
                    f.write(f"Reject H_0: {'True' if p_value < 0.05 else 'False'}\n\n")


if __name__ == "__main__":

    # Load data
    data_Exp = pd.read_csv("data/best_dist_Exponential.csv")
    data_Log = pd.read_csv("data/best_dist_Logarithmic.csv")
    data_Lin = pd.read_csv("data/best_dist_Linear.csv")
    
    # Prepare the data
    data_Exp = preprocess_data(data_Exp)
    data_Log = preprocess_data(data_Log)
    data_Lin = preprocess_data(data_Lin)

    # Cooling strategies
    strategies = {
        "Exponential": data_Exp,
        "Linear": data_Lin,
        "Logarithmic": data_Log,
    }

    # Write results in file
    output_file = "statistical_test_results.txt"
    open(output_file, "w").close()

    compare_between_strategies(strategies, output_file)
    compare_within_strategy_chain_length(strategies, output_file)
    compare_within_strategy_T0_values(strategies, output_file)
    print(f"Results saved to {output_file}")
