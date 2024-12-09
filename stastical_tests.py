# performing a pairwise T-test

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

def compare_between_strategies(strategies):
    """
    Perform pairwise T-tests to test significance between cooling strategies
    based on the shortest best distances.
    
    This function compares the minimum best distances across strategies:
    Exponential, Logarithmic, and Linear.
    """
    
    # Perform pairwise comparisons
    results = []
    strategy_names = list(strategies.keys())

    for i in range(len(strategy_names)):
        for j in range(i + 1, len(strategy_names)):
            strat1 = strategy_names[i]
            strat2 = strategy_names[j]
            group1 = strategies[strat1]
            group2 = strategies[strat2]

            t_stat, p_value = ttest_ind(group1, group2, equal_var=False)

            results.append({
                "Strategy 1": strat1,
                "Strategy 2": strat2,
                "t-statistic": t_stat,
                "p-value": p_value,
                "Reject H_0": p_value < 0.05
            })

    # Print results
    for result in results:
        print(f"Comparison: {result['Strategy 1']} vs {result['Strategy 2']}")
        print(f"t-statistic = {result['t-statistic']:.4f}")
        print(f"p-value = {result['p-value']:.4e}")
        print(f"Reject H_0: {result['Reject H_0']}\n")

def compare_within_strategy(strategies):
    """
    Compare means for different hypotheses within a cooling strategy.
    """
    
    for strategy_name, data in strategies.items():
        
        # Labels of T_0 and T_min
        t0_labels = data['label'].apply(lambda x: x.split(',')[0]).unique()
        tmin_labels = data['label'].apply(lambda x: x.split(',')[1]).unique()

        # Compare groups based on T_0
        group_t0_1 = data[data['label'].str.contains(t0_labels[0])]['best_distance']
        group_t0_2 = data[data['label'].str.contains(t0_labels[1])]['best_distance']

        # Perform t-test between the first two T_0 groups
        t_stat_t0, p_value_t0 = ttest_ind(group_t0_1, group_t0_2, equal_var=False)
        reject_h0_t0 = p_value_t0 < 0.05
        print(f"\nT-test for T_0 in {strategy_name} ({t0_labels[0]} vs {t0_labels[1]}):")
        print(f"T-statistic: {t_stat_t0:.4f}, P-value: {p_value_t0:.4e}, Reject H_0: {reject_h0_t0}")

        # Compare groups based on T_min
        group_tmin_1 = data[data['label'].str.contains(tmin_labels[0])]['best_distance']
        group_tmin_2 = data[data['label'].str.contains(tmin_labels[1])]['best_distance']

        # Perform t-test between the first two T_min groups
        t_stat_tmin, p_value_tmin = ttest_ind(group_tmin_1, group_tmin_2, equal_var=False)
        reject_h0_tmin = p_value_tmin < 0.05
        print(f"\nT-test for T_min in {strategy_name} ({tmin_labels[0]} vs {tmin_labels[1]}):")
        print(f"T-statistic: {t_stat_tmin:.4f}, P-value: {p_value_tmin:.4e}, Reject H_0: {reject_h0_tmin}")

if __name__ == "__main__":

    # Load data
    data_Exp = pd.read_csv("best_dist_Exponential.csv")
    #data_Log = pd.read_csv("best_dist_Logarithmic.csv")
    data_Lin = pd.read_csv("best_dist_Linear.csv")

    strategies = {
        "Exponential": data_Exp,
        "Linear": data_Lin,
        # "Logarithmic": data_Log,
    }

    print(80 * '-')
    print("Performing statistical tests for different T_0 and T_min values within one cooling strategy\n")
    compare_within_strategy(strategies)

    distances_Exp = data_Exp['best_distance']
    # distances_Log = data_Log['best_distance']
    distances_Lin = data_Lin['best_distance']

    strategies = {
    "Exponential": distances_Exp,
    "Linear": distances_Lin,
    # "Logarithmic": distances_Log,
    }

    print(80 * '-')
    print("Performing statistical tests between different cooling strategies\n")
    compare_between_strategies(strategies)
    
    print(80 * '-')


