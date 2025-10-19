import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from scipy import stats
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
from multiplex_core import MultiplexTrial


def load_experiment_config(config_path="experiment_config.json"):
    """
    Load experiment configuration from JSON file
    """
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: Config file {config_path} not found. Using default configuration.")
        return {
            "groups": {"control": [], "experimental": []},
            "comparisons": [],
            "analysis_settings": {
                "midline_borders": 0.6,
                "threshold": 4,
                "multiple_testing_correction": "bonferroni",
                "significance_level": 0.05
            }
        }


def collect_trial_data(folder_path, threshold=4, midline_borders=0.6, filter_phase='both'):
    """
    Collect data from all trials in experiment folder with comprehensive tracking
    
    Parameters:
    -----------
    folder_path : str
        Path to the experiment folder
    threshold : int
        Minimum choices required for valid fly
    midline_borders : float
        Midline border threshold (0.0 to 1.0)
    filter_phase : str
        Which phases to filter ('both', 'initial', 'test', or 'none')
    
    Returns:
        DataFrame with columns: genotype, trial_date, trial_number, fly_id,
        learned_index, cs_plus_odor, initial_valence_cs_side, test_cs_side,
        sides_switched, num_valid_flies, num_total_flies
    """
    all_results = []
    
    print(f"Scanning experiment folder: {folder_path}")
    
    for date_folder in os.listdir(folder_path):
        date_path = os.path.join(folder_path, date_folder)
        if not os.path.isdir(date_path):
            continue
            
        print(f"  Processing date folder: {date_folder}")
        
        for trial_folder in os.listdir(date_path):
            trial_path = os.path.join(date_path, trial_folder)
            if not os.path.isdir(trial_path):
                continue
                
            # Check for required files
            metadata_path = os.path.join(trial_path, 'experiment_metadata.json')
            data_path = os.path.join(trial_path, 'fly_loc.csv')
            
            if not (os.path.exists(metadata_path) and os.path.exists(data_path)):
                print(f"    Skipping {trial_folder}: Missing required files")
                continue
            
            try:
                # Load metadata for genotype
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                fly_genotype = metadata.get('flyGenotype', 'unknown')
                
                print(f"    Processing trial: {trial_folder} (genotype: {fly_genotype})")
                
                # Create trial object and run analysis
                trial = MultiplexTrial()
                trial.load_data(data_path)
                trial.filter_by_num_choices(midline_borders, threshold, filter_phase)
                
                # Get results with CS+ information
                results_df = trial.analyse_time()
                
                # Add metadata to each fly's results
                for _, row in results_df.iterrows():
                    all_results.append({
                        'genotype': fly_genotype,
                        'trial_date': date_folder,
                        'trial_number': trial_folder,
                        'fly_id': row['fly_id'],
                        'learned_index': row['learned_index'],
                        'cs_plus_odor': row['cs_plus_odor'],
                        'initial_valence_cs_side': row['initial_valence_cs_side'],
                        'test_cs_side': row['test_cs_side'],
                        'sides_switched': row['sides_switched'],
                        'initial_valence_cs_preference': row['initial_valence_cs_preference'],
                        'test_cs_preference': row['test_cs_preference'],
                        'valid_fly': row['valid_fly']
                    })
                
                print(f"      Added {len(results_df)} flies from {trial_folder}")
                
            except Exception as e:
                print(f"    Error processing {trial_folder}: {str(e)}")
                continue
    
    print(f"Total flies collected: {len(all_results)}")
    return pd.DataFrame(all_results)


def test_assumptions(data1, data2, group1_name, group2_name):
    """
    Test statistical assumptions for parametric tests
    
    Returns:
        dict with test results and recommendations
    """
    results = {
        'group1_name': group1_name,
        'group2_name': group2_name,
        'n1': len(data1),
        'n2': len(data2),
        'normality_passed': False,
        'homogeneity_passed': False,
        'recommended_test': 'mannwhitneyu'
    }
    
    # Test normality (Shapiro-Wilk)
    if len(data1) >= 3 and len(data2) >= 3:
        try:
            _, p1 = shapiro(data1)
            _, p2 = shapiro(data2)
            results['normality_p1'] = p1
            results['normality_p2'] = p2
            results['normality_passed'] = p1 > 0.05 and p2 > 0.05
        except:
            results['normality_passed'] = False
    
    # Test homogeneity of variance (Levene's test)
    if len(data1) >= 2 and len(data2) >= 2:
        try:
            _, p_homogeneity = levene(data1, data2)
            results['homogeneity_p'] = p_homogeneity
            results['homogeneity_passed'] = p_homogeneity > 0.05
        except:
            results['homogeneity_passed'] = False
    
    # Determine recommended test
    if results['normality_passed'] and results['homogeneity_passed']:
        results['recommended_test'] = 'ttest_ind'
    elif results['normality_passed'] and not results['homogeneity_passed']:
        results['recommended_test'] = 'ttest_ind_unequal'
    else:
        results['recommended_test'] = 'mannwhitneyu'
    
    return results


def perform_statistical_test(data1, data2, test_type):
    """
    Perform the appropriate statistical test
    
    Returns:
        dict with test results
    """
    if test_type == 'ttest_ind':
        statistic, p_value = ttest_ind(data1, data2)
        test_name = "Independent t-test"
    elif test_type == 'ttest_ind_unequal':
        statistic, p_value = ttest_ind(data1, data2, equal_var=False)
        test_name = "Welch's t-test"
    elif test_type == 'mannwhitneyu':
        statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
        test_name = "Mann-Whitney U test"
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    # Calculate effect size
    if test_type in ['ttest_ind', 'ttest_ind_unequal']:
        # Cohen's d
        pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                             (len(data2) - 1) * np.var(data2, ddof=1)) / 
                            (len(data1) + len(data2) - 2))
        effect_size = (np.mean(data1) - np.mean(data2)) / pooled_std
        effect_size_name = "Cohen's d"
    else:
        # Rank-biserial correlation
        n1, n2 = len(data1), len(data2)
        effect_size = 1 - (2 * statistic) / (n1 * n2)
        effect_size_name = "Rank-biserial correlation"
    
    # Calculate confidence interval for mean difference
    mean_diff = np.mean(data1) - np.mean(data2)
    se_diff = np.sqrt(np.var(data1, ddof=1)/len(data1) + np.var(data2, ddof=1)/len(data2))
    ci_lower = mean_diff - 1.96 * se_diff
    ci_upper = mean_diff + 1.96 * se_diff
    
    return {
        'test_name': test_name,
        'statistic': statistic,
        'p_value': p_value,
        'effect_size': effect_size,
        'effect_size_name': effect_size_name,
        'mean_difference': mean_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


def perform_statistical_analysis(data, config):
    """
    Perform comprehensive statistical analysis with assumption testing
    """
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    results = []
    comparisons = config.get('comparisons', [])
    
    if not comparisons:
        print("No comparisons defined in configuration.")
        return pd.DataFrame()
    
    for i, comparison in enumerate(comparisons):
        exp_genotype = comparison['experimental']
        control_genotypes = comparison['controls']
        
        print(f"\nComparison {i+1}: {exp_genotype} vs {', '.join(control_genotypes)}")
        print("-" * 50)
        
        # Get experimental data
        exp_data = data[data['genotype'] == exp_genotype]['learned_index'].dropna()
        
        if len(exp_data) == 0:
            print(f"  Warning: No data found for experimental group {exp_genotype}")
            continue
        
        for control_genotype in control_genotypes:
            # Get control data
            ctrl_data = data[data['genotype'] == control_genotype]['learned_index'].dropna()
            
            if len(ctrl_data) == 0:
                print(f"  Warning: No data found for control group {control_genotype}")
                continue
            
            print(f"\n  {exp_genotype} (n={len(exp_data)}) vs {control_genotype} (n={len(ctrl_data)})")
            
            # Test assumptions
            assumptions = test_assumptions(exp_data, ctrl_data, exp_genotype, control_genotype)
            
            print(f"    Normality test: p1={assumptions.get('normality_p1', 'N/A'):.3f}, "
                  f"p2={assumptions.get('normality_p2', 'N/A'):.3f} "
                  f"({'PASSED' if assumptions['normality_passed'] else 'FAILED'})")
            print(f"    Homogeneity test: p={assumptions.get('homogeneity_p', 'N/A'):.3f} "
                  f"({'PASSED' if assumptions['homogeneity_passed'] else 'FAILED'})")
            print(f"    Recommended test: {assumptions['recommended_test']}")
            
            # Perform statistical test
            test_results = perform_statistical_test(exp_data, ctrl_data, assumptions['recommended_test'])
            
            print(f"    {test_results['test_name']}: statistic={test_results['statistic']:.3f}, "
                  f"p={test_results['p_value']:.3f}")
            print(f"    Effect size ({test_results['effect_size_name']}): {test_results['effect_size']:.3f}")
            print(f"    Mean difference: {test_results['mean_difference']:.3f} "
                  f"(95% CI: {test_results['ci_lower']:.3f}, {test_results['ci_upper']:.3f})")
            
            # Store results
            results.append({
                'comparison': f"{exp_genotype} vs {control_genotype}",
                'experimental_group': exp_genotype,
                'control_group': control_genotype,
                'n_experimental': len(exp_data),
                'n_control': len(ctrl_data),
                'test_name': test_results['test_name'],
                'statistic': test_results['statistic'],
                'p_value': test_results['p_value'],
                'effect_size': test_results['effect_size'],
                'effect_size_name': test_results['effect_size_name'],
                'mean_difference': test_results['mean_difference'],
                'ci_lower': test_results['ci_lower'],
                'ci_upper': test_results['ci_upper'],
                'normality_passed': assumptions['normality_passed'],
                'homogeneity_passed': assumptions['homogeneity_passed']
            })
    
    # Apply multiple testing correction
    if results:
        results_df = pd.DataFrame(results)
        correction_method = config.get('analysis_settings', {}).get('multiple_testing_correction', 'bonferroni')
        
        if correction_method == 'bonferroni':
            results_df['corrected_p_value'] = results_df['p_value'] * len(results_df)
            results_df['corrected_p_value'] = results_df['corrected_p_value'].clip(upper=1.0)
        else:
            # Benjamini-Hochberg FDR
            from scipy.stats import false_discovery_control
            results_df['corrected_p_value'] = false_discovery_control(results_df['p_value'])
        
        print(f"\nMultiple testing correction ({correction_method}):")
        for _, row in results_df.iterrows():
            significance = "***" if row['corrected_p_value'] < 0.001 else \
                          "**" if row['corrected_p_value'] < 0.01 else \
                          "*" if row['corrected_p_value'] < 0.05 else "ns"
            print(f"  {row['comparison']}: p={row['p_value']:.3f} -> "
                  f"p_corrected={row['corrected_p_value']:.3f} {significance}")
        
        return results_df
    
    return pd.DataFrame()


def create_plots(data, stats_results, output_folder):
    """
    Create comprehensive plots with dynamic limits and CS+ information
    """
    print(f"\nCreating plots in {output_folder}")
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Calculate dynamic y-axis limits
    all_values = data['learned_index'].dropna()
    y_min = all_values.min() - 0.1 * (all_values.max() - all_values.min())
    y_max = all_values.max() + 0.1 * (all_values.max() - all_values.min())
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Multiplex Learning Analysis Results', fontsize=16, fontweight='bold')
    
    # 1. Bar plot with error bars
    ax1 = axes[0, 0]
    sample_sizes = data.groupby('genotype').size()
    sns.barplot(data=data, x='genotype', y='learned_index', hue='genotype', 
                palette="deep", errorbar="se", estimator="mean", capsize=0.1, ax=ax1)
    ax1.set_title('Mean Learned Index by Genotype')
    ax1.set_ylabel('Learned Index (%)')
    ax1.set_ylim(y_min, y_max)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add sample sizes to legend
    legend_labels = [f"{genotype}: n={count}" for genotype, count in sample_sizes.items()]
    ax1.legend(title="Sample Sizes", labels=legend_labels, loc='upper right')
    
    # 2. Box plot
    ax2 = axes[0, 1]
    sns.boxplot(data=data, x='genotype', y='learned_index', hue='genotype', palette="deep", ax=ax2)
    ax2.set_title('Distribution of Learned Index by Genotype')
    ax2.set_ylabel('Learned Index (%)')
    ax2.set_ylim(y_min, y_max)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend().remove()
    
    # 3. Swarm plot
    ax3 = axes[1, 0]
    sns.swarmplot(data=data, x='genotype', y='learned_index', hue='genotype', palette="deep", ax=ax3)
    ax3.set_title('Individual Fly Learned Index by Genotype')
    ax3.set_ylabel('Learned Index (%)')
    ax3.set_ylim(y_min, y_max)
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend().remove()
    
    # 4. CS+ configuration plot
    ax4 = axes[1, 1]
    cs_config = data.groupby(['genotype', 'cs_plus_odor', 'sides_switched']).size().reset_index(name='count')
    cs_pivot = cs_config.pivot_table(index='genotype', columns=['cs_plus_odor', 'sides_switched'], 
                                    values='count', fill_value=0)
    cs_pivot.plot(kind='bar', stacked=True, ax=ax4, colormap='Set3')
    ax4.set_title('CS+ Configuration by Genotype')
    ax4.set_ylabel('Number of Flies')
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend(title='CS+ Odor & Side Switch', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save combined plot
    combined_path = os.path.join(output_folder, 'combined_analysis_plots.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual plots
    plot_types = [
        ('bar_plot', axes[0, 0]),
        ('box_plot', axes[0, 1]),
        ('swarm_plot', axes[1, 0]),
        ('cs_config_plot', axes[1, 1])
    ]
    
    for plot_name, ax in plot_types:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if plot_name == 'bar_plot':
            sns.barplot(data=data, x='genotype', y='learned_index', hue='genotype', 
                       palette="deep", errorbar="se", estimator="mean", capsize=0.1, ax=ax)
            ax.set_title('Mean Learned Index by Genotype')
        elif plot_name == 'box_plot':
            sns.boxplot(data=data, x='genotype', y='learned_index', hue='genotype', palette="deep", ax=ax)
            ax.set_title('Distribution of Learned Index by Genotype')
        elif plot_name == 'swarm_plot':
            sns.swarmplot(data=data, x='genotype', y='learned_index', hue='genotype', palette="deep", ax=ax)
            ax.set_title('Individual Fly Learned Index by Genotype')
        elif plot_name == 'cs_config_plot':
            cs_config = data.groupby(['genotype', 'cs_plus_odor', 'sides_switched']).size().reset_index(name='count')
            cs_pivot = cs_config.pivot_table(index='genotype', columns=['cs_plus_odor', 'sides_switched'], 
                                           values='count', fill_value=0)
            cs_pivot.plot(kind='bar', stacked=True, ax=ax, colormap='Set3')
            ax.set_title('CS+ Configuration by Genotype')
            ax.legend(title='CS+ Odor & Side Switch', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax.set_ylabel('Learned Index (%)')
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis='x', rotation=45)
        
        if plot_name != 'cs_config_plot':
            ax.legend().remove()
        
        plt.tight_layout()
        
        # Save individual plot
        individual_path = os.path.join(output_folder, f'{plot_name}.png')
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to {output_folder}")


def save_results_to_csv(folder_path, data, stats_results):
    """
    Save analysis results to CSV files
    """
    output_folder = os.path.join(folder_path, 'output')
    os.makedirs(output_folder, exist_ok=True)
    
    # Save raw data
    data_path = os.path.join(output_folder, 'experiment_data_cleaned.csv')
    data.to_csv(data_path, index=False)
    print(f"Raw data saved to: {data_path}")
    
    # Save statistical results
    if not stats_results.empty:
        stats_path = os.path.join(output_folder, 'statistical_results.csv')
        stats_results.to_csv(stats_path, index=False)
        print(f"Statistical results saved to: {stats_path}")


def analyze_experiment_folder(folder_path, config_path="experiment_config.json", 
                            threshold=4, midline_borders=0.6, filter_phase='both'):
    """
    Main function to analyze an entire experiment folder
    
    Parameters:
    -----------
    folder_path : str
        Path to the experiment folder
    config_path : str
        Path to the configuration file
    threshold : int
        Minimum choices required for valid fly
    midline_borders : float
        Midline border threshold (0.0 to 1.0)
    filter_phase : str
        Which phases to filter ('both', 'initial', 'test', or 'none')
    """
    print("="*80)
    print("MULTIPLEX BATCH ANALYSIS")
    print("="*80)
    
    # Load configuration
    config = load_experiment_config(config_path)
    
    # Override config settings with provided parameters
    config['analysis_settings']['threshold'] = threshold
    config['analysis_settings']['midline_borders'] = midline_borders
    config['analysis_settings']['filter_phase'] = filter_phase
    
    # Collect data from all trials
    print(f"\nCollecting data from: {folder_path}")
    print(f"Filtering parameters: threshold={threshold}, midline_borders={midline_borders}, filter_phase='{filter_phase}'")
    data = collect_trial_data(folder_path, threshold=threshold, midline_borders=midline_borders, filter_phase=filter_phase)
    
    if data.empty:
        print("No data collected. Exiting.")
        return
    
    # Perform statistical analysis
    stats_results = perform_statistical_analysis(data, config)
    
    # Create plots
    output_folder = os.path.join(folder_path, 'output', 'plots')
    create_plots(data, stats_results, output_folder)
    
    # Save results
    save_results_to_csv(folder_path, data, stats_results)
    
    print(f"\nAnalysis complete! Results saved to {os.path.join(folder_path, 'output')}")


# Example usage
if __name__ == "__main__":
    # =============================================================================
    # ANALYSIS PARAMETERS - Modify these as needed
    # =============================================================================
    ANALYSIS_PARAMS = {
        'folder_path': r"D:\multiplex\raw_files\serotonin_shibire\csdn",
        'config_path': "experiment_config.json",
        'threshold': 0,                    # Minimum choices required for valid fly
        'midline_borders': 0.6,           # Midline border threshold (0.0 to 1.0)
        'filter_phase': 'both'            # Which phases to filter: 'both', 'initial', 'test', or 'none'
    }
    
    # =============================================================================
    # RUN ANALYSIS
    # =============================================================================
    analyze_experiment_folder(
        folder_path=ANALYSIS_PARAMS['folder_path'],
        config_path=ANALYSIS_PARAMS['config_path'],
        threshold=ANALYSIS_PARAMS['threshold'],
        midline_borders=ANALYSIS_PARAMS['midline_borders'],
        filter_phase=ANALYSIS_PARAMS['filter_phase']
    )
