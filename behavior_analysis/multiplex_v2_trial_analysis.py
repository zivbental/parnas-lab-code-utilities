import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os


class FolderAnalysis:
    """
    This class is meant to collect all trials in a given folder and perform analysis on all of them.
    It will use the other class (MultiplexTrial), through which the various manipulations of the data will be performed.

    Therefore, the goal of this class is to organize all the files in a given folder and present statistics,
    figures, and data tables for the analysis of multiple trials.
    """
    def __init__(self) -> None:
        pass

    def create_datatables(self):
        # Placeholder for future implementation to generate summary data tables
        pass

    def create_figures(self):
        # Placeholder for future implementation to generate visualizations
        pass


class MultiplexTrial:
    def __init__(self) -> None:
        # Initialize attributes for raw and processed data
        self.raw_data = None
        self.processed_data = None  # May be filtered during analysis
        self.processed_data_full = None  # Always holds full data for the trial
        self.filtered_flies = []  # List of fly columns that pass filtering criteria
        self.valence_df = None
        self.test_df = None
        self.mm_per_unit = 0.225  # mm per coordinate unit

    def load_data(self, data_path):
        """
        Loads a Multiplex trial CSV file into the object.
        Stores both the full dataset and a copy to be modified later.
        """
        self.data_path = data_path
        self.processed_data_full = pd.read_csv(data_path)
        self.processed_data = self.processed_data_full.copy()

    def set_analysis_stages(self, initial_values: str, changed_values: str):
        """
        Sets self.valence_df and self.test_df based on the provided stage names.
        """
        def extract_stage_df(stage_name):
            df = self.processed_data
            stage_df = df[df['experiment_step'] == stage_name]
            stage_df = stage_df.filter(regex=r'^(Timestamp|chamber_\d+_loc)$')
            stage_df.set_index('Timestamp', inplace=True)
            return stage_df

        self.valence_df = extract_stage_df(initial_values)
        self.test_df = extract_stage_df(changed_values)


    def filter_by_num_choices(self, midline_borders, threshold=1, filter='both'):
        """
        Filters flies based on the number of times they cross the midline during selected stages.
        Must call set_analysis_stages() before this function.
        """
        if self.valence_df is None or self.test_df is None:
            raise ValueError("Please set analysis stages using set_analysis_stages() before filtering.")

        df_mapping = {
            'both': [('valence_df', self.valence_df), ('test_df', self.test_df)],
            'test': [('test_df', self.test_df)],
            'valence': [('valence_df', self.valence_df)]
        }

        filtered_dfs = {}

        for key, df in df_mapping.get(filter, []):
            filtered_dfs[f"filtered_{key}"] = self.filter_by_midline(df, midline_borders=midline_borders, threshold=threshold)

        if filter == 'both':
            common_columns = filtered_dfs['filtered_valence_df'].columns.intersection(filtered_dfs['filtered_test_df'].columns)
            self.processed_data = filtered_dfs['filtered_valence_df'][common_columns], filtered_dfs['filtered_test_df'][common_columns]
        elif filter in ['test', 'valence']:
            key = f"filtered_{filter}_df"
            self.processed_data = filtered_dfs[key][filtered_dfs[key].columns]


    def filter_by_midline(self, df, midline_borders, threshold=1):
        """
        Helper function that counts midline crossings for each fly and filters based on a threshold.
        """
        crossing_counts = {}
        for col in df.columns:
            values = df[col]
            crossings = (
                ((values.shift(1) < midline_borders) & (values >= midline_borders)) | 
                ((values.shift(1) > midline_borders) & (values <= midline_borders)) |
                ((values.shift(1) > -midline_borders) & (values <= -midline_borders)) |
                ((values.shift(1) < -midline_borders) & (values >= -midline_borders))
            )
            crossing_counts[col] = crossings.sum()
        filtered_columns = [col for col, count in crossing_counts.items() if count >= threshold]
        filtered_df = df[filtered_columns]
        return filtered_df

    @staticmethod
    def time_spent(df, width_size=10, sampling_rate=0.1):
        """
        Determines how much time each fly spent on each side of the chamber.
        Assumes positions beyond width_size or -width_size define side occupancy.
        """
        def process_counts(counts):
            df_transposed = counts.reset_index().T
            df_transposed.columns = df_transposed.iloc[0]
            return df_transposed.drop(df_transposed.index[0])

        mask_greater = df > width_size
        mask_less = df < -width_size

        count_greater = mask_greater.sum() * sampling_rate
        count_less = mask_less.sum() * sampling_rate

        count_greater_processed = process_counts(count_greater)
        count_less_processed = process_counts(count_less)

        df_combined = pd.concat([count_greater_processed, count_less_processed])
        df_combined.index = ['right_side', 'left_side']

        return df_combined

    def analyse_time(self, width_size=10):
        """
        Compares side preference during two selected stages (defined earlier via set_analysis_stages),
        and saves results to CSV files in the same folder as the raw data.

        The learned index is calculated as the change in side preference (in %) between the two stages.
        Filters flies based on activity and stores those passing in self.filtered_flies.
        """
        import os

        # Step 1: Calculate time spent on each side (left and right)
        valence_df = self.time_spent(self.processed_data[0], width_size=width_size)
        test_df = self.time_spent(self.processed_data[1], width_size=width_size)

        # Step 2: Filter out flies with no time recorded in either phase
        valence_denominator = valence_df.iloc[1] + valence_df.iloc[0]
        test_denominator = test_df.iloc[0] + test_df.iloc[1]

        combined_mask = (valence_denominator != 0) & (test_denominator != 0)
        filtered_valence_df = valence_df.loc[:, combined_mask]
        filtered_test_df = test_df.loc[:, combined_mask]

        # Step 3: Require flies to be active for at least 30s during the initial stage
        initial_val_filter = filtered_valence_df.iloc[1] >= 30
        filtered_valence_df = filtered_valence_df.loc[:, initial_val_filter]
        filtered_test_df = filtered_test_df.loc[:, initial_val_filter]

        # Step 4: Calculate preferences and learned index
        initial_val = (filtered_valence_df.iloc[1]) / (filtered_valence_df.iloc[0] + filtered_valence_df.iloc[1])
        end_valence = (filtered_test_df.iloc[0]) / (filtered_test_df.iloc[0] + filtered_test_df.iloc[1])
        learned_index = (end_valence - initial_val) * 100

        # Step 5: Store fly IDs
        self.filtered_flies = learned_index.index.to_list()

        # Step 6: Create results DataFrame
        results_df = pd.DataFrame({
            'Fly ID': learned_index.index,
            'Initial Preference (left)': initial_val.values * 100,
            'Final Preference (right)': end_valence.values * 100,
            'Learned Index (%)': learned_index.values
        })

        mean_index_df = pd.DataFrame({
            'Metric': ['Mean Learned Index'],
            'Value': [learned_index.mean()]
        })

        # Step 7: Save to CSV if data_path is defined
        if hasattr(self, 'data_path'):
            folder_path = os.path.dirname(self.data_path)
            results_path = os.path.join(folder_path, "learned_index_by_fly.csv")
            mean_path = os.path.join(folder_path, "learned_index_summary.csv")

            results_df.to_csv(results_path, index=False)
            mean_index_df.to_csv(mean_path, index=False)

            print(f"Learned index data saved to:\n- {results_path}\n- {mean_path}")
        else:
            print("Warning: data_path not set. Results not saved.")

        

    def split_by_stage(self) -> dict[str, pd.DataFrame]:
        """
        Return dict{stage: position_df} limited to *filtered_flies*.
        This version also inserts:
        - 'Habituation Stage' before the first labeled stage
        - 'Rest X' periods between consecutive stages
        """
        stage_dict: dict[str, pd.DataFrame] = {}

        # Group data by experimental steps (phases)
        grouped = self.processed_data_full.groupby("experiment_step", sort=False)
        steps = list(grouped)

        # Extract the full Timestamp column for reference
        full_time = self.processed_data_full["Timestamp"]
        rest_counter = 1

        # Add Habituation Stage (everything before the first stage)
        first_start_time = steps[0][1]["Timestamp"].iloc[0]
        habituation_df = self.processed_data_full[full_time < first_start_time][self.filtered_flies].copy()
        if not habituation_df.empty:
            habituation_df.index = self.processed_data_full.loc[habituation_df.index, "Timestamp"]
            stage_dict["Habituation Stage"] = habituation_df

        # Add actual stages and intermediate rest periods
        for i, (stage_name, df) in enumerate(steps):
            # Current stage
            stage_data = df[self.filtered_flies].copy()
            stage_data.index = df["Timestamp"].values
            stage_dict[stage_name] = stage_data

            # Add rest period between this stage and the next
            if i < len(steps) - 1:
                current_end = df["Timestamp"].iloc[-1]
                next_start = steps[i + 1][1]["Timestamp"].iloc[0]

                rest_df = self.processed_data_full[
                    (full_time > current_end) & (full_time < next_start)
                ][self.filtered_flies].copy()

                if not rest_df.empty:
                    rest_df.index = self.processed_data_full.loc[rest_df.index, "Timestamp"]
                    stage_dict[f"Rest {rest_counter}"] = rest_df
                    rest_counter += 1

        return stage_dict

    def calculate_speed(self, df, sampling_rate=0.1):
        """
        Computes instantaneous speed (abs(dx)/dt) per fly based on position trace,
        then converts it to mm/s using chamber scale.
        """
        speed_df = df.diff().abs() / sampling_rate  # raw speed in units/s
        speed_df *= self.mm_per_unit  # convert to mm/s
        return speed_df


    def summarize_speed(self, speed_df):
        """
        Computes the average and SEM (standard error of mean) fly speed for a given stage.
        """
        mean_per_fly = speed_df.mean()
        mean_speed = mean_per_fly.mean()
        sem_speed = mean_per_fly.sem()
        return {'mean_speed': mean_speed, 'sem_speed': sem_speed}

    def analyze_and_plot_speed(self, *, sampling_rate: float = 0.1) -> pd.DataFrame:
        """
        High-level wrapper – computes & plots average speed per stage.
        Also saves per-fly speed data and the summary bar plot to disk.
        """
        import seaborn as sns
        import os

        # 1. Compute per-stage positions and speeds
        stage_pos = self.split_by_stage()
        stage_speed = {
            stage: self.calculate_speed(df, sampling_rate=sampling_rate)
            for stage, df in stage_pos.items()
        }

        # 2. Summary statistics for plotting
        summary = {
            stage: self.summarize_speed(df)
            for stage, df in stage_speed.items()
        }

        summary_df = pd.DataFrame(summary).T.reset_index()
        summary_df.columns = ["Stage", "Mean Speed", "SEM"]

        # 3. Also export the individual fly means per stage
        fly_speed_df = pd.DataFrame({
            stage: df.mean()  # mean per fly per stage
            for stage, df in stage_speed.items()
        })

        # 4. Plot the summary
        plt.figure(figsize=(10, 5))
        sns.set(style="whitegrid")

        bar_color = sns.color_palette("muted")[0]

        plt.bar(
            summary_df["Stage"],
            summary_df["Mean Speed"],
            yerr=summary_df["SEM"],
            capsize=5,
            color=bar_color,
            edgecolor='black'
        )

        plt.ylabel("Mean Speed (mm/s)")
        plt.xlabel("Experiment Stage")
        plt.title("Average Fly Speed per Stage")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # 5. Save figure and raw data
        if hasattr(self, 'data_path'):
            folder_path = os.path.dirname(self.data_path)

            # Save plot
            fig_path = os.path.join(folder_path, "speed_summary.png")
            plt.savefig(fig_path, dpi=300)
            print(f"Figure saved to: {fig_path}")

            # Save data to Excel
            # Save data to CSV
            fly_speed_csv_path = os.path.join(folder_path, "fly_speeds_by_fly.csv")
            summary_csv_path = os.path.join(folder_path, "fly_speed_summary.csv")

            fly_speed_df.to_csv(fly_speed_csv_path)
            summary_df.to_csv(summary_csv_path, index=False)

            print(f"Fly speed data saved to: {fly_speed_csv_path}")
            print(f"Summary speed data saved to: {summary_csv_path}")


        else:
            print("Warning: data_path not set. Outputs not saved.")

        plt.close()
        return summary_df

    def filter_specific_fly(self):
        # Placeholder for manually selecting or excluding specific flies in the future
        pass

    def plot_trial(self):
        """
        Visualizes fly position traces during a filtered condition (e.g., odor-on test phase).
        """
        df = self.processed_data
        test_df = df[(df['LEFTODOR2'] == 1) & (df['RIGHTODOR1'] == 1)]
        test_df_location_only = test_df.filter(regex='^(Timestamp|cX\d{3})$')
        test_df_location_only.replace(0, np.nan, inplace=True)
        test_df_location_only.set_index('Time', inplace=True)
        cX_columns = [col for col in test_df_location_only.columns if col.startswith('cX')]

        fig, axes = plt.subplots(len(cX_columns), 1, figsize=(10, 0.5 * len(cX_columns)), sharex=True)
        if len(cX_columns) == 1:
            axes = [axes]

        for i, col in enumerate(cX_columns):
            sns.lineplot(ax=axes[i], x=test_df_location_only.index, y=test_df_location_only[col])
            axes[i].set_ylim(-1, 1)
            axes[i].set_ylabel('')
            axes[i].set_yticks([-1, 0, 1])
            axes[i].annotate(col, xy=(0, 0.5), xytext=(-axes[i].yaxis.labelpad - 10, 0),
                             xycoords=axes[i].yaxis.label, textcoords='offset points',
                             size='large', ha='right', va='center', rotation=0)
            axes[i].grid(True)

        axes[-1].set_xlabel('Time')
        plt.tight_layout()
        plt.show()


"""
Main segment of code that runs the functions
"""

"""
Define the location of the file that you want to analyze.

The file should be a CSV file containing the fly location tracking data and 
experiment step annotations. Make sure it includes at least the following columns:

- 'Timestamp': records the time of each sample (should be in ascending order).
- 'experiment_step': indicates the behavioral stage, e.g., "Initial Valence", "Test".
- 'chamber_X_loc' columns: represent the position of a fly in chamber X (e.g., 'chamber_1_loc').
"""
file_path = "/home/ziv-lab/codes/parnas_codes/behavior_analysis/test_trial/fly_loc.csv"

"""
Create a MultiplexTrial object.

This object encapsulates all data and methods related to analyzing a single behavioral trial.
It provides functions for data loading, filtering, selecting behavioral stages, and visualizing results.
"""
trial_1 = MultiplexTrial()

"""
Load the data of a specific trial from the CSV file.

This reads the file and stores the full dataset in memory, making it available for 
stage selection and analysis functions.
"""
trial_1.load_data(file_path)

"""
Define which two experimental stages to compare in your analysis.

This replaces hardcoded stage selection and allows dynamic comparison of any two stages.
Arguments:
- initial_values: the baseline stage (e.g., "Initial Valence").
- changed_values: the post-manipulation stage (e.g., "Test").

These must match the 'experiment_step' values in your CSV file exactly.
"""
trial_1.set_analysis_stages(initial_values="Initial Valence", changed_values="Test")

"""
Filter flies based on their activity (crossing the midline) during the selected stages.

Arguments:
- midline_borders: defines the central boundary of the chamber. A crossing is counted
  when a fly moves from one side of this boundary to the other (positive to negative or vice versa).
- threshold: the minimum number of crossings required to include a fly in the analysis.
- filter:
    - 'both': apply the threshold to both stages.
    - 'valence': apply the threshold only to the initial stage.
    - 'test': apply the threshold only to the changed stage.

Only flies meeting the activity criteria are retained for further analysis.
"""
trial_1.filter_by_num_choices(midline_borders=0.6, threshold=4, filter='both')

"""
Analyze behavior change between the selected stages.

This calculates how much time each fly spent on the left vs. right side of the chamber
during both stages. It then computes a 'learned index', which is the change in side preference
(in percent) from the initial to the changed stage.

Only flies that passed the filtering step are included in this computation.
"""
trial_1.analyse_time(width_size=10)

"""
Calculate and plot average fly speed during all experiment stages.

This function:
- Splits the trial into stages and rest periods based on 'experiment_step' labels.
- Calculates instantaneous speed for each fly (in mm/s), using position changes over time.
- Summarizes the mean and SEM (standard error of the mean) speed per stage.
- Plots a bar graph of the speed summary and saves it to the same folder as the CSV file.

You can customize the sampling rate (in seconds) by passing `sampling_rate=X` as an argument.
"""
trial_1.analyze_and_plot_speed()
