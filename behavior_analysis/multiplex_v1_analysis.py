import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os



class FolderAnalysis:
    """
    This class is meant to collect all trials in a given folder and perform analysis on all of them.
    It will use the other class (MultiplexTrial), through which the various manipulations of the data will be performed.

    Therefore, the goal of this class is to orgnaize all the files in a given folder and present statistics,
    figures and data tables for the analysis of multiple trials.
    """
    def __init__(self) -> None:
        pass

    def create_datatables(self):
        pass

    def create_figures(self):
        pass

class MultiplexTrial:
    def __init__(self) -> None:
        raw_data = None
        processed_data = None

        pass

    def load_data(self, data_path):
        """
        This function loads a Multiplex log file (.txt) and coverts it to pandas dataframe
        """
        # This three lines of code allow for clean data to be loaded to a dataframe
        temp_df = pd.read_csv(data_path, sep=r'\s+')
        temp_df.to_csv('temp.csv')
        df = pd.read_csv('temp.csv', skiprows=1)
        # Store the corrected dataframe into the data attribute of MultiplexTrial object
        self.raw_data = df

    def filter_no_movement(self):
        """
        This function assigns NaN values where the location of the fly in a given time equals to 0.
        When the algorithm does not detect the fly, it assigns a value of 0, and this value will influence further analyses if we do not discard it.
        """
        # Select columns that match the cX### pattern and replace 0 with NaN
        movement_columns = self.raw_data.filter(regex='^(cX\d{3})$').replace(0, np.nan)
        
        # Maintain the other columns in the original DataFrame
        other_columns = self.raw_data.drop(columns=movement_columns.columns)
        
        # Combine the movement columns (with NaNs) and the other columns
        self.processed_data = pd.concat([movement_columns, other_columns], axis=1)

    def select_test_period(self):
        """
        This function returns the period that corresponds to the initial valence of the fly to the two odors
        """
        # Load the MultiplexTrial object data into a temporary dataframe that will be manipulated
        df = self.processed_data
        initial_valence_df = df[(df['LEFTODOR2'] == 1) & (df['RIGHTODOR1'] == 1)]
        initial_valence_df = initial_valence_df.filter(regex='^(Time|cX\d{3})$')
        initial_valence_df.set_index('Time', inplace=True)
        return initial_valence_df

    def select_inital_valence_period(self):
        """
        This function returns the period that corresponds to the test period of the trial.
        """
        df = self.processed_data
        test_df = df[(df['LEFTODOR1'] == 1) & (df['RIGHTODOR2'] == 1)]
        test_df = test_df.filter(regex='^(Time|cX\d{3})$')
        test_df.set_index('Time', inplace=True)
        return test_df

    def filter_by_num_choices(self, area_size, limit_range, threshold=1, filter='both'):
        """
        This function counts the number of times a fly has entered a center area.
        Values will indicate how well the fly has explored the chamber during the initial valence/test period.
        """
        valence_df = self.select_inital_valence_period()
        test_df = self.select_test_period()

        # Create a mapping of filters to DataFrames
        df_mapping = {
            'both': [('valence_df', valence_df), ('test_df', test_df)],
            'test': [('test_df', test_df)],
            'valence': [('valence_df', valence_df)]
        }

        # Dictionary to store the filtered DataFrames
        filtered_dfs = {}

        # Apply filtering based on the filter key, if it exists in the mapping
        for key, df in df_mapping.get(filter, []):
            filtered_dfs[f"filtered_{key}"] = self.filter_by_midline(df, area_size=area_size, limit_range=limit_range, threshold=threshold)

        # If 'both' is selected, find the common columns between the two filtered DataFrames
        if filter == 'both':
            common_columns = filtered_dfs['filtered_valence_df'].columns.intersection(filtered_dfs['filtered_test_df'].columns)
            self.processed_data = filtered_dfs['filtered_valence_df'][common_columns], filtered_dfs['filtered_test_df'][common_columns]

        # Otherwise, return the filtered DataFrame itself
        elif filter in ['test', 'valence']:
            key = f"filtered_{filter}_df"
            self.processed_data = filtered_dfs[key][filtered_dfs[key].columns]


    def filter_by_midline(self, df, area_size, limit_range, threshold = 1):
        # Define the area limits
        center_size = area_size
        right_limit = 0 + center_size/2
        left_limit = 0 - center_size/2
        # Create a dataframe that will highlight when a fly crosses the limits, thereby creating 'chunks'.
        crosses_df = df[(((df >= right_limit - limit_range) & (df <= right_limit + limit_range)) | ((df <= left_limit + limit_range) & (df >= left_limit - limit_range)))]
        
        """
        The following code segment will count how many times the fly has crossed the limit. Because we get a lot of values for each limit cross,
        We combine the values we get into 'chunks', and count the number of chunks, which indicates the number of crosses.
        """

        # Shift the DataFrame down by 1 position
        shifted_df = crosses_df.shift(1)

        # Compare the original DataFrame to the shifted one
        # Identify where new NaNs appear in the shifted DataFrame compared to the original
        chunk_borders = (crosses_df.notna() & shifted_df.isna())

        # Count the number of chunk borders for each column
        chunk_border_count = chunk_borders.sum()

        columns_to_keep = chunk_border_count[chunk_border_count >= threshold].index
        filtered_df = df[columns_to_keep]
        
        return filtered_df

    def time_spent(self, df=None, width_size=0.2):
        """
        This function will determine border values to decide which 'side' the fly is currently in.
        Then it will calculate the time period the fly spent on this side for a given dataframe.
        It will return a dataframe that shows for each fly the ratio of the time spent on the two sides of the chamber.
        """
        # Use self.processed_data if df is None
        if df is None:
            df = self.processed_data
        
            if x >= 0.2:
                return 1
            elif x < -0.2:
                return -1
            else:
                return 0

        # Apply the classification function to the entire DataFrame
        classified_df = df.applymap(classify_value)
        
        return classified_df


    def filter_compare_to_valence(self):
        pass

    def filter_specific_fly(self):
        pass

    
    def plot_trial(self):
        # Load the MultiplexTrial object data into a temporary dataframe that will be manipulated
        df = self.processed_data

        # Select only the test period (may vary if I will use other protocols)
        test_df = df[(df['LEFTODOR2'] == 1) & (df['RIGHTODOR1'] == 1)]

        # Extract time and location columns of flies
        test_df_location_only = test_df.filter(regex='^(Time|cX\d{3})$')

        # Replace 0 values with NaN, to not include during the calculation 0, which happens when the system is not detectiing the flies
        test_df_location_only.replace(0, np.nan, inplace=True)

        # Set the 'Time' column as the index
        test_df_location_only.set_index('Time', inplace=True)
        # Plot data
        cX_columns = [col for col in test_df_location_only.columns if col.startswith('cX')]

        # Set up the figure and axis for subplots
        fig, axes = plt.subplots(len(cX_columns), 1, figsize=(10, 0.5 * len(cX_columns)), sharex=True)

        # If there's only one subplot, wrap the axes in a list for consistency
        if len(cX_columns) == 1:
            axes = [axes]

        # Plot each cX column in its own subplot
        for i, col in enumerate(cX_columns):
            sns.lineplot(ax=axes[i], x=test_df_location_only.index, y=test_df_location_only[col])
            axes[i].set_ylim(-1, 1)  # Set Y-axis limits from -1 to 1
            axes[i].set_ylabel('')  # Remove the Y-axis label
            axes[i].set_yticks([-1, 0, 1])  # Optionally, set specific y-ticks
            
            # Place the column name on the left side
            axes[i].annotate(col, xy=(0, 0.5), xytext=(-axes[i].yaxis.labelpad - 10, 0),
                             xycoords=axes[i].yaxis.label, textcoords='offset points',
                             size='large', ha='right', va='center', rotation=0)

            axes[i].grid(True)

        # Set the common x-label
        axes[-1].set_xlabel('Time')

        # Adjust the layout to prevent overlap
        plt.tight_layout()

        # Display the plot
        plt.show()


"""
Main segment of code that runs the functions
"""
file_path = "/Users/zivbentulila/Library/CloudStorage/GoogleDrive-zivbental@gmail.com/My Drive/Work/MSc Neuroscience/Moshe Parnas/Experiments/Serotonergic system/5ht_behavior/operant_conditioning/raw_data/behavior/13.8.24/5ht_rnai/mb247/20240813_075645_Log.txt"
# file_path = "G:/My Drive/Work/MSc Neuroscience/Moshe Parnas/Experiments/Serotonergic system/5ht_behavior/operant_conditioning/raw_data/behavior/13.8.24/5ht_rnai/32471/20240813_083500_Log.txt"

# Create a MultiplexTrial object
trial_1 = MultiplexTrial()

# Load a single Trial to the object
trial_1.load_data(file_path)

trial_1.filter_no_movement()

trial_1.filter_by_num_choices(0.5, 0.3, 1)

print(trial_1.time_spent())


'''
Filtrations:
Master filter function to call various other functions:
- Filter by movement
- Filter by number of choises
- Filter out specific flies that I manually specify

Things I want to show:
Allow to select folder for analysis
Single fly trace (+ odor & shock timing)
All fly traces (+ odor & shock timing)
'''