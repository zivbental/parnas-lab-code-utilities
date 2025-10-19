import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import json


class MultiplexTrial:
    """
    Core class for analyzing individual multiplex trials with automatic CS+ detection
    and side-switching support.
    """
    
    def __init__(self) -> None:
        self.raw_data = None
        self.processed_data = None
        self.file_path = None

    def load_data(self, data_path):
        """
        Load a Multiplex log file (.csv) and convert it to pandas dataframe
        """
        self.file_path = data_path
        self.raw_data = pd.read_csv(data_path)
        self.processed_data = self.raw_data.copy()

    def select_test_period(self):
        """
        Return the period that corresponds to the 'Test' phase of the trial.
        """
        df = self.processed_data
        test_df = df[df['experiment_step'] == 'Test']
        test_df = test_df.filter(regex=r'^(Timestamp|chamber_\d+_loc)$')
        test_df.set_index('Timestamp', inplace=True)
        return test_df

    def select_initial_valence_period(self):
        """
        Return the period that corresponds to the 'Initial Valence' phase of the trial.
        """
        df = self.processed_data
        initial_valence_df = df[df['experiment_step'] == 'Initial Valence']
        initial_valence_df = initial_valence_df.filter(regex=r'^(Timestamp|chamber_\d+_loc)$')
        initial_valence_df.set_index('Timestamp', inplace=True)
        return initial_valence_df

    def filter_by_num_choices(self, midline_borders, threshold=1, filter='both'):
        """
        Filter flies based on the number of times they have crossed the midline.
        Values indicate how well the fly has explored the chamber during the initial valence/test period.
        """
        valence_df = self.select_initial_valence_period()
        test_df = self.select_test_period()

        df_mapping = {
            'both': [('valence_df', valence_df), ('test_df', test_df)],
            'test': [('test_df', test_df)],
            'valence': [('valence_df', valence_df)]
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
        Filter out flies that have not crossed the midline threshold number of times.
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
        Calculate time spent on each side of the chamber.
        Returns a dataframe showing for each fly the time spent on each side.
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

    def identify_cs_plus_odor(self):
        """
        Identify CS+ odor based on conditioning type and experimental rules:
        - Operant: Choice between MOIL and another odor → the other odor is CS+
        - Classical: Odor paired with electrical shock → that odor is CS+
        Returns: ('odor_name', 'side') indicating which odor and side is CS+
        """
        learning_shock_df = self.raw_data[self.raw_data['experiment_step'] == 'Learning Shock']
        
        if learning_shock_df.empty:
            print("Warning: No Learning Shock phase found in data")
            return ('mch', 'left')
        
        # Get protocol type from metadata
        metadata_path = os.path.join(os.path.dirname(self.file_path), 'experiment_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            protocol = metadata.get('protocol', '').lower()
        else:
            print("Warning: No metadata file found, defaulting to operant conditioning")
            protocol = 'operant'  # Default assumption
        
        # Get odor status during Learning Shock stage
        odor_columns = ['mch_right_status', 'oct_right_status', 'moil_right_status', 
                       'mch_left_status', 'oct_left_status', 'moil_left_status']
        odor_data = learning_shock_df[odor_columns]
        
        mch_left = odor_data['mch_left_status'].iloc[0]
        mch_right = odor_data['mch_right_status'].iloc[0]
        moil_left = odor_data['moil_left_status'].iloc[0]
        moil_right = odor_data['moil_right_status'].iloc[0]
        oct_left = odor_data['oct_left_status'].iloc[0]
        oct_right = odor_data['oct_right_status'].iloc[0]
        
        # Determine CS+ based on conditioning type
        if 'operant' in protocol:
            # Operant: Choice between MOIL and another odor → the other odor is CS+
            if (moil_left == 1 or moil_right == 1):
                if (mch_left == 1 or mch_right == 1):
                    cs_plus_odor = 'mch'
                    cs_plus_side = 'left' if mch_left == 1 else 'right'
                elif (oct_left == 1 or oct_right == 1):
                    cs_plus_odor = 'oct'
                    cs_plus_side = 'left' if oct_left == 1 else 'right'
                else:
                    print("Warning: MOIL present but no other odor found")
                    return ('mch', 'left')
            else:
                print("Warning: Operant conditioning but no MOIL found")
                return ('mch', 'left')
        
        elif 'classical' in protocol:
            # Classical: Find which odor is active during shock stage
            # Check all active odors and determine which one is CS+
            active_odors = []
            if mch_left == 1: active_odors.append(('mch', 'left'))
            if mch_right == 1: active_odors.append(('mch', 'right'))
            if oct_left == 1: active_odors.append(('oct', 'left'))
            if oct_right == 1: active_odors.append(('oct', 'right'))
            if moil_left == 1: active_odors.append(('moil', 'left'))
            if moil_right == 1: active_odors.append(('moil', 'right'))
            
            if len(active_odors) == 1:
                cs_plus_odor, cs_plus_side = active_odors[0]
            else:
                print("Warning: Multiple odors active during classical conditioning")
                return ('mch', 'left')
        
        else:
            print("Warning: Unknown protocol type, defaulting to operant logic")
            # Default to operant logic
            if (moil_left == 1 or moil_right == 1) and (mch_left == 1 or mch_right == 1):
                cs_plus_odor = 'mch'
                cs_plus_side = 'left' if mch_left == 1 else 'right'
            else:
                return ('mch', 'left')
        
        print(f"CS+ identified as {cs_plus_odor.upper()} on {cs_plus_side.upper()} side (protocol: {protocol})")
        return (cs_plus_odor, cs_plus_side)

    def detect_cs_plus_side_in_phase(self, phase_data, cs_plus_odor):
        """
        Detect which side the CS+ odor appears on in a specific phase
        Returns: 'left' or 'right'
        """
        odor_columns = ['mch_right_status', 'oct_right_status', 'moil_right_status', 
                       'mch_left_status', 'oct_left_status', 'moil_left_status']
        
        if not all(col in phase_data.columns for col in odor_columns):
            print(f"Warning: Odor status columns not found in phase data")
            return 'left'  # Default fallback
        
        odor_data = phase_data[odor_columns]
        
        # Check which side has the CS+ odor active
        left_odor_col = f'{cs_plus_odor}_left_status'
        right_odor_col = f'{cs_plus_odor}_right_status'
        
        if left_odor_col in odor_data.columns and right_odor_col in odor_data.columns:
            left_status = odor_data[left_odor_col].iloc[0] if not odor_data.empty else 0
            right_status = odor_data[right_odor_col].iloc[0] if not odor_data.empty else 0
            
            if left_status == 1 and right_status == 0:
                return 'left'
            elif right_status == 1 and left_status == 0:
                return 'right'
            elif left_status == 1 and right_status == 1:
                print(f"Warning: {cs_plus_odor} active on both sides, defaulting to left")
                return 'left'
            else:
                print(f"Warning: {cs_plus_odor} not active in this phase, defaulting to left")
                return 'left'
        else:
            print(f"Warning: {cs_plus_odor} status columns not found, defaulting to left")
            return 'left'

    def analyse_time(self):
        """
        Calculate learned index with automatic CS+/CS- detection and side-switching handling
        """
        # 1. Identify CS+ odor and side from Learning Shock phase
        cs_plus_odor, learning_cs_side = self.identify_cs_plus_odor()
        
        # 2. Get phase data from raw_data (before filtering)
        valence_phase_data = self.raw_data[self.raw_data['experiment_step'] == 'Initial Valence']
        test_phase_data = self.raw_data[self.raw_data['experiment_step'] == 'Test']
        
        # 3. Detect CS+ side in each phase
        valence_cs_side = self.detect_cs_plus_side_in_phase(valence_phase_data, cs_plus_odor)
        test_cs_side = self.detect_cs_plus_side_in_phase(test_phase_data, cs_plus_odor)
        
        # 4. Check if sides switched
        sides_switched = (valence_cs_side != test_cs_side)
        print(f"CS+ side in Initial Valence: {valence_cs_side.upper()}")
        print(f"CS+ side in Test: {test_cs_side.upper()}")
        print(f"Sides switched: {sides_switched}")
        
        # 5. Calculate time spent for both phases using filtered data
        valence_df = self.time_spent(self.processed_data[0])
        test_df = self.time_spent(self.processed_data[1])

        # 6. Apply existing filtering logic
        valence_denominator = valence_df.iloc[1] + valence_df.iloc[0]
        test_denominator = test_df.iloc[0] + test_df.iloc[1]
        combined_mask = (valence_denominator != 0) & (test_denominator != 0)

        filtered_valence_df = valence_df.loc[:, combined_mask]
        filtered_test_df = test_df.loc[:, combined_mask]

        # Filter out flies with < 30 seconds in initial valence
        initial_val_filter = filtered_valence_df.iloc[1] >= 30
        filtered_valence_df = filtered_valence_df.loc[:, initial_val_filter]
        filtered_test_df = filtered_test_df.loc[:, initial_val_filter]

        # 7. Calculate CS+ preference for each phase (adaptive to actual side)
        if valence_cs_side == 'left':
            initial_val = (filtered_valence_df.iloc[1]) / (filtered_valence_df.iloc[0] + filtered_valence_df.iloc[1])  # left/total
        else:
            initial_val = (filtered_valence_df.iloc[0]) / (filtered_valence_df.iloc[0] + filtered_valence_df.iloc[1])  # right/total
            
        if test_cs_side == 'left':
            end_valence = (filtered_test_df.iloc[1]) / (filtered_test_df.iloc[0] + filtered_test_df.iloc[1])  # left/total
        else:
            end_valence = (filtered_test_df.iloc[0]) / (filtered_test_df.iloc[0] + filtered_test_df.iloc[1])  # right/total

        # 8. Calculate learned index
        learned_index = (end_valence - initial_val) * 100
        
        # 9. Create enhanced results dataframe
        results_data = []
        for i, col in enumerate(filtered_valence_df.columns):
            fly_id = col.replace('chamber_', '').replace('_loc', '')
            results_data.append({
                'fly_id': fly_id,
                'cs_plus_odor': cs_plus_odor,
                'initial_valence_cs_side': valence_cs_side,
                'test_cs_side': test_cs_side,
                'sides_switched': sides_switched,
                'initial_valence_cs_preference': initial_val.iloc[i] * 100,  # Convert to percentage
                'test_cs_preference': end_valence.iloc[i] * 100,  # Convert to percentage
                'learned_index': learned_index.iloc[i],
                'valid_fly': True
            })
        
        results_df = pd.DataFrame(results_data)
        
        # 10. Print summary
        print(f"\nCS+ Odor: {cs_plus_odor.upper()}")
        print(f"Number of valid flies: {len(results_df)}")
        print(f"Mean learned index: {learned_index.mean():.2f}")
        print(f"Std learned index: {learned_index.std():.2f}")
        
        # 11. Save to CSV
        self.save_analysis_results(results_df)
        
        return results_df

    def save_analysis_results(self, results_df):
        """
        Save analysis results to CSV file in analysis folder
        """
        # Create analysis folder
        analysis_folder = os.path.join(os.path.dirname(self.file_path), 'analysis')
        os.makedirs(analysis_folder, exist_ok=True)
        
        # Create filename with timestamp to avoid permission issues
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{base_name}_learned_index_analysis_{timestamp}.csv"
        csv_path = os.path.join(analysis_folder, csv_filename)
        
        # Save CSV
        results_df.to_csv(csv_path, index=False)
        print(f"Analysis results saved to: {csv_path}")

    def plot_trial(self):
        """
        Plot individual fly trajectories during test period
        """
        df = self.processed_data

        # Select only the test period (may vary if I will use other protocols)
        test_df = df[(df['LEFTODOR2'] == 1) & (df['RIGHTODOR1'] == 1)]

        # Extract time and location columns of flies
        test_df_location_only = test_df.filter(regex=r'^(Timestamp|cX\d{3})$')

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
