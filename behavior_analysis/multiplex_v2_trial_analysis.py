"""
Simple analysis script for individual multiplex trials.
Uses the shared core module for all analysis functionality.
"""

from multiplex_core import MultiplexTrial

# Example usage
if __name__ == "__main__":
    # Load a single Trial to the object
    file_path = "fly_loc.csv"
    trial_1 = MultiplexTrial()
    trial_1.load_data(file_path)

    trial_1.filter_by_num_choices(midline_borders=0.6, threshold=4, filter='both')

    # Use the enhanced analysis method with side-switching detection
    print("Running enhanced analysis with automatic CS+/CS- detection and side-switching handling...")
    results = trial_1.analyse_time()

    print("\nIndividual fly results:")
    print(results)