import sys
import os
import logging

# Add the main application directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import shock_all_left, wait

def custom_logic():
    logging.info('started experiment')
    # Apply shock to all chambers on the left side for 10 seconds
    logging.info('shocking left side for 10 seconds')
    shock_all_left(10)
    # Wait for 10 seconds
    logging.info('waiting for 10 seconds')
    wait(10)
    # Apply shock to all chambers on the left side for 5 seconds
    logging.info('shocking left side for 5 seconds')
    shock_all_left(5)
    logging.info('finished')

# Run the custom logic
custom_logic()
