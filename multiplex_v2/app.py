import sys
import os
# Add the parent directory of 'protocols' to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from protocols.ziv_protocols import *
from flask import Flask, render_template, redirect, url_for, request, jsonify
from classes import Chamber, OdorColumn
from functions import *


import inspect



# Function to get the list of Python files in the 'protocols' subfolder
def get_protocol_files():
    protocols_dir = os.path.join(os.path.dirname(__file__), 'protocols')
    return [f for f in os.listdir(protocols_dir) if f.endswith('.py')]

# Update options to hold the names of the files in the 'protocols' subfolder
options = get_protocol_files()

app = Flask(__name__)

# Create 20 Chamber objects with unique IDs from 1 to 20
for id in range(1, 21):
    Chamber(id)

# Create OdorColumn objects for Column A and Column B
OdorColumn('A')
OdorColumn('B')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', chambers=Chamber._instances, odor_columns=OdorColumn._instances, options=options)

@app.route('/get_functions', methods=['POST'])
def get_functions():
    selected_option = request.json['selected_option']
    module_name, _ = os.path.splitext(selected_option)  # Remove the .py extension
    module = __import__(f'protocols.{module_name}', fromlist=[None])
    functions = [func for func, obj in inspect.getmembers(module, inspect.isfunction) if obj.__module__ == module.__name__]
    return jsonify(functions=functions)

@app.route('/start_experiment', methods=['POST'])
def start_experiment():
    selected_option = request.json['experiment_option']
    function_name = request.json['function_name']
    module_name, _ = os.path.splitext(selected_option)  # Remove the .py extension
    # Import and execute the selected protocol
    module = __import__(f'protocols.{module_name}', fromlist=[None])
    if hasattr(module, function_name):
        func = getattr(module, function_name)
        func()
    print(f'Experiment {selected_option} has started with function {function_name}')
    return jsonify(message=f'Experiment {selected_option} with function {function_name} has started and completed')

@app.route('/get_chamber_states', methods=['GET'])
def get_chamber_states():
    chambers_data = [{'id': chamber.chamber_id, 'leftShock': chamber.leftShock, 'rightShock': chamber.rightShock, 'currentFlyLoc': chamber.currentFlyLoc} for chamber in Chamber._instances]
    odor_columns_data = [{'id': odor_column.column_id, 'leftOdor': odor_column.leftOdor, 'rightOdor': odor_column.rightOdor, 'airFlow': odor_column.airFlow} for odor_column in OdorColumn._instances]
    return jsonify(chambers=chambers_data, odor_columns=odor_columns_data)

@app.route('/shock_left')
def shock_left():
    Chamber.shockLeft()
    return redirect(url_for('index'))

@app.route('/remove_shock_left')
def remove_shock_left():
    Chamber.removeShockLeft()
    return redirect(url_for('index'))

@app.route('/shock_right')
def shock_right():
    Chamber.shockRight()
    return redirect(url_for('index'))

@app.route('/remove_shock_right')
def remove_shock_right():
    Chamber.removeShockRight()
    return redirect(url_for('index'))

@app.route('/activate_odor_left')
def activate_odor_left():
    OdorColumn.activateOdorLeft()
    return redirect(url_for('index'))

@app.route('/activate_odor_right')
def activate_odor_right():
    OdorColumn.activateOdorRight()
    return redirect(url_for('index'))

@app.route('/disable_odor_left')
def disable_odor_left():
    OdorColumn.disableOdorLeft()
    return redirect(url_for('index'))

@app.route('/disable_odor_right')
def disable_odor_right():
    OdorColumn.disableOdorRight()
    return redirect(url_for('index'))

@app.route('/activate_air_flow')
def activate_air_flow():
    OdorColumn.activateAirflow()
    return redirect(url_for('index'))

@app.route('/disable_air_flow')
def disable_air_flow():
    OdorColumn.disableAirflow()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
