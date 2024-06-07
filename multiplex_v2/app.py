from flask import Flask, render_template, redirect, url_for, request  # Ensure `request` is imported
from classes import Chamber, OdorColumn

app = Flask(__name__)

# Create 20 Chamber objects with unique IDs from 1 to 20
for id in range(1, 21):
    Chamber(id)

# Create OdorColumn objects for Column A and Column B
OdorColumn('A')
OdorColumn('B')

options = ['classical_ziv', 'classical_eyal', 'operant_ziv', 'operant_dekel']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_option = request.form['experiment_option']
        return redirect(url_for('experiment_start', arg1=selected_option))
    return render_template('index.html', chambers=Chamber._instances, odor_columns=OdorColumn._instances, options=options)

@app.route('/experiment_start/<arg1>')
def experiment_start(arg1):
    # Call your experimentStart function here
    print('hello world')
    # experimentStart(arg1)
    return f"Experiment started with argument: {arg1}"

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
