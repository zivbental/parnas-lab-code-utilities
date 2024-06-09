
import sys
import os

# Add the main directory to sys.path
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir)

# Import classes.py from the main directory
from classes import *
from functions import *


def zivOperant():
    OdorColumn.activateAirflow()
    wait(1)
    OdorColumn.activateOdorLeft()
    Chamber.shockLeft()
    wait(3)
    Chamber.removeShockLeft()
    OdorColumn.disableOdorLeft()
    wait(3)
    OdorColumn.activateOdorRight()
    wait(3)
    OdorColumn.disableOdorRight()
    wait(3)
    OdorColumn.activateOdorLeft()
    OdorColumn.activateOdorRight()
    wait(3)
    OdorColumn.disableOdorLeft()
    OdorColumn.disableOdorRight()
    OdorColumn.disableAirflow()

def zivClassical():
    OdorColumn.activateAirflow()
    wait(1)
    OdorColumn.activateOdorLeft()
    OdorColumn.activateOdorRight()
    wait(3)
    OdorColumn.disableOdorLeft()
    OdorColumn.disableOdorRight()
    wait(3)
    OdorColumn.activateOdorLeft()
    OdorColumn.activateOdorRight()
    Chamber.shockLeft()
    Chamber.shockRight()
    wait(3)
    Chamber.removeShockLeft()
    Chamber.removeShockRight()
    OdorColumn.disableOdorLeft()
    OdorColumn.disableOdorRight()
    wait(3)
    OdorColumn.activateOdorLeft()
    OdorColumn.activateOdorRight()
    wait(3)
    OdorColumn.disableOdorLeft()
    OdorColumn.disableOdorRight()
    OdorColumn.disableAirflow()

def zivClassical_v2():
    OdorColumn.activateAirflow()
    wait(1)
    OdorColumn.activateOdorLeft()
    OdorColumn.activateOdorRight()
    wait(3)
    OdorColumn.disableOdorLeft()
    OdorColumn.disableOdorRight()
    wait(3)
    OdorColumn.activateOdorLeft()
    OdorColumn.activateOdorRight()
    Chamber.shockLeft()
    Chamber.shockRight()
    wait(3)
    Chamber.removeShockLeft()
    Chamber.removeShockRight()
    OdorColumn.disableOdorLeft()
    OdorColumn.disableOdorRight()
    wait(3)
    OdorColumn.activateOdorLeft()
    OdorColumn.activateOdorRight()
    wait(3)
    OdorColumn.disableOdorLeft()
    OdorColumn.disableOdorRight()
    OdorColumn.disableAirflow()