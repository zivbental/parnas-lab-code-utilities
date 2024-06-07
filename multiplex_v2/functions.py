from classes import *
import time

def shock_all_left(duration):
    for chamber in Chamber._instances:
        chamber.shockLeft()
    time.sleep(duration)
    for chamber in Chamber._instances:
        chamber.removeShockLeft()

def wait(duration):
    time.sleep(duration)
