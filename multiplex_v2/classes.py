class Chamber:
    _used_ids = set()
    _instances = []

    def __init__(self, chamber_id):
        if chamber_id in Chamber._used_ids:
            raise ValueError(f"Chamber ID '{chamber_id}' is already in use.")
        
        self.chamber_id = chamber_id
        self.leftShock = False
        self.rightShock = False
        self.currentFlyLoc = -1

        Chamber._used_ids.add(chamber_id)
        Chamber._instances.append(self)

    def release(self):
        Chamber._used_ids.discard(self.chamber_id)
        Chamber._instances.remove(self)

    def update_fly_location(self, location):
        self.currentFlyLoc = location

    @classmethod
    def shockLeft(cls):
        for chamber in cls._instances:
            chamber.leftShock = True

    @classmethod
    def removeShockLeft(cls):
        for chamber in cls._instances:
            chamber.leftShock = False

    @classmethod
    def shockRight(cls):
        for chamber in cls._instances:
            chamber.rightShock = True

    @classmethod
    def removeShockRight(cls):
        for chamber in cls._instances:
            chamber.rightShock = False

class OdorColumn:
    _instances = []

    def __init__(self, column_id):
        self.column_id = column_id
        self.leftOdor = False
        self.rightOdor = False
        self.airFlow = False
        OdorColumn._instances.append(self)

    @classmethod
    def activateAirflow(cls):
        for odor_column in cls._instances:
            odor_column.airFlow = True

    @classmethod
    def disableAirflow(cls):
        for odor_column in cls._instances:
            odor_column.airFlow = False

    @classmethod
    def activateOdorLeft(cls):
        for odor_column in cls._instances:
            odor_column.leftOdor = True

    @classmethod
    def activateOdorRight(cls):
        for odor_column in cls._instances:
            odor_column.rightOdor = True

    @classmethod
    def disableOdorLeft(cls):
        for odor_column in cls._instances:
            odor_column.leftOdor = False

    @classmethod
    def disableOdorRight(cls):
        for odor_column in cls._instances:
            odor_column.rightOdor = False
