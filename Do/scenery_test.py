import swift
from scenery import Scenery


env = swift.Swift()
env.launch(realtime=True)

scenery = Scenery()
scenery.add_to_env(env)

env.step(1)

# Test linear platform movement
# scenery.move1(env)
# scenery.move2(env)
# scenery.home(env)
scenery.printer1.door.open_door(env)
scenery.printer1.door.close_door(env)


while True:
    env.step(0.05)