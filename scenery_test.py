import swift
from spatialgeometry import Mesh
from spatialmath import SE3
from math import pi


env = swift.Swift()
env.launch(realtime=True)

table = Mesh(filename = 'models/Table.STL', scale = 0.001)
printer1 = Mesh(filename = 'models/P1S_BambuLab.stl', scale = 0.001)
printer2 = Mesh(filename = 'models/P1S_BambuLab.stl', scale = 0.001)
printer3 = Mesh(filename = 'models/P1S_BambuLab.stl', scale = 0.001)


table_origin = SE3(-0.8, 0.3, 0)
table.T = table_origin @ SE3.Rx(pi/2)
table._collision = False

printer1_origin = table_origin * SE3(1.52, -0.05, 0.75) * SE3.Rz(pi)
printer1.T = printer1_origin
printer1._collision = False

printer2_origin = printer1_origin * SE3(0.53, 0, 0)
printer2.T = printer2_origin
printer2._collision = False

printer3_origin = printer2_origin * SE3(0.53, 0, 0)
printer3.T = printer3_origin
printer3._collision = False

env.add(table)
env.add(printer1)
env.add(printer2)
env.add(printer3)

while True:
    env.step(0.05)