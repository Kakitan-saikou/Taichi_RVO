from Solver_New import Solver
import numpy as np
import taichi as ti
import taichi.math as tm
import argparse
import os

ti.init(arch=ti.cpu)
gui = ti.GUI("Demo2dim_Naive_Crafts", res=1024, background_color=0x112F41)


craft_type = np.array([1, 1])
craft_camp = np.array([1, 0])
x = np.array([[10, 10, 0], [400, 10, 0]])
length = np.array([25, 35])
width = np.array([10, 15])



MR = Solver([1000, 1000], 3, 5, 10, substeps=1)
MR.init_env(craft_type, craft_camp, x, length, width)
info = MR.reset()
print(info)

for frame in range(400):
    combined_actions = np.array([[0, 2], [0, 2]])
    info = MR.solver_step(combined_actions)
    # print(info)
    # gui.circles(info['position']/1000,
    #             radius=6,
    #             color=np.array([0xED553B, 0xFFFF00],dtype=np.uint32))
    gui.lines(begin=info['RenderX']/1000, end=info['RenderY']/1000, radius=2, color=0xED553B)
    gui.show()

for frame in range(400):
    combined_actions = np.array([[1, 0], [-1, 0]])
    info = MR.solver_step(combined_actions)
    # gui.circles(info['position']/1000,
    #             radius=6,
    #             color=np.array([0xED553B, 0xFFFF00],dtype=np.uint32))
    gui.lines(begin=info['RenderX']/1000, end=info['RenderY']/1000, radius=2, color=0xED553B)
    print(info['Collided'])
    gui.show()

