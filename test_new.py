from Solver_New import Solver
import numpy as np
import taichi as ti
import taichi.math as tm
import argparse
import os

ti.init(arch=ti.cuda, kernel_profiler=True, debug=True, default_fp=ti.f32)
# ti.init(arch=ti.cpu, debug=True)
gui = ti.GUI("Demo2dim_Naive_Crafts", res=1024, background_color=0x112F41)


# craft_type = np.array([1, 1])
# craft_camp = np.array([1, 0])
# x = np.array([[100, 100, 0], [400, 100, 0]])
# length = np.array([25, 35])
# width = np.array([10, 15])
x_coord = np.linspace(300, 700, 20)
grid_x, grid_y = np.meshgrid(x_coord, x_coord)
x = np.column_stack((grid_x.ravel(), grid_y.ravel()))
#
x = np.random.uniform(low=200, high=700, size=(400, 2))
# combined_actions = np.random.uniform(low=-0.5, high=0.5, size=(400, 2)) * 3
combined_actions = np.ones((400,400)) * 2.0
# combined_actions = np.array([[0, 1]] * 500) * 0.1
craft_type = np.ones(400)
craft_camp = np.ones(400)
length = np.ones(400)
width = np.ones(400) * 2

input_list = [[[100, 100], [200,200], [100, 300]], [[700, 700], [800, 700], [700, 800]]]

MR = Solver('L-BFGS', [1000, 1000], 3, 20.0, 40.0, accuracy='f32', substeps=1)
MR.init_env(craft_type, craft_camp, x, length, width, input_list)
info = MR.reset()
# print(info)

for frame in range(1800):
    # combined_actions = np.array([[0, 0.2], [0, 0.2]])
    # combined_actions = np.random.uniform(low=-0.5, high=0.5, size=(400, 2))
    info = MR.optimize_LBFGS(combined_actions)
    # print(info['position']/1000)
    gui.circles(info['position']/1000,
                radius=3,
                color = 0xED553B)
    gui.lines(begin=MR.start_np, end=MR.end_np, radius=2, color=0xED553B)
                # color=np.array([0xED553B, 0xFFFF00],dtype=np.uint32))
    # gui.lines(begin=info['RenderX']/1000, end=info['RenderY']/1000, radius=2, color=0xED553B)
    gui.show()

# for frame in range(400):
#     # combined_actions = np.array([[0.1, 0], [-0.1, 0]])
#     info = MR.optimize_LBFGS(combined_actions)
#     # gui.circles(info['position']/1000,
#     #             radius=6,
#     #             color=np.array([0xED553B, 0xFFFF00],dtype=np.uint32))
#     gui.circles(info['position'] / 1000,
#                 radius=6,
#                 color=0xED553B)
#     # gui.lines(begin=info['RenderX']/1000, end=info['RenderY']/1000, radius=2, color=0xED553B)
#     # print(info['Collided'])
#     gui.show()

ti.profiler.print_kernel_profiler_info()  # The default mode: 'count'