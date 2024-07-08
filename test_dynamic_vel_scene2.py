from Solver_Dynamic import Solver
import numpy as np
import taichi as ti
import taichi.math as tm
import argparse
import os
import matplotlib.pyplot as plt

ti.init(arch=ti.cpu, kernel_profiler=True, debug=True, default_fp=ti.f32)
# ti.init(arch=ti.cpu, debug=True)
gui = ti.GUI("Demo2dim_Naive_Crafts", res=1024, background_color=0x112F41)


# craft_type = np.array([1, 1])
# craft_camp = np.array([1, 0])
# x = np.array([[100, 100, 0], [400, 100, 0]])
# length = np.array([25, 35])
# width = np.array([10, 15])
# x_coord = np.linspace(200, 650, 30)
# grid_x, grid_y = np.meshgrid(x_coord, x_coord)
# x = np.column_stack((grid_x.ravel(), grid_y.ravel()))
#
# x = np.random.uniform(low=200, high=650, size=(900, 2))
# combined_actions = np.random.uniform(low=-0.5, high=0.5, size=(900, 2)) * 2
# # combined_actions = np.ones((900,900)) * 2.0
# # combined_actions = np.array([[0, 1]] * 500) * 0.1
# craft_type = np.ones(900)
# craft_camp = np.zeros(900)
# length = np.ones(900)
# width = np.ones(900) * 2.0

x = np.array([[215.0, 205.0], [215., 795.], [795., 785.], [795., 215.]])
craft_type = np.ones(4)
craft_camp = np.array([0, 1, 2, 3])
length = np.ones(4)
width = np.ones(4) * 1.0
combined_actions = np.array([[0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5]])
max_vel = np.ones(4) * 1.8
# combined_actions = np.array([[1.5, 0.8], [-1.5, -0.8]])

input_list = [[[200, 200], [200, 800], [800, 800], [800, 200]]]

# input_list.append([[300, 700], [600, 600], [500, 720]])
input_list.append([[710, 220], [760, 320], [700, 330], [600, 280], [560, 240]])

input_list.append([[300, 300], [350, 320], [350, 350],  [320, 350]])
# input_list.append([ [300, 400], [400, 400], [400, 300], [300, 300]])

input_list.append([[650, 650], [670, 650], [700, 700], [650, 670]])
# input_list.append([[600, 700], [700, 700], [700, 600], [600, 600]])
# [[400, 500], [300,500], [300, 550], [400, 550]], [[700, 450], [600, 450], [600, 500], [700, 500]]
# [[300, 500], [400,500], [400, 550], [300, 550]], [[600, 450], [700, 450], [700, 500], [600, 500]]
MR = Solver('L-BFGS', [1000, 1000], 3, 5.0, 5.0, accuracy='f32', substeps=1, control_freq=10, control='velocity')
# MR.init_env(craft_type, craft_camp, x, length, width, input_list)
# info = MR.reset()
# print(info)
MR.n_crafts[None] = 0
# MR.add_crafts_dynamic(craft_type, craft_camp, x, combined_actions, length, width)
MR.setup_line_obstacles(input_list)
MR.setup_ao_grid_gpu()
MR.init_ao_grid()
MR.init_obstacle_grid()

MR.setup_grid_gpu()
MR.setup_hessian_builder()

# terminals = np.array([[555.0, 595.0], [445.0, 405.0]])
terminals = np.array([[795., 795.], [795.0, 205.0], [205.0, 205.0], [205.0, 795.0]])
MR.register_terminals(terminals)
substeps = 0
# MR.add_crafts_dynamic(craft_type, craft_camp, x, combined_actions, length, width)
for frame in range(1000):
    # combined_actions = np.array([[0, 0.2], [0, 0.2]])
    # combined_actions = np.random.uniform(low=-0.5, high=0.5, size=(400, 2))
    if substeps % 100==0:
        MR.add_crafts_dynamic(craft_type, craft_camp, x, combined_actions, length, width, max_vel)
    info, substep = MR.optimize_LBFGS(None)
    substeps += substep
    print(substeps % 250)
    # print(info['position']/1000)
    pos = info['position']/1000
    perf_end = info['perf_end']/1000
    grad_end = info['grad_end']/1000

    gui.circles(pos,
                radius=3,
                color=0xED553B)
    gui.lines(begin=MR.start_np, end=MR.end_np, radius=1, color=0xED553B)
    gui.lines(begin=pos, end=perf_end, radius=1, color=0xEEDD11)
    gui.lines(begin=pos, end=grad_end, radius=1, color=0x22DDFF)
                # color=np.array([0xED553B, 0xFFFF00],dtype=np.uint32))
    # gui.lines(begin=info['RenderX']/1000, end=info['RenderY']/1000, radius=2, color=0xED553B)
    gui.show()
    print(MR.n_crafts[None])

# for frame in range(1000):
#     print('new', frame)
#     info, substep = MR.optimize_LBFGS()
#     # print(info['position']/1000)
#     gui.circles(info['position'] / 1000,
#                 radius=3,
#                 color=0xED553B)
#     gui.lines(begin=MR.start_np, end=MR.end_np, radius=1, color=0xED553B)
#     # color=np.array([0xED553B, 0xFFFF00],dtype=np.uint32))
#     # gui.lines(begin=info['RenderX']/1000, end=info['RenderY']/1000, radius=2, color=0xED553B)
#     gui.show()
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
E_info = np.array(MR.logged_energy)
plt.plot(E_info[600:800])
plt.show()
print(E_info[600:800])