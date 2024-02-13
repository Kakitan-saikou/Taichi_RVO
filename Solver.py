import taichi as ti
import numpy as np
import time
import numbers
import math
import multiprocessing as mp
import taichi.math as tm

@ti.data_oriented
class Solver:
    object_uav   = 0
    object_ship  = 1

    def __init__(self,
                 res,
                 scale,
                 control_freq=20,
                 max_num=1024,
                 substeps=1,
                 ep_length=9999,
                 discrete=True):
        self.dim = 2
        self.scale = scale
        self.max_num = max_num
        self.discrete = discrete

        self.substeps = substeps
        self.ep_length = ep_length
        self.global_substep = 0
        self.dt = 1/(control_freq * substeps)

        self.T = 0.0
        self.res = ti.Vector([res[0], res[1]])

        # self.res = res
        self.n_crafts = ti.field(ti.i32, shape=())

        self.craft_type = ti.field(dtype=ti.i32)
        self.craft_camp = ti.field(dtype=ti.i32)

        self.x = ti.Vector.field(self.dim, dtype=ti.f32)
        self.v = ti.Vector.field(self.dim, dtype=ti.f32)

        self.length = ti.field(dtype=ti.f32)
        self.width = ti.field(dtype=ti.f32)
        self.half_slide = ti.field(dtype=ti.f32)
        self.angle = ti.field(dtype=ti.f32)

        self.collided = ti.field(dtype=ti.i32)

        self.crafts = ti.root.dynamic(ti.i, self.max_num, chunk_size=None)

        self.crafts.place(self.craft_type, self.craft_camp, self.x, self.v, self.length, self.width, self.half_slide, self.angle, self.collided)


    def init_env(self, craft_type, craft_camp, x, length, width):

        self.n_crafts[None] = 0
        self.init_levelsets()

        self.craft_type_np = craft_type
        self.craft_camp_np = craft_camp
        self.length_np     = length
        self.width_np      = width
        self.x_np          = x



    def reset(self):
        self.n_crafts[None] = 0
        self.add_crafts_kernel(self.craft_type_np, self.craft_camp_np, self.x_np, self.length_np, self.width_np)

        craft_info = self.render_info()

        return craft_info



    @ti.kernel
    def add_crafts_kernel(self,
                           craft_type: ti.types.ndarray(),
                           craft_camp: ti.types.ndarray(),
                           x         : ti.types.ndarray(),
                           length    : ti.types.ndarray(),
                           width     : ti.types.ndarray()):

        new_crafts = craft_type.shape[0]

        for i in range(self.n_crafts[None], self.n_crafts[None] + new_crafts):
            self.craft_type[i] = craft_type[i]
            self.craft_camp[i] = craft_camp[i]
            self.length[i]     = length[i]
            self.width[i]      = width[i]
            self.half_slide[i] = 0.5 * tm.sqrt(length[i]**2 + width[i]**2)
            self.angle[i]      = tm.asin(width[i]/self.half_slide[i])
            self.collided[i]   = 0
            self.v[i]          = ti.Vector.zero(ti.f32, self.dim)

            for j in ti.static(range(self.dim)):
                self.x[i][j] = x[i, j]

        self.n_crafts[None] += new_crafts

    @ti.kernel
    def rasterize(self):
        for i in range(ti.static(self.n_crafts[None])):
            pass

    @ti.kernel
    def init_levelsets(self):
        pass

    @ti.kernel
    def collision_detection(self):
        #TODO: Maybe add levelset colliders?
        for i in range(self.n_crafts[None]):
            self.collided[i] = 0
            local_normal = tm.normalize(self.v[i])
            local_angle = tm.atan2(self.v[i][1], self.v[i][0])
            local_width = self.width[i] / 2

            for j in range(self.n_crafts[None]):
                if j != i:
                    absolute_dis = tm.length(self.x[i] - self.x[j])
                    if absolute_dis < (self.width[i] + self.width[j])/2:
                        self.collided[i] = 1
                        break

                    elif absolute_dis < (self.half_slide[i] + self.half_slide[j])/2:
                        j2normal_dis = tm.dot(local_normal, self.x[j])
                        j_angle = tm.atan2(self.v[j][1], self.v[j][0])
                        rel_angle = tm.pi/2 - local_angle + j_angle - self.angle[j]
                        pos_angle0 = rel_angle - self.angle[j]
                        pos_angle1 = rel_angle + self.angle[j]
                        para = tm.max(ti.abs(tm.cos(pos_angle0)), ti.abs(tm.cos(pos_angle1)))
                        if j2normal_dis < para * self.half_slide[j] + local_width:
                            self.collided[i] = 1
                            break



    @ti.kernel
    def dynamic_step(self):
        #Semi-implicit Euler:
        for i in range(self.n_crafts[None]):
            self.x[i] += self.v[i] * self.dt



    @ti.kernel
    def process_action(self, actions: ti.types.ndarray()):
        for i in range(self.n_crafts[None]):
                # velocity control
            self.v[i] = ti.Vector([actions[i, 0], actions[i, 1]])


    def solver_step(self, actions):
        self.process_action(actions)

        for k in range(0, self.substeps):
            self.dynamic_step()
            self.collision_detection()
            self.global_substep += 1

        craft_info = self.render_info()

        return craft_info


    @ti.kernel
    def get_render_info(self, type:ti.types.ndarray(), camp:ti.types.ndarray(), x:ti.types.ndarray(), v:ti.types.ndarray(), renderx:ti.types.ndarray(), rendery:ti.types.ndarray(), collided:ti.types.ndarray()):
        for i in range(self.n_crafts[None]):
            normal_angle = tm.atan2(self.v[i][1], self.v[i][0])
            rot_mat = tm.rotation2d(normal_angle)

            x0 = self.x[i] + ti.Vector([self.length[i]/2, self.width[i]/2]) @ rot_mat
            x1 = self.x[i] + ti.Vector([-self.length[i]/2, self.width[i]/2]) @ rot_mat
            x2 = self.x[i] + ti.Vector([-self.length[i]/2, -self.width[i]/2]) @ rot_mat
            x3 = self.x[i] + ti.Vector([self.length[i]/2, -self.width[i]/2]) @ rot_mat

            type[i] = self.craft_type[i]
            camp[i] = self.craft_camp[i]
            collided[i] = self.collided[i]

            for j in ti.static(range(self.dim)):
                x[i, j] = self.x[i][j]
                v[i, j] = self.v[i][j]
                renderx[i * 4, j] = x0[j]
                renderx[i * 4 + 1, j] = x1[j]
                renderx[i * 4 + 2, j] = x2[j]
                renderx[i * 4 + 3, j] = x3[j]

                rendery[i * 4, j] = x1[j]
                rendery[i * 4 + 1, j] = x2[j]
                rendery[i * 4 + 2, j] = x3[j]
                rendery[i * 4 + 3, j] = x0[j]

    def render_info(self):
        np_x = np.ndarray((self.n_crafts[None], self.dim), dtype=np.float32)
        np_v = np.ndarray((self.n_crafts[None], self.dim), dtype=np.float32)
        np_type = np.ndarray((self.n_crafts[None], ), dtype=np.float32)
        np_camp = np.ndarray((self.n_crafts[None], ), dtype=np.float32)
        np_collide = np.ndarray((self.n_crafts[None],), dtype=np.int32)

        #For rendering rectangles
        np_renderx = np.ndarray((4 * self.n_crafts[None], self.dim), dtype=np.float32)
        np_rendery = np.ndarray((4 * self.n_crafts[None], self.dim), dtype=np.float32)
        self.get_render_info(np_type, np_camp, np_x, np_v, np_renderx, np_rendery, np_collide)

        craft_data = {
            'position': np_x,
            'velocity': np_v,
            'type': np_type,
            'camp': np_camp,
            'RenderX': np_renderx,
            'RenderY': np_rendery,
            'Collided': np_collide
        }

        return craft_data

    @ti.kernel
    def copy_dynamic_nd(self, np_x: ti.types.ndarray(), input_x: ti.template()):
        for i in self.x:
            for j in ti.static(range(self.dim)):
                np_x[i, j] = input_x[i][j]

    @ti.kernel
    def copy_dynamic(self, np_x: ti.types.ndarray(), input_x: ti.template()):
        for i in self.x:
            np_x[i] = input_x[i]





