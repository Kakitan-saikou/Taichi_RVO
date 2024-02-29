import taichi as ti
import numpy as np
import time
import numbers
import math
import multiprocessing as mp
import taichi.math as tm

@ti.data_oriented
class Solver:
    def __init__(self,
                 res,
                 scale,
                 GridR,
                 d0,
                 control_freq=20,
                 max_num=1024,
                 substeps=1,
                 ep_length=9999,
                 discrete=True):
        self.dim = 2
        self.scale = scale
        self.max_num = max_num
        self.discrete = discrete
        self.maxInGrid = 1024

        self.substeps = substeps
        self.ep_length = ep_length
        self.global_substep = 0
        self.dt = 1/(control_freq * substeps)

        self.GridR = GridR
        self.invGridR = 1 / GridR
        self.searchR = d0
        self.neighbor_grids = int(d0/GridR + 0.5)
        self.coef = 0.001

        self.T = 0.0
        self.res = ti.Vector([res[0], res[1]])
        # self.global_E = 0.0
        # self.prev_global_E = 0.0
        self.alpha = ti.field(ti.f32, shape=())
        self.alpha_dec = 0.6
        self.global_alpha_min = 0.1

        self.global_E = ti.field(ti.f32, shape=())
        self.prev_global_E = ti.field(ti.f32, shape=())

        # self.res = res
        self.n_neigbors = ti.field(ti.i32, shape=())
        self.n_crafts = ti.field(ti.i32, shape=())
        self.n_obstacles = ti.field(ti.i32, shape=())
        self.n_ao = ti.field(ti.i32, shape=())

        self.craft_type = ti.field(dtype=ti.i32)
        self.craft_camp = ti.field(dtype=ti.i32)

        self.x = ti.Vector.field(self.dim, dtype=ti.f32)
        self.v = ti.Vector.field(self.dim, dtype=ti.f32)
        self.gradient = ti.Vector.field(self.dim, dtype=ti.f32)
        self.new_v = ti.Vector.field(self.dim, dtype=ti.f32)
        self.new_pos = ti.Vector.field(self.dim, dtype=ti.f32)

        self.obstacle_pos = ti.Vector.field(self.dim, dtype=ti.f32)
        self.obstacle_r = ti.field(dtype=ti.f32)

        # self.diag = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32)

        self.length = ti.field(dtype=ti.f32)
        self.width = ti.field(dtype=ti.f32)
        self.half_slide = ti.field(dtype=ti.f32)
        self.angle = ti.field(dtype=ti.f32)

        self.collided = ti.field(dtype=ti.i32)

        self.crafts = ti.root.dynamic(ti.i, self.max_num, chunk_size=None)
        self.obstacles = ti.root.dynamic(ti.i, 32, chunk_size=None)

        self.crafts.place(self.craft_type, self.craft_camp, self.x, self.new_pos, self.v, self.gradient, self.new_v, self.length, self.width, self.half_slide, self.angle, self.collided)
        self.obstacles.place(self.obstacle_pos, self.obstacle_r)

        # self.craft_hessian = ti.root.dense


    def setup_grid_gpu(self):
        # Whenever add agents, reset grid
        # ao means agent-obstacle
        self.gridCount       = ti.field(dtype=ti.i32)
        self.grid            = ti.field(dtype=ti.i32)
        # self.neighborCount   = ti.field(dtype=ti.i32)
        # self.neighbor        = ti.field(dtype=ti.i32)
        # self.ao_gridCount    = ti.field(dtype=ti.i32)
        self.ao_grid         = ti.field(dtype=ti.i32)

        self.occupied_num    = ti.field(dtype=ti.i32, shape=())
        self.occupied_index  = ti.field(dtype=ti.i32)


        self.blockSize       = ti.Vector.field(self.dim, dtype=ti.i32, shape=(1))
        self.min_boundary    = ti.Vector.field(self.dim, dtype=ti.f32, shape=(1))
        self.max_boundary    = ti.Vector.field(self.dim, dtype=ti.f32, shape=(1))

        self.maxNeighbourPair= self.n_crafts[None] * 1024
        self.maxAOPair = self.n_crafts[None] * self.n_obstacles[None]

        # self.candidate_neighbor   = ti.types.ndarray(dtype=ti.math.vec2, dim=self.maxNeighbourPair)
        self.candidate_neighbor   = ti.Vector.field(self.dim, dtype=ti.i32)
        self.diag = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32)

        self.candidate_ao = ti.Vector.field(self.dim, dtype=ti.i32)
        self.diag_ao = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32)


        # self.hash_grid = ti.root.dynamic(ti.i, self.max_num, chunk_size=None)
        # self.hash_neighbour = ti.root.dynamic(ti.ij, self.max_num, chunk_size=None)
        # Hash grids for aa
        ti.root.dense(ti.i, self.max_num).place(self.gridCount)
        ti.root.dense(ti.ij, (self.max_num, self.maxInGrid)).place(self.grid)
        ti.root.dynamic(ti.i, self.maxNeighbourPair).place(self.candidate_neighbor, self.diag)
        ti.root.dynamic(ti.i, self.maxAOPair).place(self.candidate_ao, self.diag_ao)

        # Hash grids for ao
        ti.root.dense(ti.i, self.max_num).place(self.ao_gridCount)
        ti.root.dense(ti.ij, (self.max_num, self.n_obstacles[None])).place(self.ao_grid)
        ti.root.dynamic(ti.i, 1000).place(self.occupied_index)

        # ti.root.dense(ti.i, self.n_crafts[None]).place(self.neighborCount)
        # ti.root.dense(ti.ij, (self.n_crafts[None], self.maxNeighbour)).place(self.neighbor)
        #TODO: Using ti.root.pointer to activate sparse architectures

    def setup_hessian_builder(self):
        # self.partial_hessian = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32)
        # self.craft_hessian = ti.root.dense(ti.ij, (self.n_crafts[None], self.n_crafts[None])).place(self.partial_hessian)
        self.Global_Hessian = ti.linalg.SparseMatrixBuilder(self.dim * self.n_crafts[None], self.dim * self.n_crafts[None], max_num_triplets=100000)
        self.Global_Grad = ti.ndarray(ti.f32, self.dim * self.n_crafts[None])

    @ti.kernel
    def init_occupied_grids(self):
        #Find obstacles occupied grids; circle obstacles
        self.occupied_num[None] = 0
        self.occupied_index.deactive()
        for i in range(self.n_obstacles[None]):
            sqrt_r = tm.sqrt(self.obstacle_r[i])
            index_x = self.obstacle_pos[0]
            index_y = self.obstacle_pos[1]

            index_l = ti.cast((index_x - sqrt_r) * self.invGridR, ti.i32) - 1
            index_r = ti.cast((index_x + sqrt_r) * self.invGridR, ti.i32) + 1
            index_d = ti.cast((index_y - sqrt_r) * self.invGridR, ti.i32) - 1
            index_u = ti.cast((index_y + sqrt_r) * self.invGridR, ti.i32) + 1

            for m in range(index_l, index_r + 1):
                for n in ti.static(range(index_d, index_u + 1)):
                    indexO = ti.Vector([m, n])
                    hash_index = self.get_cell_hash(indexO)
                    old = ti.atomic_add(self.ao_gridCount[hash_index], 1)
                    self.ao_grid[hash_index, old] = i
                    self.occupied_num[None] += 1
                    self.occupied_index.append(hash_index)



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
        self.setup_grid_gpu()

        craft_info = self.render_info()
        self.setup_hessian_builder()

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
            self.gradient[i]   = ti.Vector.zero(ti.f32, self.dim)
            self.new_pos[i]    = ti.Vector.zero(ti.f32, self.dim)

            for j in ti.static(range(self.dim)):
                self.x[i][j] = x[i, j]

        self.n_crafts[None] += new_crafts
        #Setup hessians here
        # self.setup_hessian_builder()

    @ti.kernel
    def init_grid(self):
        for i, j in self.grid:
            self.grid[i, j] = -1
            self.gridCount[i] = 0


    @ti.kernel
    def insert_grid_pos(self):
        for i in range(self.n_crafts[None]):
            indexV = ti.cast(self.x[i] * self.invGridR, ti.i32)
            # print(indexV)
            hash_index = self.get_cell_hash(indexV)
            # print(hash_index)
            old = ti.atomic_add(self.gridCount[hash_index], 1)
            # print(old)
            if old > self.maxInGrid - 1:
                print("exceed grid", old)
                self.gridCount[hash_index] = self.maxInGrid
            else:
                self.grid[hash_index, old] = i

    @ti.kernel
    def find_neighbour(self):
        # find neighbours and add neighbor pairs to a dynamic field
        self.n_neigbors[None] = 0
        for i in range(self.n_crafts[None]):
            indexV = ti.cast(self.x[i] * self.invGridR, ti.i32) - ti.Vector([self.neighbor_grids, self.neighbor_grids])
            # print(indexV)
            for offset in ti.static(ti.grouped(self.stencil_range())):
            # for offset in self.stencil_range():
            #     print(str(offset))
                hash_index = self.get_cell_hash(indexV + offset)
                # print(hash_index)
                k = 0
                new_pair = 0
                while k < self.gridCount[hash_index]:
                    j = self.grid[hash_index, k]
                    if j > i:  # prevent calculating twice
                        new_pair = ti.atomic_add(self.n_neigbors[None], 1)
                        self.candidate_neighbor[new_pair] = ti.Vector([i, j])
                        # print("Nei", int(i), int(j))
                    k += 1

        self.n_neigbors[None] -= 1


    @ti.kernel
    def find_ao_pair(self):
        # find Agent-Obstacle pairs
        self.n_ao[None] = 0
        for hash_index in self.occupied_index:
            for m in ti.static(range(self.ao_gridCount[hash_index])):
                #Obstacle index m
                k = 0
                while k < self.gridCount[hash_index]:
                    j = self.grid[hash_index, k]
                    new_pair = ti.atomic_add(self.n_ao[None], 1)
                    self.candidate_ao[new_pair] = ti.Vector([m, j])
                k += 1

        self.n_ao[None] -= 1





    #               self.insert_neighbor(i, indexV + offset)
    #
    # @ti.func
    # def insert_neighbor(self, i, index_neigh):
    #     # if index_neigh.x >= 0 and index_neigh.x < self.blockSize[0].x and \
    #     #         index_neigh.y >= 0 and index_neigh.y < self.blockSize[0].y and \
    #     #         index_neigh.z >= 0 and index_neigh.z < self.blockSize[0].z:
    #
    #     hash_index = self.get_cell_hash(index_neigh)
    #     k = 0
    #     while k < self.gridCount[hash_index]:
    #         j = self.grid[hash_index, k]
    #         if j > i:
    #
    #         if j >= 0 and (i != j):
    #             r = self.particle_data.pos[i] - self.particle_data.pos[j]
    #             old = ti.atomic_add(self.neighborCount[i], 1)
    #             if old > self.maxNeighbour - 1:
    #                 old = old
    #                 print("exceed neighbor", old)
    #             else:
    #                 self.neighbor[i, old] = j
    #         k += 1

    @ti.func
    def get_cell_hash(self, a):
        p1 = 73856093 * a.x
        p2 = 19349663 * a.y

        return ((p1 ^ p2) % self.max_num + self.max_num) % self.max_num



    @ti.kernel
    def rasterize(self):
        for i in range(ti.static(self.n_crafts[None])):
            pass

    @ti.func
    def stencil_range(self):
        # return ti.ndrange(*((-self.neighbor_grids, self.neighbor_grids + 1) * self.dim))
        return ti.ndrange(*((self.neighbor_grids * 2 + 1, ) * self.dim))

    @ti.kernel
    def init_levelsets(self):
        pass

    # @ti.kernel
    # def collision_detection(self):
    #     #TODO: Maybe add levelset colliders?
    #     for i in range(self.n_crafts[None]):
    #         self.collided[i] = 0
    #         local_normal = tm.normalize(self.v[i])
    #         local_angle = tm.atan2(self.v[i][1], self.v[i][0])
    #         local_width = self.width[i] / 2
    #
    #         for j in range(self.n_crafts[None]):
    #             if j != i:
    #                 absolute_dis = tm.length(self.x[i] - self.x[j])
    #                 if absolute_dis < (self.width[i] + self.width[j])/2:
    #                     self.collided[i] = 1
    #                     break
    #
    #                 elif absolute_dis < (self.half_slide[i] + self.half_slide[j])/2:
    #                     j2normal_dis = tm.dot(local_normal, self.x[j])
    #                     j_angle = tm.atan2(self.v[j][1], self.v[j][0])
    #                     rel_angle = tm.pi/2 - local_angle + j_angle - self.angle[j]
    #                     pos_angle0 = rel_angle - self.angle[j]
    #                     pos_angle1 = rel_angle + self.angle[j]
    #                     para = tm.max(ti.abs(tm.cos(pos_angle0)), ti.abs(tm.cos(pos_angle1)))
    #                     if j2normal_dis < para * self.half_slide[j] + local_width:
    #                         self.collided[i] = 1
    #                         break

    @ti.kernel
    def init_grad(self):
        #Given x and v, calculate E
        for i in range(self.n_crafts[None]):
            self.gradient[i] = -self.v[i] / self.dt
            xx = - self.v[i]*self.dt
            XX = 0.0

            for j in ti.static(ti.ndrange(self.dim)):
                XX += xx[j] ** 2

            dE = XX / (2.0 * self.dt * self.dt)
            self.prev_global_E[None] += dE

    @ti.kernel
    def compute_gradAA(self):
        for i in range(self.n_neigbors[None]):
            a_id = self.candidate_neighbor[i][0]
            b_id = self.candidate_neighbor[i][1]
            ab = self.x[a_id] - self.x[b_id]
            margin = - self.width[a_id] * self.width[a_id] * 4
            for j in ti.static(ti.ndrange(self.dim)):
                margin += ab[j] ** 2
            # margin = ab.squaredNorm()
            d0 = self.searchR - 2 * self.width[a_id]
            if margin < d0:
                valLog = tm.log(margin / d0)
                valLogC = valLog * (margin - d0)
                relD = 1 - d0 / margin
                D = - self.coef * (2 * valLogC + (margin - d0) * relD)

                self.gradient[a_id] += D * 2 * ab
                self.gradient[b_id] -= D * 2 * ab
                self.prev_global_E[None] -= valLogC * (margin - d0) * self.coef

                #Hessian Computation Part
                DD = -(4 * relD - relD * relD + 2 * valLog) * self.coef
                # mat1 = 4 * DD * ab * ab
                # print(mat1)
                ab0 = ab[0]
                ab1 = ab[1]
                mat1 = 4 * DD * tm.mat2([ab0 * ab0, ab0 * ab1], [ab0 * ab1, ab1 * ab1])
                diag = D * 2 * ti.Matrix.identity(ti.f32, self.dim) + mat1.transpose()
                self.diag[i] = diag
                # self.partial_hessian[a_id, a_id] += diag
                # self.partial_hessian[b_id, b_id] += diag
                # self.partial_hessian[a_id, b_id] -= diag
                # self.partial_hessian[b_id, a_id] -= diag.transpose()

    @ti.kernel
    def compute_gradAO(self):
        for i in range(self.n_ao[None]):
            o_id = self.candidate_ao[i][0]
            a_id = self.candidate_ao[i][1]

            ao =  - self.obstacle_pos[o_id] + self.x[a_id]

            margin = (ao.length - self.width[a_id] - self.obstacle_r[o_id]) ** 2
            d0 = self.searchR - self.width[a_id]
            if margin < d0:
                valLog = tm.log(margin / d0)
                valLogC = valLog * (margin - d0)
                relD = 1 - d0 / margin
                D = - self.coef * (2 * valLogC + (margin - d0) * relD)

                scale = self.width[a_id] / self.obstacle_r[o_id]

                self.gradient[a_id] += D * 2 * ao * scale
                self.prev_global_E[None] -= valLogC * (margin - d0) * self.coef

                # Hessian Computation Part
                DD = -(4 * relD - relD * relD + 2 * valLog) * self.coef
                # mat1 = 4 * DD * ab * ab
                # print(mat1)
                ao0 = ao[0] * scale
                ao1 = ao[1] * scale
                mat1 = 4 * DD * tm.mat2([ao0 * ao0, ao0 * ao1], [ao0 * ao1, ao1 * ao1])
                diag = D * 2 * ti.Matrix.identity(ti.f32, self.dim) + mat1.transpose()
                self.diag_ao[i] = diag

    @ti.kernel
    def assemble_hessian(self, H: ti.types.sparse_matrix_builder()):
        for i in range(self.n_neigbors[None]):
            a_id = self.candidate_neighbor[i][0]
            b_id = self.candidate_neighbor[i][1]
            diag = self.diag[i]
            diag_T = diag.transpose()

            for m, n in ti.static(ti.ndrange(2, 2)):
                H[2 * a_id + m, 2 * a_id + n] += diag[m, n]
                H[2 * b_id + m, 2 * b_id + n] += diag[m, n]
                H[2 * a_id + m, 2 * b_id + n] -= diag[m, n]
                H[2 * b_id + m, 2 * a_id + n] -= diag_T[m, n]

    @ti.kernel
    def assemble_hessian_ao(self, H: ti.types.sparse_matrix_builder()):
        for i in range(self.n_ao[None]):
            a_id = self.candidate_ao[i][0]
            diag = self.diag_ao[i]

            for m, n in ti.static(ti.ndrange(2, 2)):
                H[2 * a_id + m, 2 * a_id + n] += diag[m, n]

    @ti.kernel
    def assemble_grad(self, grad: ti.types.ndarray(), source: ti.template()):
        for i in range(self.n_crafts[None]):
            # self.Global_Grad[2 * i] = self.gradient[i][0]
            # self.Global_Grad[2 * i + 1] = self.gradient[i][1]
            for m in ti.static(range(2)):
                grad[2 * i + m] = source[i][m]

    @ti.kernel
    def compute_energy_U(self):
        for i in range(self.n_neigbors[None]):
            a_id = self.candidate_neighbor[i][0]
            b_id = self.candidate_neighbor[i][1]
            # ab = self.x[a_id] - self.x[b_id] + (-self.gradient[a_id] + self.gradient[b_id]) * self.alpha[None]
            ab = self.x[a_id] - self.x[b_id] + (self.new_v[a_id] - self.new_v[b_id]) * self.alpha[None]
            margin = - self.width[a_id] * self.width[a_id] * 4
            for j in ti.static(ti.ndrange(self.dim)):
                margin += ab[j] ** 2

            d0 = self.searchR - 2 * self.width[a_id]
            if margin < d0:
                valLog = tm.log(margin / d0)
                valLogC = valLog * (margin - d0)

                self.global_E[None] -= valLogC * (margin - d0) * self.coef



    @ti.kernel
    def compute_energy_E(self):
        # global_E = 0.0
        for i in range(self.n_crafts[None]):
            # xx = -self.gradient[i] * self.alpha[None] - self.v[i]*self.dt
            xx = self.new_v[i] * self.alpha[None] - self.v[i] * self.dt
            XX = 0

            for j in ti.static(ti.ndrange(self.dim)):
                XX += xx[j] ** 2

            dE = XX / (2 * self.dt * self.dt)
            self.global_E[None] += dE



    # @ti.kernel
    def line_search(self):
        # for i in range(self.n_neigbors[None]):
        self.alpha[None] = 1.0

        while self.alpha[None] > self.global_alpha_min:
            self.compute_energy_E()
            self.compute_energy_U()

            if self.global_E[None] < self.prev_global_E[None]:
                break
            else:
                self.alpha[None] *= self.alpha_dec

        # return self.alpha[None]


    @ti.kernel
    def dynamic_step(self):
        #WASTED
        #Semi-implicit Euler:
        for i in range(self.n_crafts[None]):
            self.x[i] += self.v[i] * self.dt

    @ti.kernel
    def update_pos(self, new_alpha: ti.f32):
        for i in range(self.n_crafts[None]):
            # self.x[i] -= self.gradient[i] * new_alpha
            self.x[i] += self.new_v[i] * new_alpha


    @ti.kernel
    def process_action(self, actions: ti.types.ndarray()):
        for i in range(self.n_crafts[None]):
                # velocity control
            self.v[i] = ti.Vector([actions[i, 0], actions[i, 1]])

    def sparse_solving(self):
        #TODO: add perturbation
        self.assemble_hessian(self.Global_Hessian)
        Hessian = self.Global_Hessian.build()

        self.assemble_grad(self.Global_Grad, self.gradient)

        sparse_solver = ti.linalg.SparseSolver(solver_type="LLT")
        sparse_solver.analyze_pattern(Hessian)
        sparse_solver.factorize(Hessian)
        new_direction = sparse_solver.solve(self.Global_Grad)

        self.Rasterize_Grad(new_direction, self.new_v)

    @ti.kernel
    def Rasterize_Grad(self, new_direction: ti.types.ndarray(), new_v: ti.template()):
        for i in range(self.n_crafts[None]):
            for m in ti.static(2):
                new_v[i][m] = new_direction[2 * i + m]

    def solver_step(self, actions):
        self.global_E[None] = 0.0
        self.prev_global_E[None] = 0.0

        self.process_action(actions)
        self.init_grid()
        self.insert_grid_pos()
        self.find_neighbour()
        # print(self.candidate_neighbor)

        self.init_grad()
        self.compute_gradAA()

        self.sparse_solving()

        self.line_search()
        self.update_pos(self.alpha[None])

        # WASTED
        # for k in range(0, self.substeps):
        #     self.dynamic_step()
        #     self.collision_detection()
        #     self.global_substep += 1

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





