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
                 optimizer,
                 res,
                 scale,
                 GridR,
                 d0,
                 control_freq=20,
                 max_num=2048,
                 accuracy='f32',
                 arch=ti.cuda,
                 debug=False,
                 substeps=1,
                 ep_length=9999,
                 discrete=True):
        self.dim = 2
        self.scale = scale
        self.max_num = max_num
        self.discrete = discrete
        self.maxInGrid = 2048
        self.max_iters = 1

        self.substeps = substeps
        self.ep_length = ep_length
        self.global_substep = 0
        self.dt = 1/(control_freq * substeps)

        self.GridR = GridR
        self.invGridR = 1 / GridR
        self.searchR = d0
        self.neighbor_grids = int(d0/GridR + 0.5)
        self.coef = 0.5

        self.accuracy = ti.f32
        if accuracy == 'f64':
            self.accuracy = ti.f64
        else:
            self.accuracy = ti.f32

        # ti.init(arch=arch, debug=debug, default_fp=self.accuracy)

        self.tmp_value = ti.field(self.accuracy, shape=())
        self.tmp_value[None] = 0.0

        self.optimizer = optimizer  # GD, Newton, BFGS, L-BFGS
        if self.optimizer == 'L-BFGS':
            self.m_iters = 5  # m = 5
            # self.alpha_i = ti.field(self.accuracy, shape=())
            # self.alpha_i[None] = 0.0
            self.beta_i = ti.field(self.accuracy, shape=())
            self.beta_i[None] = 0.0


        print(self.optimizer)
        # BFGS or Newton

        self.T = 0.0
        self.res = ti.Vector([res[0], res[1]])
        # self.global_E = 0.0
        # self.prev_global_E = 0.0
        self.alpha = ti.field(self.accuracy, shape=())
        self.alpha_dec = 0.6
        self.global_alpha_min = 0.00001

        self.global_E = ti.field(self.accuracy, shape=())
        self.prev_global_E = ti.field(self.accuracy, shape=())

        # self.res = res
        self.n_neigbors = ti.field(ti.i32, shape=())
        self.n_crafts = ti.field(ti.i32, shape=())
        self.n_obstacles = ti.field(ti.i32, shape=())
        self.n_ao = ti.field(ti.i32, shape=())

        self.craft_type = ti.field(dtype=ti.i32)
        self.craft_camp = ti.field(dtype=ti.i32)

        self.x = ti.Vector.field(self.dim, dtype=self.accuracy)
        self.v = ti.Vector.field(self.dim, dtype=self.accuracy)
        self.gradient = ti.Vector.field(self.dim, dtype=self.accuracy)
        self.new_v = ti.Vector.field(self.dim, dtype=self.accuracy)
        self.new_pos = ti.Vector.field(self.dim, dtype=self.accuracy)

        self.obstacle_pos = ti.Vector.field(self.dim, dtype=self.accuracy)
        self.obstacle_r = ti.field(dtype=self.accuracy)

        # self.diag = ti.Matrix.field(self.dim, self.dim, dtype=self.accuracy)

        self.length = ti.field(dtype=self.accuracy)
        self.width = ti.field(dtype=self.accuracy)
        self.half_slide = ti.field(dtype=self.accuracy)
        self.angle = ti.field(dtype=self.accuracy)

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
        #
        # self.occupied_num    = ti.field(dtype=ti.i32, shape=())
        # self.occupied_index  = ti.field(dtype=ti.i32)

        self.blockSize       = ti.Vector.field(self.dim, dtype=ti.i32, shape=(1))
        self.min_boundary    = ti.Vector.field(self.dim, dtype=self.accuracy, shape=(1))
        self.max_boundary    = ti.Vector.field(self.dim, dtype=self.accuracy, shape=(1))

        self.maxNeighbourPair= self.n_crafts[None] * 1024

        # self.candidate_neighbor   = ti.types.ndarray(dtype=ti.math.vec2, dim=self.maxNeighbourPair)
        self.candidate_neighbor   = ti.Vector.field(self.dim, dtype=ti.i32)
        self.diag = ti.Matrix.field(self.dim, self.dim, dtype=self.accuracy)


        # self.hash_grid = ti.root.dynamic(ti.i, self.max_num, chunk_size=None)
        # self.hash_neighbour = ti.root.dynamic(ti.ij, self.max_num, chunk_size=None)
        # Hash grids for aa
        self.aa_root = ti.root
        self.aa_root.dense(ti.i, self.max_num).place(self.gridCount)
        self.aa_root.dense(ti.ij, (self.max_num, self.maxInGrid)).place(self.grid)
        self.aa_root.dynamic(ti.i, self.maxNeighbourPair).place(self.candidate_neighbor, self.diag)
        # ti.root.dynamic(ti.i, self.maxAOPair).place(self.candidate_ao, self.diag_ao)


        # ti.root.dense(ti.i, self.n_crafts[None]).place(self.neighborCount)
        # ti.root.dense(ti.ij, (self.n_crafts[None], self.maxNeighbour)).place(self.neighbor)
        #TODO: Using ti.root.pointer to activate sparse architectures

    def setup_ao_grid_gpu(self):
        self.ao_gridCount = ti.field(dtype=ti.i32)
        self.ao_grid = ti.field(dtype=ti.i32)

        self.maxAOPair = self.n_crafts[None] * self.n_obstacles[None]

        self.candidate_ao = ti.Vector.field(self.dim, dtype=ti.i32)
        self.diag_ao = ti.Matrix.field(self.dim, self.dim, dtype=self.accuracy)

        # Hash grids for ao
        self.ao_root = ti.root
        self.ao_root.dense(ti.i, self.max_num).place(self.ao_gridCount)
        self.ao_root.dense(ti.ij, (self.max_num, self.n_obstacles[None])).place(self.ao_grid)
        self.ao_root.dynamic(ti.i, self.maxAOPair).place(self.candidate_ao, self.diag_ao)
        # ti.root.dynamic(ti.i, 1000).place(self.occupied_index)

        self.occupied_indexes = ti.field(ti.i32)
        ti.root.dynamic(ti.i, self.maxAOPair).place(self.occupied_indexes)

    def setup_line_obstacles(self, input_list):
        '''
        preprocess it in python scope
        Input of the vertices should be like:
        [[[x1, y1],
          [x2, y2],
          ```
          ], ```
          ]

        '''
        array_list = []
        array_num = len(input_list)
        for i in range(array_num):
            sliced_array = input_list[i]
            node_num = len(sliced_array)
            for j in range(node_num):
                if j != node_num - 1:
                    array_list.append([sliced_array[j], sliced_array[j+1]])
                else:
                    array_list.append([sliced_array[j], sliced_array[0]])

        vertice_num = len(array_list)
        self.n_obstacles[None] = vertice_num
        print(self.n_obstacles[None])

        self.start_np = np.zeros((vertice_num, self.dim))
        self.end_np = np.zeros((vertice_num, self.dim))
        #These two numpy arrays are for rendering in taichi gui

        self.obstacle_v = ti.Vector.field(self.dim, dtype=self.accuracy)
        ti.root.dense(ti.ij, (vertice_num, self.dim)).place(self.obstacle_v)
        for k in range(vertice_num):
            for l in range(self.dim):
                self.obstacle_v[k, l] = ti.Vector([array_list[k][l][0], array_list[k][l][1]])
                self.start_np[k, l] = array_list[k][0][l]
                self.end_np[k, l] = array_list[k][1][l]
        #the cell in obstacle_v should be a vector of the coordinate of vertices,
        #shape of it should be [n_lines, 2-dim]

        self.start_np /= self.res[0]
        self.end_np /= self.res[0]


    @ti.func
    def return_projection_point(self, v0, v1, p):
        #normal point to outer:
        #for example: start(0,0) end (3,0), normal:[0, 1]
        #projection is weighted, for example ,if in line, 0<projection<1
        d = v1 - v0
        # v = ti.Vector([-d.y, d.x]) / tm.length(d)
        projection = (p - v0).dot(d) / d.dot(d)
        return d, projection

    @ti.func
    def return_distance(self, v0, v1, p):
        d = v1 - v0
        # v = ti.Vector([-d.y, d.x]) / tm.length(d)
        projection = (p - v0).dot(d) / d.dot(d)
        p_point = v0 + projection * d
        return tm.length(p - p_point)

    @ti.kernel
    def init_obstacle_grid(self):
        #from starting grid[i,j] to ending[m,n], find each grids' middle points' distances to the line
        #if the distance is smaller than searchR + 0.5 * hashgrids' length. Then the hash grid is occupied
        #by the obstacle
        valid_distance = self.searchR + 0.7 * self.GridR
        print('valid_dis', valid_distance)
        for i in range(self.n_obstacles[None]):
            #i is the index of the lines
            v0 = self.obstacle_v[i, 0]
            v1 = self.obstacle_v[i, 1]
            indexV0 = ti.cast(v0 * self.invGridR, ti.i32)
            indexV1 = ti.cast(v1 * self.invGridR, ti.i32)

            v0x, v0y = indexV0[0], indexV0[1]
            v1x, v1y = indexV1[0], indexV1[1]

            x_min = ti.min(v0x, v1x)
            y_min = ti.min(v0y, v1y)

            x_max = ti.max(v0x, v1x)
            y_max = ti.max(v0y, v1y)

            print("iiiii", i, indexV0, indexV1)

            for m in range(x_min - 1, x_max + 2):
                for n in range(y_min - 1, y_max + 2):
                    vmx = (ti.cast(m, self.accuracy) + 0.5) * self.GridR
                    vmy = (ti.cast(n, self.accuracy) + 0.5) * self.GridR

                    indexM = ti.Vector([m, n])

                    middle_point = ti.Vector([vmx, vmy])
                    distance = self.return_distance(v0, v1, middle_point)
                    print('ith_dis', i, distance)

                    if distance < valid_distance:
                        hash_index = self.get_cell_hash(indexM)
                        old = ti.atomic_add(self.ao_gridCount[hash_index], 1)
                        # print('count', i, self.ao_gridCount[hash_index])
                        self.ao_grid[hash_index, old] = i
                        print('ith indices', i)
                        self.occupied_indexes.append(hash_index)
                        print(hash_index)



    @ti.kernel
    def find_ao_neighbour(self):
        self.n_ao[None] = 0
        for i in self.occupied_indexes:
            k = 0
            new_pair = 0
            indexes = self.occupied_indexes[i]
            # print('index:', indexes)
            while k < self.ao_gridCount[indexes]:
                # print('k:', k)
                v_id = self.ao_grid[indexes, k]
                j = 0
                while j < self.gridCount[indexes]:
                    p_id = self.grid[indexes, j]
                    if p_id > -1:
                        print("ith_indice", i)
                        new_pair = ti.atomic_add(self.n_ao[None], 1)
                        self.candidate_ao[new_pair] = ti.Vector([v_id, p_id])
                    j += 1
                k += 1

        # self.n_ao[None] -= 1
        print("AOs:", self.n_ao[None])

    @ti.func
    def clog(self, d, d0):
        valLog = tm.log(d / d0)
        valLogC = valLog * (d - d0)
        relD = (d - d0) / d

        D = - self.coef * (2 * valLogC + (d - d0) * relD)
        E = - valLogC * (d - d0) * self.coef

        return E, D

    @ti.func
    def clog_E(self, d, d0):
        valLog = tm.log(d / d0)
        valLogC = valLog * (d - d0)

        E = - valLogC * (d - d0) * self.coef

        return E


    @ti.kernel
    def compute_gradAO(self):
        for i in range(self.n_ao[None]):
            v_id = self.candidate_ao[i][0]
            p_id = self.candidate_ao[i][1]
            print('vid:', v_id)
            print('pid:', p_id)

            v0 = self.obstacle_v[v_id, 0]
            v1 = self.obstacle_v[v_id, 1]
            p = self.x[p_id]
            radsq = self.width[p_id] ** 2

            obsVec = v1 - v0
            relpos0 = v0 - p
            relpos1 = v1 - p

            lensq = obsVec.dot(obsVec)
            s = - relpos0.dot(obsVec) / lensq

            if s < 0:
                dsitsq0 = relpos0.dot(relpos0)
                if dsitsq0 < radsq + self.searchR:
                    E, D = self.clog(dsitsq0-radsq, self.searchR)
                    self.prev_global_E[None] += E
                    self.gradient[p_id] -= D * 2 * relpos0 * 10
            elif s > 1:
                dsitsq1 = relpos1.dot(relpos1)
                if dsitsq1 < radsq + self.searchR:
                    E, D = self.clog(dsitsq1-radsq, self.searchR)
                    self.prev_global_E[None] += E
                    self.gradient[p_id] -= D * 2 * relpos1 * 10
            else:
                v = ti.Vector([-obsVec.y, obsVec.x]) / tm.sqrt(lensq)
                dist = relpos0.dot(v)
                print('dist:', dist)
                distsq = dist ** 2
                if distsq < radsq + self.searchR:
                    E, D = self.clog(distsq - radsq, self.searchR)
                    print("D:", D)
                    self.prev_global_E[None] += E
                    self.gradient[p_id] -= D * 2 * v * 10

                    print(self.gradient[p_id])


    def setup_hessian_builder(self):
        # self.partial_hessian = ti.Matrix.field(self.dim, self.dim, dtype=self.accuracy)
        # self.craft_hessian = ti.root.dense(ti.ij, (self.n_crafts[None], self.n_crafts[None])).place(self.partial_hessian)
        if self.optimizer == 'L-BFGS':
            # self.Global_Hessian = ti.ndarray(self.accuracy,
            #                                  shape=(self.dim * self.n_crafts[None], self.dim * self.n_crafts[None]))
            # self.Global_Identity = ti.ndarray(self.accuracy,
            #                                  shape=(self.dim * self.n_crafts[None], self.dim * self.n_crafts[None]))
            # self.Global_Identity.fill(0)
            # self.init_Identity()

            # self.Global_Grad = ti.ndarray(self.accuracy, shape=(self.m_iters, self.dim * self.n_crafts[None]))
            # self.Global_Value = ti.ndarray(self.accuracy, shape=(self.m_iters, self.dim * self.n_crafts[None]))
            self.rho_i   = ti.field(dtype=self.accuracy)
            self.alpha_i = ti.field(dtype=self.accuracy)
            self.y_i     = ti.field(dtype=self.accuracy)
            self.s_i     = ti.field(dtype=self.accuracy)

            self.lbfgs_loop = ti.root.dense(ti.i, self.max_iters)
            self.lbfgs_loop.place(self.alpha_i)
            self.lbfgs_loop.place(self.rho_i)

            self.lbfgs_loop.dense(ti.j, self.dim * self.n_crafts[None]).place(self.s_i)
            self.lbfgs_loop.dense(ti.j, self.dim * self.n_crafts[None]).place(self.y_i)

            self.alpha_i.fill(0)
            self.rho_i.fill(0)
            self.s_i.fill(0)
            self.y_i.fill(0)


        elif self.optimizer == 'Newton':
            self.Global_Hessian = ti.linalg.SparseMatrixBuilder(self.dim * self.n_crafts[None], self.dim * self.n_crafts[None], max_num_triplets=100000)
            self.Global_Grad    = ti.ndarray(self.accuracy, self.dim * self.n_crafts[None])

        elif self.optimizer == 'BFGS':
            print('here')
            self.Global_Hessian = ti.ndarray(self.accuracy, shape=(self.dim * self.n_crafts[None], self.dim * self.n_crafts[None]))
            # self.test_id()
            print('here0')
            self.Global_Grad    = ti.ndarray(self.accuracy, self.dim * self.n_crafts[None])
            print('here1')
            self.Global_Value   = ti.ndarray(self.accuracy, self.dim * self.n_crafts[None])

        else:
            pass

    @ti.kernel
    def clear_y_s_rho(self):
        self.alpha_i.fill(0)
        self.rho_i.fill(0)
        self.s_i.fill(0)
        self.y_i.fill(0)


    @ti.kernel
    def init_Identity(self):
        for i in range(self.dim * self.n_crafts[None]):
            self.Global_Identity[i, i] = 1.0

    # @ti.kernel
    # def init_occupied_grids(self):
    #     #Find obstacles occupied grids; circle obstacles
    #     self.occupied_num[None] = 0
    #     self.occupied_index.deactive()
    #     for i in range(self.n_obstacles[None]):
    #         sqrt_r = tm.sqrt(self.obstacle_r[i])
    #         index_x = self.obstacle_pos[0]
    #         index_y = self.obstacle_pos[1]
    #
    #         index_l = ti.cast((index_x - sqrt_r) * self.invGridR, ti.i32) - 1
    #         index_r = ti.cast((index_x + sqrt_r) * self.invGridR, ti.i32) + 1
    #         index_d = ti.cast((index_y - sqrt_r) * self.invGridR, ti.i32) - 1
    #         index_u = ti.cast((index_y + sqrt_r) * self.invGridR, ti.i32) + 1
    #
    #         for m in range(index_l, index_r + 1):
    #             for n in ti.static(range(index_d, index_u + 1)):
    #                 indexO = ti.Vector([m, n])
    #                 hash_index = self.get_cell_hash(indexO)
    #                 old = ti.atomic_add(self.ao_gridCount[hash_index], 1)
    #                 self.ao_grid[hash_index, old] = i
    #                 self.occupied_num[None] += 1
    #                 self.occupied_index.append(hash_index)



    def init_env(self, craft_type, craft_camp, x, length, width, input_list):

        self.n_crafts[None] = 0
        # self.init_levelsets()
        self.craft_type_np = craft_type
        self.craft_camp_np = craft_camp
        self.length_np     = length
        self.width_np      = width
        self.x_np          = x

        self.add_crafts_kernel(self.craft_type_np, self.craft_camp_np, self.x_np, self.length_np, self.width_np)
        self.setup_line_obstacles(input_list)
        self.setup_ao_grid_gpu()
        self.init_ao_grid()
        self.init_obstacle_grid()
        # Obstacles' occupied grids will not be changed

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
            self.v[i]          = ti.Vector.zero(self.accuracy, self.dim)
            self.gradient[i]   = ti.Vector.zero(self.accuracy, self.dim)
            self.new_pos[i]    = ti.Vector.zero(self.accuracy, self.dim)

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
    def init_ao_grid(self):
        for i, j in self.ao_grid:
            self.ao_grid[i, j] = -1
            self.ao_gridCount[i] = 0
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

        # self.n_neigbors[None] -= 1


    # @ti.kernel
    # def find_ao_pair(self):
    #     # find Agent-Obstacle pairs
    #     self.n_ao[None] = 0
    #     for hash_index in self.occupied_index:
    #         for m in ti.static(range(self.ao_gridCount[hash_index])):
    #             #Obstacle index m
    #             k = 0
    #             while k < self.gridCount[hash_index]:
    #                 j = self.grid[hash_index, k]
    #                 new_pair = ti.atomic_add(self.n_ao[None], 1)
    #                 self.candidate_ao[new_pair] = ti.Vector([m, j])
    #             k += 1
    #
    #     # self.n_ao[None] -= 1


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
            velr = self.v[a_id] - self.v[b_id]
            margin = - self.width[a_id] * self.width[a_id] * 4
            d0 = self.searchR * 0.1
            for j in ti.static(ti.ndrange(self.dim)):
                margin += ab[j] ** 2
                d0 += (velr[j] ** 2) * 5
            # margin = ab.squaredNorm()
            if margin < d0:
                valLog = tm.log(margin / d0)
                valLogC = valLog * (margin - d0)
                relD = 1 - d0 / margin
                D = - self.coef * (2 * valLogC + (margin - d0) * relD)

                self.gradient[a_id] += D * 2 * ab
                self.gradient[b_id] -= D * 2 * ab
                self.prev_global_E[None] -= valLogC * (margin - d0) * self.coef

            # print("grad_a:", self.gradient[a_id])
        # print('prev_E:', self.prev_global_E[None])

                # #Hessian Computation Part
                # DD = -(4 * relD - relD * relD + 2 * valLog) * self.coef
                # # mat1 = 4 * DD * ab * ab
                # # print(mat1)
                # ab0 = ab[0]
                # ab1 = ab[1]
                # mat1 = 4 * DD * tm.mat2([ab0 * ab0, ab0 * ab1], [ab0 * ab1, ab1 * ab1])
                # diag = D * 2 * ti.Matrix.identity(self.accuracy, self.dim) + mat1.transpose()
                # self.diag[i] = diag
                # # self.partial_hessian[a_id, a_id] += diag
                # # self.partial_hessian[b_id, b_id] += diag
                # # self.partial_hessian[a_id, b_id] -= diag
                # # self.partial_hessian[b_id, a_id] -= diag.transpose()

    # @ti.kernel
    # def compute_gradAO(self):
    #     #WASTED
    #     for i in range(self.n_ao[None]):
    #         o_id = self.candidate_ao[i][0]
    #         a_id = self.candidate_ao[i][1]
    #
    #         ao =  - self.obstacle_pos[o_id] + self.x[a_id]
    #
    #         margin = (ao.length - self.width[a_id] - self.obstacle_r[o_id]) ** 2
    #         d0 = self.searchR - self.width[a_id]
    #         if margin < d0:
    #             valLog = tm.log(margin / d0)
    #             valLogC = valLog * (margin - d0)
    #             relD = 1 - d0 / margin
    #             D = - self.coef * (2 * valLogC + (margin - d0) * relD)
    #
    #             scale = self.width[a_id] / self.obstacle_r[o_id]
    #
    #             self.gradient[a_id] += D * 2 * ao * scale
    #             self.prev_global_E[None] -= valLogC * (margin - d0) * self.coef
    #
    #             # # Hessian Computation Part
    #             # DD = -(4 * relD - relD * relD + 2 * valLog) * self.coef
    #             # # mat1 = 4 * DD * ab * ab
    #             # # print(mat1)
    #             # ao0 = ao[0] * scale
    #             # ao1 = ao[1] * scale
    #             # mat1 = 4 * DD * tm.mat2([ao0 * ao0, ao0 * ao1], [ao0 * ao1, ao1 * ao1])
    #             # diag = D * 2 * ti.Matrix.identity(self.accuracy, self.dim) + mat1.transpose()
    #             # self.diag_ao[i] = diag



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
            velr = self.v[a_id] - self.v[b_id]
            margin = - self.width[a_id] * self.width[a_id] * 4.0
            d0 = self.searchR
            for j in ti.static(ti.ndrange(self.dim)):
                margin += ab[j] ** 2
                d0 += (velr[j] ** 2) * 5.0

            if margin < d0:
                valLog = tm.log(margin / d0)
                valLogC = valLog * (margin - d0)

                self.global_E[None] -= valLogC * (margin - d0) * self.coef

    @ti.kernel
    def compute_energy_U_ao(self):
        for i in self.candidate_ao:
            v_id = self.candidate_ao[i][0]
            p_id = self.candidate_ao[i][1]

            v0 = self.obstacle_v[v_id, 0]
            v1 = self.obstacle_v[v_id, 1]
            p = self.x[p_id] + self.new_v[p_id] * self.alpha[None]
            radsq = self.width[p_id] ** 2

            obsVec = v1 - v0
            relpos0 = v0 - p
            relpos1 = v1 - p

            lensq = obsVec.dot(obsVec)
            s = - relpos0.dot(obsVec) / lensq

            if s < 0:
                dsitsq0 = relpos0.dot(relpos0)
                if dsitsq0 < radsq + self.searchR:
                    E = self.clog_E(dsitsq0-radsq, self.searchR)
                    self.global_E[None] += E

            elif s > 1:
                dsitsq1 = relpos1.dot(relpos1)
                if dsitsq1 < radsq + self.searchR:
                    E = self.clog_E(dsitsq1-radsq, self.searchR)
                    self.global_E[None] += E

            else:
                v = ti.Vector([-obsVec.y, obsVec.x]) / tm.sqrt(lensq)
                dist = relpos0.dot(v)
                distsq = dist ** 2
                if distsq < radsq + self.searchR:
                    E = self.clog_E(distsq - radsq, self.searchR)
                    self.global_E[None] += E





    @ti.kernel
    def compute_energy_E(self):
        # global_E = 0.0
        for i in range(self.n_crafts[None]):
            # xx = -self.gradient[i] * self.alpha[None] - self.v[i]*self.dt
            xx = self.new_v[i] * self.alpha[None] - self.v[i] * self.dt
            XX = 0.0

            for j in ti.static(ti.ndrange(self.dim)):
                XX += xx[j] ** 2

            dE = XX / (2 * self.dt * self.dt)
            self.global_E[None] += dE



    # @ti.kernel
    def line_search(self):
        # for i in range(self.n_neigbors[None]):
        self.alpha[None] = 1.0
        print("PREV-E", self.prev_global_E[None])

        while self.alpha[None] > self.global_alpha_min:
            self.global_E[None] = 0.0
            self.compute_energy_E()
            # self.compute_energy_U()
            # self.compute_energy_U_ao()

            print('new_E:', self.global_E[None])

            if self.global_E[None] < self.prev_global_E[None]:
                break
            else:
                self.alpha[None] *= self.alpha_dec

        # return self.alpha[None]

    # @ti.kernel
    # def dynamic_step(self):
    #     #WASTED
    #     #Semi-implicit Euler:
    #     for i in range(self.n_crafts[None]):
    #         self.x[i] += self.v[i] * self.dt

    @ti.kernel
    def update_pos_32(self, new_alpha: ti.f32):
        for i in range(self.n_crafts[None]):
            # self.x[i] -= self.gradient[i] * new_alpha
            self.x[i] += self.new_v[i] * new_alpha

    @ti.kernel
    def update_pos_64(self, new_alpha: ti.f64):
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

        self.Rasterize_Vector(new_direction, self.new_v)

    @ti.kernel
    def direct_solving(self):
        for i in self.new_v:
            self.new_v[i] = -self.gradient[i]

    @ti.kernel
    def assemble_Value(self, x_mat: ti.types.ndarray(), single_x: ti.types.vector(2, dtype=ti.f32)):
        for i in range(self.n_crafts[None]):
            for m in ti.static(range(2)):
                x_mat[2 * i + m] = single_x[i][m]

    def sparse_solving_BFGS(self):
        x_old = self.Global_Value
        grad_old = self.Global_Grad

        self.assemble_Value(self.Global_Value, self.x)
        self.assemble_grad(self.Global_Grad, self.gradient)

        y_k = self.Global_Grad - grad_old
        s_k = self.Global_Value - x_old

        rho_k = 1 / (y_k @ s_k)
        prev_mat_1 = ti.Matrix.identity(self.accuracy, self.dim * self.n_crafts[None]) - rho_k * s_k.outer_product(y_k)
        GH = self.Global_Hessian
        GH = prev_mat_1 @ GH @ prev_mat_1 + rho_k * s_k.outer_product(s_k)

        new_direction = - GH @ self.Global_Grad
        self.Global_Hessian = GH
        self.Rasterize_Vector(new_direction, self.new_v)

    @ti.kernel
    def get_x_s(self, iters: ti.i32):
        for i in range(self.n_crafts[None]):
            scale_value = self.new_v[i] * self.alpha[None]
            # ti.static_print(scale_value[0])
            self.x[i] += scale_value
            for j in ti.static(range(2)):
                self.s_i[iters, 2 * i + j] = scale_value[j]

    @ti.kernel
    def tempora_save_grad(self, iters: ti.i32):
        # yk = gk+1 - gk; unless k=0, gk appears twice in total computation
        for i in range(self.n_crafts[None]):
            if iters > 0:
                for j in ti.static(range(2)):
                    self.y_i[iters, 2 * i + j] -= self.gradient[i][j]
                    self.y_i[iters - 1, 2 * i + j] += self.gradient[i][j]
            else:
                for j in ti.static(range(2)):
                    self.y_i[iters, 2 * i + j] -= self.gradient[i][j]

    @ti.kernel
    def get_rho_scale(self, iters: ti.i32):
        for i in range(self.dim * self.n_crafts[None]):
            self.rho_i[iters] += self.y_i[iters, i] * self.s_i[iters, i]
            self.tmp_value[None] += self.y_i[iters, i] ** 2

        self.tmp_value[None] /= self.rho_i[iters]

    @ti.kernel
    def vector_dot_32(self, iters: ti.i32, vec1: ti.types.vector(2, dtype=ti.f32), vec2: ti.types.vector(2, dtype=ti.f32)):
        for i in range(self.dim * self.n_crafts[None]):
            self.tmp_value[None] += vec1[iters, i] * vec2[iters, i]

    @ti.kernel
    def vector_dot_64(self, iters: ti.i32, vec1: ti.types.vector(2, dtype=ti.f64), vec2: ti.types.vector(2, dtype=ti.f64)):
        for i in range(self.dim * self.n_crafts[None]):
            self.tmp_value[None] += vec1[iters, i] * vec2[iters, i]

    @ti.kernel
    def apply_first_loop_1(self, iters: ti.i32):
        self.alpha_i[iters] = 0.0  # fresh
        for i in range(self.n_crafts[None]):
            for j in ti.static(range(2)):
                self.alpha_i[iters] += self.gradient[i][j] * self.s_i[iters, 2*i+j]
        self.alpha_i[iters] /= self.rho_i[iters]
        # print(self.rho_i[iters])
        # print("alpha:", self.alpha_i[iters])

    @ti.kernel
    def apply_first_loop_2(self, iters: ti.i32):
        for i in range(self.n_crafts[None]):
            for j in ti.static(range(2)):
                self.gradient[i][j] -= self.alpha_i[iters] * self.y_i[iters, 2*i+j]

    @ti.kernel
    def get_r_32(self, scale: ti.f32):
        #TODO: What if initialized inv-H is not gamma*I
        for i in range(self.n_crafts[None]):
            self.gradient[i] *= scale

    @ti.kernel
    def get_r_64(self, scale: ti.f64):
        #TODO: What if initialized inv-H is not gamma*I
        for i in range(self.n_crafts[None]):
            self.gradient[i] *= scale

    @ti.kernel
    def apply_second_loop_1(self, iters: ti.i32):
        for i in range(self.n_crafts[None]):
            for j in ti.static(range(2)):
                self.beta_i[None] += self.y_i[iters, 2*i+j] * self.gradient[i][j]
        self.beta_i[None] /= self.rho_i[iters]

    @ti.kernel
    def apply_second_loop_2(self, iters: ti.i32):
        loop2_scale = self.alpha_i[iters] - self.beta_i[None]
        for i in range(self.n_crafts[None]):
            for j in ti.static(range(2)):
                self.gradient[i][j] += self.s_i[iters, 2*i+j] * loop2_scale

    @ti.kernel
    def set_direction(self):
        for i in self.new_v:
            self.new_v[i] = -self.gradient[i]
            # print(self.new_v[i])


    def sparse_solving_LBFGS(self, iters):
        m_iters = iters if iters < self.m_iters else self.m_iters

        self.tempora_save_grad(iters)
        # real grad already saved

        if m_iters == 0:
            self.direct_solving()
            self.line_search()
        # elif m_iters == 1:
        #     #two-loop recursions
        #     self.tmp_value[None] = 0.0
        #     self.get_rho_scale(iters - 1)
        #     scale = 1 / self.tmp_value[None]
        #
        #     # self.vector_dot(iters-1, self.y_i, self.y_i)
        #     # scale = self.rho_i[iters - 1] / self.tmp_value[None]
        #     # self.help_init_inv_H(scale)
        #
        #     self.apply_first_loop_1(m_iters - 1)
        #     self.apply_first_loop_2(m_iters - 1)
        #
        #     self.get_r()
        #
        #     self.apply_second_loop_1(m_iters - 1)
        #     self.apply_second_loop_2(m_iters - 1)
        #
        #     self.set_direction()
        #     self.line_search()
        else:
            #two-loop recursions
            self.tmp_value[None] = 0.0
            self.get_rho_scale(iters - 1)
            scale = 1 / self.tmp_value[None]

            # self.help_init_inv_H(scale)

            for iter1 in range(iters-1, iters - m_iters - 1, -1):
                self.apply_first_loop_1(iter1)
                self.apply_first_loop_2(iter1)

            if ti.static(self.accuracy == 'f32'):
                self.get_r_32(scale)
            else:
                self.get_r_64(scale)

            for iter2 in range(iters - m_iters, iters):
                self.apply_second_loop_1(iter2)
                self.apply_second_loop_2(iter2)

            self.set_direction()
            self.line_search()

        # if self.alpha[None] > self.global_alpha_min:
        #     self.get_x_s(iters)
        self.get_x_s(iters)
            # self.tempora_save_grad(iters)


    @ti.kernel
    def Rasterize_Vector(self, new_direction: ti.types.ndarray(), new_v: ti.types.vector(2, dtype=ti.f32)):
        for i in range(self.n_crafts[None]):
            for m in ti.static(range(2)):
                new_v[i][m] = new_direction[2 * i + m]

    def single_step(self, step_optimizer=None, iters=None):
        self.global_E[None] = 0.0
        self.prev_global_E[None] = 0.0

        # self.process_action(actions)
        self.init_grid()
        self.insert_grid_pos()
        self.find_neighbour()
        # print(self.candidate_neighbor)

        self.init_grad()
        self.compute_gradAA()

        self.find_ao_neighbour()
        self.compute_gradAO()

        # self.sparse_solving()
        if step_optimizer == 'L-BFGS':
            self.sparse_solving_LBFGS(iters)
        elif step_optimizer == 'Newton': #Need perturbation
            self.sparse_solving()  #BFGS may start after the very first iteration ends to get a better H_0 approximation
        elif step_optimizer == 'BFGS':
            self.sparse_solving_BFGS()
        else:
            self.direct_solving()  #Gradient Descent

        self.line_search()

        #Deactivate dynamic snodes
        # self.candidate_ao.deactivate()
        # self.candidate_neighbor.deactivate()
        # self.update_pos(self.alpha[None])


    def optimize_LBFGS(self, actions):
        iter = 0
        self.clear_y_s_rho()
        self.process_action(actions)

        while iter < self.max_iters:
            self.single_step('L-BFGS', iter)
            iter += 1
            if self.alpha[None] < self.global_alpha_min:
                print("finished:", iter)
                # print(self.alpha[None])
                # craft_info = self.render_info()
                break

        craft_info = self.render_info()
        print("finished:", iter)

        return craft_info



    def optimize_BFGS(self, actions, Newton=False):
        iter = 0
        self.process_action(actions)
        if Newton:
            pass
        else:
            y_k, s_k = self.init_inv_H()
            rho_k = 1 / (y_k @ s_k)
            prev_mat_1 = ti.Matrix.identity(self.accuracy, self.dim * self.n_crafts[None]) - rho_k * s_k.outer_product(y_k)
            GH = self.Global_Hessian
            GH = prev_mat_1 @ GH @ prev_mat_1 + rho_k * s_k.outer_product(s_k)

            new_direction = - GH @ self.Global_Grad
            self.Global_Hessian = GH
            self.Rasterize_Vector(new_direction, self.new_v)

            self.line_search()

            if self.alpha[None] > self.global_alpha_min:
                self.update_pos(self.alpha[None])
            else:
                pass
                # print(iter)
                # break
        iter = 2

        while iter < self.max_iters:
            self.single_step('BFGS')
            iter += 1
            if self.alpha[None] > self.global_alpha_min:
                self.update_pos(self.alpha[None])
            else:
                print("finished:", iter)
                # craft_info = self.render_info()
                break

        craft_info = self.render_info()
        print("finished:", iter)

        return craft_info

    @ti.kernel
    def return_minus(self, out: ti.types.ndarray(), left: ti.types.ndarray(), right: ti.types.ndarray()):
        for i in ti.grouped(left):
            out[i] = left[i] - right[i]

    @ti.kernel
    def vector_array_dot(self, left: ti.types.ndarray(), right: ti.types.ndarray()):
        # value = 0.0
        for i in ti.grouped(left):
            self.tmp_value[None] += left[i] * right[i]

        # return value

    # @ti.kernel
    # def help_init_inv_H(self, scale: self.accuracy):
    #     for i in range(self.dim * self.n_crafts[None]):
    #         self.Global_Hessian[i, i] = scale


    # @ti.kernel
    # def help1(self, y_k: ti.types.ndarray(), s_k: ti.types.ndarray()):
    #     help1 = y_k @ s_k
    #     help2 = ti.math
    #     print(help1)
    #     self.Global_Hessian = ti.Matrix.identity(self.accuracy, self.dim * self.n_crafts[None]) * (y_k @ s_k) / (y_k @ y_k)


    # @ti.kernel
    def init_inv_H(self, Newton=False):
        #If not calculating 'real' inverse Hessian, use GD to perform the very first update.
        self.single_step('GD')
        x_old = self.Global_Value
        grad_old = self.Global_Grad

        self.update_pos(self.alpha[None])

        self.assemble_Value(self.Global_Value, self.x)

        self.global_E[None] = 0.0
        self.prev_global_E[None] = 0.0
        self.init_grad()
        self.compute_gradAA()
        self.assemble_grad(self.Global_Grad, self.gradient)

        # y_k = self.Global_Grad - grad_old
        # s_k = self.Global_Value - x_old
        y_k = ti.ndarray(self.accuracy, self.dim * self.n_crafts[None])
        s_k = ti.ndarray(self.accuracy, self.dim * self.n_crafts[None])
        self.return_minus(y_k, self.Global_Grad, grad_old)
        self.return_minus(s_k, self.Global_Value, x_old)
        self.v1 = ti.field(self.accuracy, shape=())
        # self.v2 = ti.field(self.accuracy, shape=())
        self.v1[None] = 0.0
        # self.v2[None] = 0.0
        self.vector_array_dot(y_k, s_k)
        v1 = self.v1[None]
        self.v1[None] = 0.0
        self.vector_array_dot(y_k, y_k)
        v2 = self.v1[None]
        # print(y_k)
        # print(v2)


        # scale = self.vector_array_dot(y_k, s_k) / self.vector_array_dot(y_k, y_k)
        self.Global_Hessian.fill(0)
        self.help_init_inv_H(v1/v2)
        #
        # self.Global_Hessian = ti.Matrix.identity(self.accuracy, self.dim * self.n_crafts[None]) * (y_k @ s_k) / (y_k @ y_k)
        # self.help1(y_k, s_k)
        return y_k, s_k

    @ti.kernel
    def get_render_info(self, type: ti.types.ndarray(), camp: ti.types.ndarray(), x: ti.types.ndarray(),
                        v: ti.types.ndarray()):
        for i in range(self.n_crafts[None]):
            type[i] = self.craft_type[i]
            camp[i] = self.craft_camp[i]

            for j in ti.static(range(self.dim)):
                x[i, j] = self.x[i][j]
                v[i, j] = self.v[i][j]


    def render_info(self):
        np_x = np.ndarray((self.n_crafts[None], self.dim), dtype=np.float32)
        np_v = np.ndarray((self.n_crafts[None], self.dim), dtype=np.float32)
        np_type = np.ndarray((self.n_crafts[None],), dtype=np.float32)
        np_camp = np.ndarray((self.n_crafts[None],), dtype=np.float32)

        self.get_render_info(np_type, np_camp, np_x, np_v)

        craft_data = {
            'position': np_x,
            'velocity': np_v,
            'type': np_type,
            'camp': np_camp
        }

        return craft_data



    #TODO: Reuse this part when reimplementing vheicles

    # @ti.kernel
    # def get_render_info(self, type:ti.types.ndarray(), camp:ti.types.ndarray(), x:ti.types.ndarray(), v:ti.types.ndarray(), renderx:ti.types.ndarray(), rendery:ti.types.ndarray(), collided:ti.types.ndarray()):
    #     for i in range(self.n_crafts[None]):
    #         normal_angle = tm.atan2(self.v[i][1], self.v[i][0])
    #         rot_mat = tm.rotation2d(normal_angle)
    #
    #         x0 = self.x[i] + ti.Vector([self.length[i]/2, self.width[i]/2]) @ rot_mat
    #         x1 = self.x[i] + ti.Vector([-self.length[i]/2, self.width[i]/2]) @ rot_mat
    #         x2 = self.x[i] + ti.Vector([-self.length[i]/2, -self.width[i]/2]) @ rot_mat
    #         x3 = self.x[i] + ti.Vector([self.length[i]/2, -self.width[i]/2]) @ rot_mat
    #
    #         type[i] = self.craft_type[i]
    #         camp[i] = self.craft_camp[i]
    #         collided[i] = self.collided[i]
    #
    #         for j in ti.static(range(self.dim)):
    #             x[i, j] = self.x[i][j]
    #             v[i, j] = self.v[i][j]
    #             renderx[i * 4, j] = x0[j]
    #             renderx[i * 4 + 1, j] = x1[j]
    #             renderx[i * 4 + 2, j] = x2[j]
    #             renderx[i * 4 + 3, j] = x3[j]
    #
    #             rendery[i * 4, j] = x1[j]
    #             rendery[i * 4 + 1, j] = x2[j]
    #             rendery[i * 4 + 2, j] = x3[j]
    #             rendery[i * 4 + 3, j] = x0[j]
    #
    # def render_info(self):
    #     np_x = np.ndarray((self.n_crafts[None], self.dim), dtype=np.float32)
    #     np_v = np.ndarray((self.n_crafts[None], self.dim), dtype=np.float32)
    #     np_type = np.ndarray((self.n_crafts[None], ), dtype=np.float32)
    #     np_camp = np.ndarray((self.n_crafts[None], ), dtype=np.float32)
    #     np_collide = np.ndarray((self.n_crafts[None],), dtype=np.int32)
    #
    #     #For rendering rectangles
    #     np_renderx = np.ndarray((4 * self.n_crafts[None], self.dim), dtype=np.float32)
    #     np_rendery = np.ndarray((4 * self.n_crafts[None], self.dim), dtype=np.float32)
    #     self.get_render_info(np_type, np_camp, np_x, np_v, np_renderx, np_rendery, np_collide)
    #
    #     craft_data = {
    #         'position': np_x,
    #         'velocity': np_v,
    #         'type': np_type,
    #         'camp': np_camp,
    #         'RenderX': np_renderx,
    #         'RenderY': np_rendery,
    #         'Collided': np_collide
    #     }
    #
    #     return craft_data

    @ti.kernel
    def copy_dynamic_nd(self, np_x: ti.types.ndarray(), input_x: ti.template()):
        for i in self.x:
            for j in ti.static(range(self.dim)):
                np_x[i, j] = input_x[i][j]

    @ti.kernel
    def copy_dynamic(self, np_x: ti.types.ndarray(), input_x: ti.template()):
        for i in self.x:
            np_x[i] = input_x[i]





