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
        self.max_iters = 10

        self.substeps = substeps
        self.ep_length = ep_length
        self.global_substep = 0
        self.dt = 1/(control_freq * substeps)

        self.GridR = GridR
        self.invGridR = 1 / GridR
        self.searchR = d0
        self.neighbor_grids = int(d0/GridR + 0.5)
        self.coef = 1.0

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

        self.intersection_check_result = ti.field(ti.i32, shape=())
        self.intersection_check_result[None] = 0
        self.alpha = ti.field(self.accuracy, shape=())
        self.alpha_dec = 0.6
        self.global_alpha_min = 1e-8

        self.global_E = ti.field(self.accuracy, shape=())
        self.prev_global_E = ti.field(self.accuracy, shape=())
        self.global_grad_norm = ti.field(self.accuracy, shape=())

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
        self.temp_grad = ti.Vector.field(self.dim, dtype=self.accuracy)

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

        self.crafts.place(self.craft_type, self.craft_camp, self.x, self.new_pos, self.v, self.gradient, self.temp_grad, self.new_v, self.length, self.width, self.half_slide, self.angle, self.collided)
        self.obstacles.place(self.obstacle_pos, self.obstacle_r)

        S = ti.root.dynamic(ti.i, 1024, chunk_size=None)
        self.active_agents = ti.field(int)
        S.place(self.active_agents)
        self.n_active = ti.field(dtype=self.accuracy, shape=())
        self.armijo = ti.field(dtype=self.accuracy, shape=())

        # self.craft_hessian = ti.root.dense


    def setup_grid_gpu(self):
        # Whenever add agents, reset grid
        # ao means agent-obstacle
        self.gridCount       = ti.field(int)
        self.grid            = ti.field(int)

        self.blockSize       = ti.Vector.field(self.dim, dtype=ti.i32, shape=(1))
        self.min_boundary    = ti.Vector.field(self.dim, dtype=self.accuracy, shape=(1))
        self.max_boundary    = ti.Vector.field(self.dim, dtype=self.accuracy, shape=(1))

        self.maxNeighbourPair= 256 * 1024

        # self.candidate_neighbor   = ti.types.ndarray(dtype=ti.math.vec2, dim=self.maxNeighbourPair)
        self.candidate_neighbor   = ti.Vector.field(self.dim, dtype=ti.i32)
        self.diag = ti.Matrix.field(self.dim, self.dim, dtype=self.accuracy)

        grid_block_size = 128
        leaf_block_size = 16

        self.aa_root = ti.root
        ###Test sparse -- failed
        # self.a1 = self.aa_root.pointer(ti.i, self.max_num // grid_block_size)
        # self.a2 = self.aa_root.pointer(ti.ij, self.max_num // grid_block_size) #currently max_num equals max in grid
        # self.a1.pointer(ti.i, grid_block_size // leaf_block_size).place(self.gridCount)
        # self.a2.pointer(ti.ij, grid_block_size // leaf_block_size).place(self.grid)

        ###Test Dynamic
        ti.root.dynamic(ti.i, self.max_num).place(self.gridCount)
        ti.root.dense(ti.i, self.max_num).dynamic(ti.j, self.maxInGrid).place(self.grid)


        # self.aa_root.dense(ti.i, self.max_num).place(self.gridCount)
        # self.aa_root.dense(ti.ij, (self.max_num, self.maxInGrid)).place(self.grid)
        self.aa_root.dynamic(ti.i, self.maxNeighbourPair).place(self.candidate_neighbor, self.diag)
        #TODO: Using ti.root.pointer to activate sparse architectures

    def setup_ao_grid_gpu(self):
        self.ao_gridCount = ti.field(dtype=ti.i32)
        self.ao_grid = ti.field(dtype=ti.i32)

        self.maxAOPair = 256 * self.n_obstacles[None]

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
        # print('valid_dis', valid_distance)
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

            # print("iiiii", i, indexV0, indexV1)

            for m in range(x_min - 1, x_max + 2):
                for n in range(y_min - 1, y_max + 2):
                    vmx = (ti.cast(m, self.accuracy) + 0.5) * self.GridR
                    vmy = (ti.cast(n, self.accuracy) + 0.5) * self.GridR

                    indexM = ti.Vector([m, n])

                    middle_point = ti.Vector([vmx, vmy])
                    distance = self.return_distance(v0, v1, middle_point)
                    # print('ith_dis', i, distance)

                    if distance < valid_distance:
                        hash_index = self.get_cell_hash(indexM)
                        old = ti.atomic_add(self.ao_gridCount[hash_index], 1)
                        # print('count', i, self.ao_gridCount[hash_index])
                        self.ao_grid[hash_index, old] = i
                        # print('ith indices', i)
                        self.occupied_indexes.append(hash_index)
                        # print(hash_index)



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
                        # print("ith_indice", i)
                        new_pair = ti.atomic_add(self.n_ao[None], 1)
                        self.candidate_ao[new_pair] = ti.Vector([v_id, p_id])
                    j += 1
                k += 1

        # self.n_ao[None] -= 1
        # print("AOs:", self.n_ao[None])

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
            # print('vid:', v_id)
            # print('pid:', p_id)

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
                    self.gradient[p_id] -= D * 2 * relpos0
            elif s > 1:
                dsitsq1 = relpos1.dot(relpos1)
                if dsitsq1 < radsq + self.searchR:
                    E, D = self.clog(dsitsq1-radsq, self.searchR)
                    self.prev_global_E[None] += E
                    self.gradient[p_id] -= D * 2 * relpos1
            else:
                v = ti.Vector([-obsVec.y, obsVec.x]) / tm.sqrt(lensq)
                dist = relpos0.dot(v)
                # print('dist:', dist)
                distsq = dist ** 2
                if distsq < radsq + self.searchR:
                    E, D = self.clog(distsq - radsq, self.searchR)
                    # print("D:", D)
                    self.prev_global_E[None] += E
                    self.gradient[p_id] -= D * 2 * v

                    # print(self.gradient[p_id])

    @ti.kernel
    def deactivate_helper(self):
        for i in range(self.max_iters):
            self.s_i[i].deactivate()
            self.y_i[i].deactivate()


    def setup_hessian_builder(self):
        if self.optimizer == 'L-BFGS':
            self.rho_i   = ti.field(dtype=self.accuracy)
            self.alpha_i = ti.field(dtype=self.accuracy)
            self.y_i     = ti.Vector.field(self.dim, dtype=self.accuracy)
            self.s_i     = ti.Vector.field(self.dim, dtype=self.accuracy)

            self.lbfgs_loop = ti.root.dense(ti.i, self.max_iters)
            self.lbfgs_loop.place(self.alpha_i)
            self.lbfgs_loop.place(self.rho_i)

            self.lbfgs_loop.dynamic(ti.j, 1024).place(self.s_i)
            self.lbfgs_loop.dynamic(ti.j, 1024).place(self.y_i)

            self.alpha_i.fill(0)
            self.rho_i.fill(0)

            self.deactivate_helper()
            # self.s_i.fill(0)
            # self.y_i.fill(0)


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
    def clear_y_s_rho_1(self):
        self.alpha_i.fill(0)
        self.rho_i.fill(0)

    @ti.kernel
    def clear_y_s_rho_2(self):
        for i in range(self.max_iters):
            self.s_i[i].deactivate()
            self.y_i[i].deactivate()

    def clear_y_s_rho(self):
        self.clear_y_s_rho_1()
        self.clear_y_s_rho_2()



    @ti.kernel
    def init_Identity(self):
        for i in range(self.dim * self.n_crafts[None]):
            self.Global_Identity[i, i] = 1.0


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

    @ti.kernel
    def add_crafts_v_kernel(self,
                           craft_type: ti.types.ndarray(),
                           craft_camp: ti.types.ndarray(),
                           x         : ti.types.ndarray(),
                           v         : ti.types.ndarray(),
                           length    : ti.types.ndarray(),
                           width     : ti.types.ndarray()):

        new_crafts = craft_type.shape[0]

        for i in range(self.n_crafts[None], self.n_crafts[None] + new_crafts):
            np_id = i - self.n_crafts[None]
            self.craft_type[i] = craft_type[np_id]
            self.craft_camp[i] = craft_camp[np_id]
            self.length[i]     = length[np_id]
            self.width[i]      = width[np_id]
            self.half_slide[i] = 0.0
            self.angle[i]      = 0.0
            self.collided[i]   = 0
            # self.v[i]          = ti.Vector.zero(self.accuracy, self.dim)
            self.gradient[i]   = ti.Vector.zero(self.accuracy, self.dim)
            self.new_pos[i]    = ti.Vector.zero(self.accuracy, self.dim)

            for j in ti.static(range(self.dim)):
                self.x[i][j] = x[np_id, j]
                self.v[i][j] = v[np_id, j]

            self.active_agents.append(i)

        self.n_crafts[None] += new_crafts

    def add_crafts_dynamic(self, craft_type, craft_camp, x, v, length, width):
        self.add_crafts_v_kernel(craft_type, craft_camp, x, v, length, width)


    # @ti.kernel
    # def predefine_v(self):


    def register_terminals(self, terminals):
        num_t = terminals.shape[0]
        self.terminal = ti.Vector.field(self.dim, dtype=self.accuracy)
        ti.root.dense(ti.i, num_t).place(self.terminal)

        for i in range(num_t):
            self.terminal[i] = ti.Vector([terminals[i, 0], terminals[i, 1]])

    @ti.kernel
    def check_activity(self):
        self.active_agents.deactivate()
        for i in range(self.n_crafts[None]):
            T_id = self.craft_camp[i]
            T = self.terminal[T_id]

            dis = self.x[i] - T
            base = dis @ dis
            # print('dis', dis, 'base', base)

            if base > 1000.0:
                self.active_agents.append(i)


    @ti.kernel
    def init_grid(self):
        # self.grid.fill(-1)
        # self.gridCount.fill(0)
        # self.a1.deactivate_all()
        # self.a2.deactivate_all()
        # self.aa_root.deactivate_all()
        # self.grid.deactivate()
        self.gridCount.deactivate()
        for i in range(self.max_num):
            self.grid[i].deactivate()
        # for i, j in self.grid:
        #     self.gridCount[i].deactivate()
        #     self.grid[i,j].deactivate()

    @ti.kernel
    def init_ao_grid(self):
        # for i, j in self.ao_grid:
        #     self.ao_grid[i, j] = -1
        #     self.ao_gridCount[i] = 0
        self.ao_grid.fill(-1)
        self.ao_gridCount.fill(0)

    @ti.kernel
    def insert_grid_pos(self):
        # for i in range(self.n_crafts[None]):
        for Id in self.active_agents:
            i = self.active_agents[Id]
            indexV = ti.cast(self.x[i] * self.invGridR, ti.i32)
            # print(indexV)
            hash_index = self.get_cell_hash(indexV)
            # print(hash_index)
            old = ti.atomic_add(self.gridCount[hash_index], 1)
            # print(old)
            #################################
            #Once tested, no need to print 'exceed'
            # if old > self.maxInGrid - 1:
            #     print("exceed grid", old)
            #     self.gridCount[hash_index] = self.maxInGrid
            # else:
            self.grid[hash_index, old] = i

    @ti.kernel
    def find_neighbour(self):
        # find neighbours and add neighbor pairs to a dynamic field
        self.n_neigbors[None] = 0
        # for i in range(self.n_crafts[None]):
        for Id in self.active_agents:
            i = self.active_agents[Id]
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

    @ti.kernel
    def intersection_check(self):
        # for i in range(self.n_neigbors[None]):
        for Id in self.active_agents:
            i = self.active_agents[Id]
            a_id = self.candidate_neighbor[i][0]
            b_id = self.candidate_neighbor[i][1]

            a = self.x[a_id]
            b = self.x[b_id]

            RHS = b - a
            LHS = tm.mat2([self.new_v[a_id][0], self.new_v[b_id][0]],
                          [self.new_v[a_id][1], self.new_v[b_id][1]]) * self.alpha[None]

            # if LHS.determinant() < 1e-4:
            #     self.intersection_check_result[None] += 0
            if LHS.determinant() > 1e-4:
                s = LHS.inverse() @ RHS
                # print(s)
                s0 = s[0]
                s1 = s[1]
                self.intersection_check_result[None] += int(0 <= s0 <= 1 and 0 <= s1 <= 1)


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


    @ti.kernel
    def init_grad(self):
        #Given x and v, calculate E
        # for i in range(self.n_crafts[None]):
        for Id in self.active_agents:
            i = self.active_agents[Id]
            # print(Id, i)
            self.gradient[i] = -self.v[i] / self.dt
            xx = - self.v[i]*self.dt
            XX = 0.0

            for j in ti.static(ti.ndrange(self.dim)):
                XX += xx[j] ** 2

            dE = XX / (2.0 * self.dt * self.dt)
            self.prev_global_E[None] += dE

    @ti.kernel
    def save_grad(self):
        for Id in self.active_agents:
            i = self.active_agents[Id]
            self.temp_grad[i] = self.gradient[i]


    @ti.kernel
    def compute_gradAA(self):
        for i in range(self.n_neigbors[None]):
            a_id = self.candidate_neighbor[i][0]
            b_id = self.candidate_neighbor[i][1]
            ab = self.x[a_id] - self.x[b_id]
            velr = self.v[a_id] - self.v[b_id]
            margin = - self.width[a_id] * self.width[a_id] * 4
            d0 = self.searchR
            for j in ti.static(ti.ndrange(self.dim)):
                margin += ab[j] ** 2
                d0 += (velr[j] ** 2)
            # margin = ab.squaredNorm()
            if margin < d0:
                valLog = tm.log(margin / d0)
                valLogC = valLog * (margin - d0)
                relD = 1 - d0 / margin
                D = - self.coef * (2 * valLogC + (margin - d0) * relD)

                self.gradient[a_id] += D * 2 * ab
                self.gradient[b_id] -= D * 2 * ab
                self.prev_global_E[None] -= valLogC * (margin - d0) * self.coef

    @ti.kernel
    def check_grad_helper(self):
        for i in range(self.n_crafts[None]):
            print('GD check:', self.gradient[i])


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
        aa = self.alpha[None]
        for i in range(self.n_neigbors[None]):
            a_id = self.candidate_neighbor[i][0]
            b_id = self.candidate_neighbor[i][1]
            # ab = self.x[a_id] - self.x[b_id] + (-self.gradient[a_id] + self.gradient[b_id]) * self.alpha[None]
            ab = self.x[a_id] - self.x[b_id] + (self.new_v[a_id] - self.new_v[b_id]) * aa
            velr = self.v[a_id] - self.v[b_id]
            margin = - self.width[a_id] * self.width[a_id] * 4.0
            d0 = self.searchR
            for j in ti.static(ti.ndrange(self.dim)):
                margin += ab[j] ** 2
                d0 += (velr[j] ** 2)

            if margin < d0:
                valLog = tm.log(margin / d0)
                valLogC = valLog * (margin - d0)

                self.global_E[None] -= valLogC * (margin - d0) * self.coef

    @ti.kernel
    def compute_energy_U_ao(self):
        aa = self.alpha[None]
        for i in self.candidate_ao:
            v_id = self.candidate_ao[i][0]
            p_id = self.candidate_ao[i][1]

            v0 = self.obstacle_v[v_id, 0]
            v1 = self.obstacle_v[v_id, 1]
            p = self.x[p_id] + self.new_v[p_id] * aa
            radsq = self.width[p_id] ** 2

            obsVec = v1 - v0
            relpos0 = v0 - p
            relpos1 = v1 - p

            lensq = obsVec.dot(obsVec)
            s = - relpos0.dot(obsVec) / lensq

            if s < 0.:
                dsitsq0 = relpos0.dot(relpos0)
                if dsitsq0 < radsq + self.searchR:
                    E = self.clog_E(dsitsq0-radsq, self.searchR)
                    self.global_E[None] += E

            elif s > 1.:
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
        # for i in range(self.n_crafts[None]):
        for Id in self.active_agents:
            i = self.active_agents[Id]
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
        self.intersect()

        while self.alpha[None] > self.global_alpha_min:
            self.global_E[None] = 0.0
            self.compute_energy_E()
            self.compute_energy_U()
            self.compute_energy_U_ao()
            self.armijo_condi()

            # self.update_E_with_grad()

            # print('new_E:', self.global_E[None])

            if self.global_E[None] <= self.prev_global_E[None] + self.armijo[None]:
                self.alpha[None] *= 1.1
                break
            else:
                self.alpha[None] *= self.alpha_dec

    @ti.kernel
    def armijo_condi(self):
        self.armijo[None] = 0.
        c = 0.1
        for Id in self.active_agents:
            i = self.active_agents[Id]
            self.armijo[None] += self.temp_grad[i] @ self.new_v[i]

        self.armijo[None] *= (c * self.alpha[None])

    def intersect(self):
        while self.alpha[None] > self.global_alpha_min:
            self.intersection_check_result[None] = 0
            self.intersection_check()

            if self.intersection_check_result[None] == 0:
                break
            else:
                self.alpha[None] *= self.alpha_dec






    @ti.kernel
    def update_pos_32(self, new_alpha: ti.f32):
        # for i in range(self.n_crafts[None]):
        for Id in self.active_agents:
            i = self.active_agents[Id]
            # self.x[i] -= self.gradient[i] * new_alpha
            self.x[i] += self.new_v[i] * new_alpha

    @ti.kernel
    def update_pos_64(self, new_alpha: ti.f64):
        # for i in range(self.n_crafts[None]):
        for Id in self.active_agents:
            i = self.active_agents[Id]
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
    def norm_grad(self):
        self.global_grad_norm[None] = 0.
        for i in self.gradient:
            self.global_grad_norm[None] += tm.dot(self.gradient[i], self.gradient[i])
            # print('grad', self.gradient[i])

        # Global norm for normalizing the local grads
        grad_norm = tm.sqrt(self.global_grad_norm[None])
        if grad_norm==0.0:
            grad_norm = 1.0

        for i in self.gradient:
            self.gradient[i] /= grad_norm


    @ti.kernel
    def direct_solving(self):
        self.global_grad_norm[None] = 0.
        for Id in self.active_agents:
            i = self.active_agents[Id]
            self.global_grad_norm[None] += tm.dot(self.gradient[i], self.gradient[i])
            self.new_v[i] = -self.gradient[i]
            # print('grad', self.gradient[i])

        #Global norm for normalizing the local grads
        grad_norm = tm.sqrt(self.global_grad_norm[None])
        if grad_norm == 0.0:
            grad_norm = 1.0

        print('grad_norm', grad_norm)
        self.alpha[None] /= grad_norm

    @ti.kernel
    def assemble_Value(self, x_mat: ti.types.ndarray(), single_x: ti.types.vector(2, dtype=ti.f32)):
        for i in range(self.n_crafts[None]):
            for m in ti.static(range(2)):
                x_mat[2 * i + m] = single_x[i][m]

    @ti.kernel
    def get_x_s(self, iters: ti.i32):
        # for i in range(self.n_crafts[None]):
        for Id in self.active_agents:
            i = self.active_agents[Id]
            scale_value = self.new_v[i] * self.alpha[None]
            # ti.static_print(scale_value[0])
            self.x[i] += scale_value
            # for j in ti.static(range(2)):
            self.s_i[iters, Id] = scale_value

    @ti.kernel
    def tempora_save_grad(self, iters: ti.i32):
        # yk = gk+1 - gk; unless k=0, gk appears twice in total computation
        # for i in range(self.n_crafts[None]):
        for Id in self.active_agents:
            i = self.active_agents[Id]
            if iters > 0:
                self.y_i[iters, Id] -= self.gradient[i]
                self.y_i[iters - 1, Id] += self.gradient[i]
                #The Id the the sequential id for each active flag, i means the id of ith agent

                    # print(iters, self.y_i[iters, 2 * i + j])
            else:
                self.y_i[iters, Id] -= self.gradient[i]



    @ti.kernel
    def get_rho_scale(self, iters: ti.i32):
        # for i in range(self.dim * self.n_crafts[None]):
        for Id in self.active_agents:
            # i = self.active_agents[Id]
            self.rho_i[iters] += self.y_i[iters, Id] @ self.s_i[iters, Id]
            self.tmp_value[None] += self.y_i[iters, Id] @ self.y_i[iters, Id]
        # print('rho scale', iters, self.rho_i[iters])
        # print('tmp scale', iters, self.tmp_value[None])

        if self.rho_i[iters] < 1e-9:
            self.rho_i[iters] = 1.0
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
    def apply_first_loop(self, iters: ti.i32):
        self.alpha_i[iters] = 0.0  # fresh
        for Id in self.active_agents:
            i = self.active_agents[Id]
            self.alpha_i[iters] += self.gradient[i] @ self.s_i[iters, Id]
        # print('alpha before rho', self.alpha_i[iters])
        self.alpha_i[iters] /= self.rho_i[iters]
        # print(self.rho_i[iters])
        # print("alpha:", self.alpha_i[iters])
        for Id in self.active_agents:
            i = self.active_agents[Id]
            self.gradient[i] -= self.alpha_i[iters] * self.y_i[iters, Id]

    @ti.kernel
    def get_r_32(self, scale: ti.f32):
        #TODO: What if initialized inv-H is not gamma*I
        for Id in self.active_agents:
            i = self.active_agents[Id]
            self.gradient[i] *= scale

    @ti.kernel
    def get_r_64(self, scale: ti.f64):
        #TODO: What if initialized inv-H is not gamma*I
        for Id in self.active_agents:
            i = self.active_agents[Id]
            self.gradient[i] *= scale

    @ti.kernel
    def apply_second_loop(self, iters: ti.i32):
        self.beta_i[None] = 0.0
        for Id in self.active_agents:
            i = self.active_agents[Id]
            self.beta_i[None] += self.y_i[iters, Id] @ self.gradient[i]
        self.beta_i[None] /= self.rho_i[iters]
        loop2_scale = self.alpha_i[iters] - self.beta_i[None]
        for Id in self.active_agents:
            i = self.active_agents[Id]
            self.gradient[i] += self.s_i[iters, Id] * loop2_scale


    @ti.kernel
    def set_direction(self):
        self.n_active[None] = 1.
        self.global_grad_norm[None] = 0.
        for Id in self.active_agents:
            i = self.active_agents[Id]
            self.global_grad_norm[None] += tm.dot(self.gradient[i], self.gradient[i])
            # self.new_v[i] = -self.gradient[i]
            # print('grad', self.gradient[i])
            self.n_active[None] += 1.0

        #Global norm for normalizing the local grads
        grad_norm = 10.0 * tm.sqrt(self.global_grad_norm[None] / self.n_active[None])

        # print('grad_norm', grad_norm)
        # self.alpha[None] /= grad_norm

        for i in self.new_v:
            self.new_v[i] = -self.gradient[i] / grad_norm


    def sparse_solving_LBFGS(self, iters):
        m_iters = iters if iters < self.m_iters else self.m_iters

        self.tempora_save_grad(iters)
        # real grad already saved

        if m_iters == 0:
            self.set_direction()
            self.line_search()
        else:
            #two-loop recursions
            self.tmp_value[None] = 0.0
            self.get_rho_scale(iters - 1)
            if self.tmp_value[None] < 1e-6:
                scale = 1.0
            else:
                scale = 1 / self.tmp_value[None]
            # scale = 1.0
            # print('look1', iters - 1)
            # print('scale', scale)

            # self.help_init_inv_H(scale)

            for iter1 in range(iters-1, iters - m_iters - 1, -1):
                self.apply_first_loop(iter1)
                # print('look2', iter1)

            if ti.static(self.accuracy == 'f32'):
                self.get_r_32(scale)
            else:
                self.get_r_64(scale)

            for iter2 in range(iters - m_iters, iters):
                self.apply_second_loop(iter2)


            self.set_direction()
            self.line_search()

        self.get_x_s(iters)


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
        # self.init_ao_grid()
        self.insert_grid_pos()
        self.find_neighbour()
        self.find_ao_neighbour()
        # print(self.candidate_neighbor)

        self.init_grad()
        ##################################
        # print("check after init grad")
        # self.check_grad_helper()

        self.compute_gradAA()
        # #################################
        # print("check after init gradAA")
        # self.check_grad_helper()
        # TODO: Exist NaN here

        self.compute_gradAO()
        ##################################
        # print("check after init gradAO")
        # self.norm_grad()
        # self.check_grad_helper()
        # TODO: Exist NaN here
        self.save_grad()




        # self.sparse_solving()
        if ti.static(step_optimizer == 'L-BFGS'):
            self.sparse_solving_LBFGS(iters)
        # elif step_optimizer == 'Newton': #Need perturbation
        #     self.sparse_solving()  #BFGS may start after the very first iteration ends to get a better H_0 approximation
        # elif step_optimizer == 'BFGS':
        #     self.sparse_solving_BFGS()
        else:
            self.direct_solving()  #Gradient Descent

        self.line_search()



    def optimize_LBFGS(self):
        iter = 0
        self.clear_y_s_rho()
        # self.process_action(actions)

        while iter < self.max_iters:
            self.single_step('L-BFGS', iter)
            iter += 1
            if self.alpha[None] < self.global_alpha_min:
                print("finished:", iter, self.alpha[None])
                # print(self.alpha[None])
                # craft_info = self.render_info()
                break

        craft_info = self.render_info()
        self.check_activity()
        print("finished:", iter, self.alpha[None])


        return craft_info, iter



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
        # for i in range(self.n_crafts[None]):
        for Id in self.active_agents:
            i = self.active_agents[Id]
            type[Id] = self.craft_type[i]
            camp[Id] = self.craft_camp[i]

            for j in ti.static(range(self.dim)):
                x[Id, j] = self.x[i][j]
                v[Id, j] = self.v[i][j]


    @ti.kernel
    def get_render_len(self) -> int:
        render_len = self.active_agents.length()
        return render_len

    def render_info(self):
        render_len = self.get_render_len()
        np_x = np.ndarray((render_len, self.dim), dtype=np.float32)
        np_v = np.ndarray((render_len, self.dim), dtype=np.float32)
        np_type = np.ndarray((render_len,), dtype=np.float32)
        np_camp = np.ndarray((render_len,), dtype=np.float32)

        self.get_render_info(np_type, np_camp, np_x, np_v)

        craft_data = {
            'position': np_x,
            'velocity': np_v,
            'type': np_type,
            'camp': np_camp
        }

        return craft_data



    #TODO: Reuse this part when reimplementing vheicles


    @ti.kernel
    def copy_dynamic_nd(self, np_x: ti.types.ndarray(), input_x: ti.template()):
        for i in self.x:
            for j in ti.static(range(self.dim)):
                np_x[i, j] = input_x[i][j]

    @ti.kernel
    def copy_dynamic(self, np_x: ti.types.ndarray(), input_x: ti.template()):
        for i in self.x:
            np_x[i] = input_x[i]





