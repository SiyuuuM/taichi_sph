import taichi as ti
import numpy as np
from functools import reduce


@ti.data_oriented
class ParticleSystem:
    def __init__(self, res):
        self.res = res
        self.dim = len(res)
        assert self.dim > 1
        self.screen_to_world_ratio = 50
        self.bound = np.array(res) / self.screen_to_world_ratio

        # Material
        self.material_boundary = 0
        self.material_fluid = 1

        self.particle_radius = 0.07  # particle radius
        self.particle_diameter = 2 * self.particle_radius
        self.support_radius = self.particle_radius * 4.0  # support radius
        self.m_V = np.pi * self.particle_radius ** self.dim
        self.particle_max_num = 2 ** 15
        self.particle_max_num_per_cell = 100
        self.particle_max_num_neighbor = 100
        self.particle_num = ti.field(int, shape=())

        # Grid related properties
        self.grid_size = self.support_radius
        self.grid_num = np.ceil(np.array(res) / self.grid_size).astype(int)
        self.grid_particles_num = ti.field(int)
        self.grid_particles = ti.field(int)
        self.padding = self.grid_size

        # Particle related properties
        self.x = ti.Vector.field(self.dim, dtype=float)
        self.v = ti.Vector.field(self.dim, dtype=float)
        self.density = ti.field(dtype=float)
        self.pressure = ti.field(dtype=float)
        self.material = ti.field(dtype=int)
        self.color = ti.field(dtype=int)
        self.particle_neighbors = ti.field(int)
        self.particle_neighbors_num = ti.field(int)

        self.particles_node = ti.root.dense(ti.i, self.particle_max_num)
        self.particles_node.place(self.x, self.v, self.density, self.pressure, self.material, self.color)
        self.particles_node.place(self.particle_neighbors_num)
        self.particle_node = self.particles_node.dense(ti.j, self.particle_max_num_neighbor)
        self.particle_node.place(self.particle_neighbors)

        index = ti.ij if self.dim == 2 else ti.ijk
        grid_node = ti.root.dense(index, self.grid_num)
        grid_node.place(self.grid_particles_num)

        cell_index = ti.k if self.dim == 2 else ti.l
        cell_node = grid_node.dense(cell_index, self.particle_max_num_per_cell)
        cell_node.place(self.grid_particles)

        # Line = Line(np.array([0, 3.6,0]), np.array([0, 2, 0]), color='r')





    @ti.func
    def add_particle(self, p, x, v, density, pressure, material, color):
        self.x[p] = x
        self.v[p] = v
        self.density[p] = density
        self.pressure[p] = pressure
        self.material[p] = material
        self.color[p] = color

    @ti.kernel
    def add_particles(self, new_particles_num: int,
                      new_particles_positions: ti.types.ndarray(),
                      new_particles_velocity: ti.types.ndarray(),
                      new_particle_density: ti.types.ndarray(),
                      new_particle_pressure: ti.types.ndarray(),
                      new_particles_material: ti.types.ndarray(),
                      new_particles_color: ti.types.ndarray()):
        for p in range(self.particle_num[None], self.particle_num[None] + new_particles_num):
            v = ti.Vector.zero(float, self.dim)
            x = ti.Vector.zero(float, self.dim)
            for d in ti.static(range(self.dim)): #TODO: 啥东西
                v[d] = new_particles_velocity[p - self.particle_num[None], d]
                x[d] = new_particles_positions[p - self.particle_num[None], d]
            self.add_particle(p, x, v,
                              new_particle_density[p - self.particle_num[None]],
                              new_particle_pressure[p - self.particle_num[None]],
                              new_particles_material[p - self.particle_num[None]],
                              new_particles_color[p - self.particle_num[None]])
        self.particle_num[None] += new_particles_num

    @ti.kernel
    def add_particles_water(self, new_particles_num: int,
                      new_particles_positions: ti.types.ndarray(),
                      new_particles_velocity: ti.types.ndarray(),
                      new_particle_density: ti.types.ndarray(),
                      new_particle_pressure: ti.types.ndarray(),
                      new_particles_material: ti.types.ndarray(),
                      new_particles_color: ti.types.ndarray(),
                      new_particles_ground: ti.types.ndarray()):
        for p in range(self.particle_num[None], self.particle_num[None] + new_particles_num):
            v = ti.Vector.zero(float, self.dim)
            x = ti.Vector.zero(float, self.dim)
            for d in ti.static(range(self.dim)):
                v[d] = new_particles_velocity[p - self.particle_num[None], d]
                x[d] = new_particles_positions[p - self.particle_num[None], d]
            self.add_particle(p, x, v,
                              new_particle_density[p - self.particle_num[None]],
                              new_particle_pressure[p - self.particle_num[None]],
                              new_particles_material[p - self.particle_num[None]],
                              new_particles_color[p - self.particle_num[None]])
        self.particle_num[None] += new_particles_num

    @ti.func
    def pos_to_index(self, pos):
        return (pos / self.grid_size).cast(int)

    @ti.func
    def is_valid_cell(self, cell):
        # Check whether the cell is in the grid
        flag = True
        for d in ti.static(range(self.dim)):
            flag = flag and (0 <= cell[d] < self.grid_num[d])
        return flag

    @ti.kernel
    def allocate_particles_to_grid(self):  # 搭在网格里，之后方便查找
        for p in range(self.particle_num[None]):
            cell = self.pos_to_index(self.x[p])
            offset = self.grid_particles_num[cell].atomic_add(1)
            self.grid_particles[cell, offset] = p

    @ti.kernel
    def search_neighbors(self):
        for p_i in range(self.particle_num[None]):
            # Skip boundary particles
            if self.material[p_i] == self.material_boundary:
                continue
            center_cell = self.pos_to_index(self.x[p_i])
            cnt = 0
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
                if cnt >= self.particle_max_num_neighbor:
                    break
                cell = center_cell + offset
                if not self.is_valid_cell(cell):
                    break
                for j in range(self.grid_particles_num[cell]):
                    p_j = self.grid_particles[cell, j]
                    distance = (self.x[p_i] - self.x[p_j]).norm()
                    if p_i != p_j and distance < self.support_radius:
                        self.particle_neighbors[p_i, cnt] = p_j
                        cnt += 1
            self.particle_neighbors_num[p_i] = cnt

    def initialize_particle_system(self):
        self.grid_particles_num.fill(0)
        self.particle_neighbors.fill(-1)
        self.allocate_particles_to_grid()
        self.search_neighbors()

    @ti.kernel
    def copy_to_numpy_nd(self, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            for j in ti.static(range(self.dim)):
                np_arr[i, j] = src_arr[i][j]

    @ti.kernel
    def copy_to_numpy(self, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            np_arr[i] = src_arr[i]

    def dump(self):
        np_x = np.ndarray((self.particle_num[None], self.dim), dtype=np.float32)
        self.copy_to_numpy_nd(np_x, self.x)

        np_v = np.ndarray((self.particle_num[None], self.dim), dtype=np.float32)
        self.copy_to_numpy_nd(np_v, self.v)

        np_material = np.ndarray((self.particle_num[None],), dtype=np.int32)
        self.copy_to_numpy(np_material, self.material)

        np_color = np.ndarray((self.particle_num[None],), dtype=np.int32)
        self.copy_to_numpy(np_color, self.color)

        return {
            'position': np_x,
            'velocity': np_v,
            'material': np_material,
            'color': np_color
        }

    def add_cube(self,
                 lower_corner,
                 cube_size,
                 material,
                 color=0xFFFFFF,
                 density=None,
                 pressure=None,
                 velocity=None):

        num_dim = []
        for i in range(self.dim):
            num_dim.append(
                np.arange(lower_corner[i], lower_corner[i] + cube_size[i],
                          self.particle_radius))
        num_new_particles = reduce(lambda x, y: x * y,
                                   [len(n) for n in num_dim])
        assert self.particle_num[
                   None] + num_new_particles <= self.particle_max_num

        new_positions = np.array(np.meshgrid(*num_dim,
                                             sparse=False,
                                             indexing='ij'),
                                 dtype=np.float32)
        new_positions = new_positions.reshape(-1,
                                              reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
        print("new position shape ", new_positions.shape)
        if velocity is None:
            velocity = np.full_like(new_positions, 0)
        else:
            velocity = np.array([velocity for _ in range(num_new_particles)], dtype=np.float32)

        material = np.full_like(np.zeros(num_new_particles), material)
        color = np.full_like(np.zeros(num_new_particles), color)
        density = np.full_like(np.zeros(num_new_particles), density if density is not None else 1000.)
        pressure = np.full_like(np.zeros(num_new_particles), pressure if pressure is not None else 0.)
        self.add_particles(num_new_particles, new_positions, velocity, density, pressure, material, color)

    def add_grond(self,
                 lower_corner,
                 cube_size,
                 material,
                 color=0x00000,
                 density=None,
                 pressure=None,
                 velocity=None):

        num_dim = []
        for i in range(self.dim):
            num_dim.append(
                np.arange(lower_corner[i], lower_corner[i] + cube_size[i],
                          self.particle_radius))
        num_new_particles = reduce(lambda x, y: x * y,
                                   [len(n) for n in num_dim])
        assert self.particle_num[
                   None] + num_new_particles <= self.particle_max_num

        new_positions = np.array(np.meshgrid(*num_dim,
                                             sparse=False,
                                             indexing='ij'),
                                 dtype=np.float32)
        new_positions = new_positions.reshape(-1,
                                              reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
        print("new position shape ", new_positions.shape)
        if velocity is None:
            velocity = np.full_like(new_positions, 0)
        else:
            velocity = np.array([velocity for _ in range(num_new_particles)], dtype=np.float32)

        material = np.full_like(np.zeros(num_new_particles), material)
        color = np.full_like(np.zeros(num_new_particles), 0x00000)
        density = np.full_like(np.zeros(num_new_particles), density if density is not None else 1000.)
        pressure = np.full_like(np.zeros(num_new_particles), pressure if pressure is not None else 0.)
        self.add_particles(num_new_particles, new_positions, velocity, density, pressure, material, color)



    def add_circle(self,
                 circle_heart,
                 radius,
                 material,
                 color=0xFFFFFF,
                 density=None,
                 pressure=None,
                 velocity=None):

        radius_list = np.arange(radius, 0, -self.particle_radius)
        result_list_x = []
        result_list_y = []
        for n in range(len(radius_list)):
            theta_list = np.linspace(0, 2 * np.pi, int(2 * np.pi * radius_list[n] / self.particle_radius))
            res_x = circle_heart[0] + radius_list[n] * np.cos(theta_list)
            res_x = np.delete(res_x, -1)
            for _ in range(len(res_x)):
                result_list_x.append(res_x[_])
            res_y = circle_heart[1] + radius_list[n] * np.sin(theta_list)
            res_y = np.delete(res_y, -1)
            for _ in range(len(res_x)):
                result_list_y.append(res_y[_])

        result_list = [result_list_x, result_list_y]

        # num_new_particles = reduce(lambda x, y: x * y,
        #                            [len(n) for n in result_list])
        num_new_particles = len(result_list_x)

        print(num_new_particles)
        assert self.particle_num[
                   None] + num_new_particles <= self.particle_max_num
        new_positions = []
        for _ in range(len(result_list_x)):
            new_positions.append([result_list[0][_], result_list[1][_]])
        # new_positions = np.unique(np.array(new_positions),axis=0)
        new_positions = np.array(new_positions)
        print(new_positions)

        print("new position shape ", new_positions.shape)
        if velocity is None:
            velocity = np.full_like(new_positions, 0)
        else:
            velocity = np.array([velocity for _ in range(num_new_particles)], dtype=np.float32)

        material = np.full_like(np.zeros(num_new_particles), material)
        color = np.full_like(np.zeros(num_new_particles), color)
        density = np.full_like(np.zeros(num_new_particles), density if density is not None else 1000.)
        pressure = np.full_like(np.zeros(num_new_particles), pressure if pressure is not None else 0.)
        self.add_particles(num_new_particles, new_positions, velocity, density, pressure, material, color)