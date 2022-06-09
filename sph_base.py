import math

import taichi as ti
import numpy as np




@ti.data_oriented
class SPHBase:
    def __init__(self, particle_system):
        self.ps = particle_system
        self.g = -9.80  # Gravity
        self.viscosity = 0.000001  # viscosity
        self.density_0 = 997.0  # reference density
        self.mass = self.ps.m_V * self.density_0
        self.dt = ti.field(float, shape=())
        self.dt[None] = 2e-4
        self.surface_tension_coefficient = 72.8
        self.contact_angle = 60

    @ti.func
    def cubic_kernel(self, r_norm):
        res = ti.cast(0.0, ti.f32)
        h = self.ps.support_radius
        # value of cubic spline smoothing kernel
        k = 1.0
        if self.ps.dim == 1:
            k = 4 / 3
        elif self.ps.dim == 2:
            k = 40 / 7 / np.pi
        elif self.ps.dim == 3:
            k = 8 / np.pi
        k /= h ** self.ps.dim
        q = r_norm / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q * q
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1)
            else:
                res = k * 2 * ti.pow(1 - q, 3.0)
        return res

    @ti.func
    def cubic_kernel_derivative(self, r):
        h = self.ps.support_radius
        # derivative of cubic spline smoothing kernel
        k = 1.0
        if self.ps.dim == 1:
            k = 4 / 3
        elif self.ps.dim == 2:
            k = 40 / 7 / np.pi
        elif self.ps.dim == 3:
            k = 8 / np.pi
        k = 6. * k / h ** self.ps.dim
        r_norm = r.norm()
        q = r_norm / h
        res = ti.Vector([0.0 for _ in range(self.ps.dim)])
        if r_norm > 1e-5 and q <= 1.0:
            grad_q = r / (r_norm * h)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * grad_q
            else:
                factor = 1.0 - q
                res = k * (-factor * factor) * grad_q
        return res

    @ti.func
    def viscosity_force(self, p_i, p_j, r):
        # Compute the viscosity force contribution
        v_xy = (self.ps.v[p_i] -
                self.ps.v[p_j]).dot(r)
        res = 2 * (self.ps.dim + 2) * self.viscosity * (self.mass / (self.ps.density[p_j])) * v_xy / (
            r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(
                r)
        return res

    # @ti.func
    # def pressure_force(self, p_i, p_j, r):
    #     # Compute the pressure force contribution, Symmetric Formula
    #     res = -self.density_0 * self.ps.m_V * (self.ps.pressure[p_i] / self.ps.density[p_i] ** 2
    #                                            + self.ps.pressure[p_j] / self.ps.density[p_j] ** 2) \
    #           * self.cubic_kernel_derivative(r)
    #     return res
    #


    # @ti.func
    # def pressure_force(self):
    #     res = 4 * math.pi * 0.07 * self.surface_tension_coefficient * self.contact_angle
    #     return res
    # # laplace pressure的表达式


    # @ti.func
    # def


    # def substep(self):
    #     pass


    #对公式的翻译，速度场的laplace项 再乘以 viscosity的系数

    # @ti.func
    # def pressure_force(self, p_i, p_j, r):
    #     # Compute the pressure force contribution, Symmetric Formula
    #     res = -self.density_0 * self.ps.m_V * (self.ps.pressure[p_i] / self.ps.density[p_i] ** 2
    #           + self.ps.pressure[p_j] / self.ps.density[p_j] ** 2) \
    #           * self.cubic_kernel_derivative(r)
    #     return res
    #
    # def substep(self):
    #     p

    @ti.func
    def simulate_collisions(self, p_i, vec, d):
        # Collision factor, assume roughly (1-c_f)*velocity loss after collision 先判断位置，再采取办法。
        c_f = 0.3
        self.ps.x[p_i] += vec * d  # d代表粒子到边界的距离
        self.ps.v[p_i] -= (
            1.0 + c_f) * self.ps.v[p_i].dot(vec) * vec
        # 有一个向外的力，先投影到法线向量上再取负号，即弹回来的速度，并且加一个系数模拟速度回来的时候剩多少；

    @ti.kernel
    def enforce_boundary(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.dim == 2:
                if self.ps.material[p_i] == self.ps.material_fluid:
                    pos = self.ps.x[p_i]
                    if pos[0] < self.ps.padding:
                        self.simulate_collisions(
                            p_i, ti.Vector([1.0, 0.0]),
                            self.ps.padding - pos[0])
                    if pos[0] > self.ps.bound[0] - self.ps.padding:
                        self.simulate_collisions(
                            p_i, ti.Vector([-1.0, 0.0]),
                            pos[0] - (self.ps.bound[0] - self.ps.padding))
                    if pos[1] > self.ps.bound[1] - self.ps.padding:
                        self.simulate_collisions(
                            p_i, ti.Vector([0.0, -1.0]),
                            pos[1] - (self.ps.bound[1] - self.ps.padding))
                    if pos[1] < self.ps.padding:
                        self.simulate_collisions(
                            p_i, ti.Vector([0.0, 1.0]),
                           self.ps.padding - pos[1])

# boundary condition 没有考虑粒子，即速度在碰到边界后消失了，所以在画布中的最底层的粒子会粘在底部不动。

    def step(self):  # 执行每一步仿真的步骤
        self.ps.initialize_particle_system()  # search neighbours
        self.substep()
        self.enforce_boundary()
