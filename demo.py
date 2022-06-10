import taichi as ti
import numpy as np
from particle_system import ParticleSystem
from wcsph import WCSPHSolver

ti.init(arch=ti.vulkan,  device_memory_fraction=0.9)

# TODO, Important: CUDA_LAUNCH_BLOCKING=1;PYTHONUNBUFFERED=1;TI_USE_UNIFIED_MEMORY=0;TI_ENABLE_CUDA=0

# ti.init(arch=ti.cuda, device_memory_GB=2)

# Use GPU for higher peformance if available
#ti.init(arch=ti.gpu, device_memory_GB=8, packed=True)


if __name__ == "__main__":
    ps = ParticleSystem((512, 512))

    ps.add_circle(circle_heart=[5, 5],
                radius=1,
                velocity=[0.0, -10.0],
                density=1.0,
                color=0x6FA8DC,
                material=1)

    ps.add_grond(lower_corner=[0, 0],
                cube_size=[10.0, 0.05],
                velocity=[0, 0],
                density=10.0,
                color=0x00000,
                material=1)



    # ps.add_cube(lower_corner=[3, 1],
    #             cube_size=[2.0, 6.0],
    #             velocity=[0.0, -20.0],
    #             density=1000.0,
    #             color=0x956333,
    #             material=1)

    wcsph_solver = WCSPHSolver(ps)
    gui = ti.GUI(background_color=0xFFFFFF)

    print('test1')
    while gui.running:
        for i in range(5):
            wcsph_solver.step()
        particle_info = ps.dump()
        gui.circles(particle_info['position'] * ps.screen_to_world_ratio / 512,
                    radius=ps.particle_radius / 1 * ps.screen_to_world_ratio,
                    color=particle_info['color'])
        gui.show()
