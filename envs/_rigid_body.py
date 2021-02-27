import math
import taichi as ti
import numpy as np

objects = []
springs = []


l_thigh_init_ang = 10
l_calf_init_ang = -10
r_thigh_init_ang = 10
r_calf_init_ang = -10
initHeight = 0.15

hip_pos = [0.3, 0.5 + initHeight]
thigh_half_length = 0.11
calf_half_length = 0.11

foot_half_length = 0.08
half_hip_length = 0.08

def rotAlong(half_length, deg, center):
    ang = math.radians(deg)
    return [
        half_length * math.sin(ang) + center[0],
        -half_length * math.cos(ang) + center[1]
    ]


def add_object(x, halfsize, rotation=0):
    objects.append([x, halfsize, rotation])
    return len(objects) - 1


# actuation 0.0 will be translated into default actuation
def add_spring(a, b, offset_a, offset_b, length, stiffness, actuation=0.0):
    springs.append([a, b, offset_a, offset_b, length, stiffness, actuation])

def robotLeg():
    #hip
    add_object(hip_pos, halfsize=[0.06, half_hip_length])
    hip_end = [hip_pos[0], hip_pos[1] - (half_hip_length - 0.01)]

    #left
    l_thigh_center = rotAlong(thigh_half_length, l_thigh_init_ang, hip_end)
    l_thigh_end = rotAlong(thigh_half_length * 2.0, l_thigh_init_ang, hip_end)
    add_object(l_thigh_center,
               halfsize=[0.02, thigh_half_length],
               rotation=math.radians(l_thigh_init_ang))
    add_object(rotAlong(calf_half_length, l_calf_init_ang, l_thigh_end),
               halfsize=[0.02, calf_half_length],
               rotation=math.radians(l_calf_init_ang))
    l_calf_end = rotAlong(2.0 * calf_half_length, l_calf_init_ang, l_thigh_end)
    add_object([l_calf_end[0] + foot_half_length, l_calf_end[1]],
               halfsize=[foot_half_length, 0.02])

    #right
    add_object(rotAlong(thigh_half_length, r_thigh_init_ang, hip_end),
               halfsize=[0.02, thigh_half_length],
               rotation=math.radians(r_thigh_init_ang))
    r_thigh_end = rotAlong(thigh_half_length * 2.0, r_thigh_init_ang, hip_end)
    add_object(rotAlong(calf_half_length, r_calf_init_ang, r_thigh_end),
               halfsize=[0.02, calf_half_length],
               rotation=math.radians(r_calf_init_ang))
    r_calf_end = rotAlong(2.0 * calf_half_length, r_calf_init_ang, r_thigh_end)
    add_object([r_calf_end[0] + foot_half_length, r_calf_end[1]],
               halfsize=[foot_half_length, 0.02])

    s = 200

    thigh_relax = 0.9
    leg_relax = 0.9
    foot_relax = 0.7

    thigh_stiff = 5
    leg_stiff = 20
    foot_stiff = 40

    #left springs
    add_spring(0, 1, [0, (half_hip_length - 0.01) * 0.4],
               [0, -thigh_half_length],
               thigh_relax * (2.0 * thigh_half_length + 0.22), thigh_stiff)
    add_spring(1, 2, [0, thigh_half_length], [0, -thigh_half_length],
               leg_relax * 4.0 * thigh_half_length, leg_stiff, 0.08)
    add_spring(
        2, 3, [0, 0], [foot_half_length, 0],
        foot_relax *
        math.sqrt(pow(thigh_half_length, 2) + pow(2.0 * foot_half_length, 2)),
        foot_stiff)

    add_spring(0, 1, [0, -(half_hip_length - 0.01)], [0.0, thigh_half_length],
               -1, s)
    add_spring(1, 2, [0, -thigh_half_length], [0.0, thigh_half_length], -1, s)
    add_spring(2, 3, [0, -thigh_half_length], [-foot_half_length, 0], -1, s)

    #right springs
    add_spring(0, 4, [0, (half_hip_length - 0.01) * 0.4],
               [0, -thigh_half_length],
               thigh_relax * (2.0 * thigh_half_length + 0.22), thigh_stiff)
    add_spring(4, 5, [0, thigh_half_length], [0, -thigh_half_length],
               leg_relax * 4.0 * thigh_half_length, leg_stiff, 0.08)
    add_spring(
        5, 6, [0, 0], [foot_half_length, 0],
        foot_relax *
        math.sqrt(pow(thigh_half_length, 2) + pow(2.0 * foot_half_length, 2)),
        foot_stiff)

    add_spring(0, 4, [0, -(half_hip_length - 0.01)], [0.0, thigh_half_length],
               -1, s)
    add_spring(4, 5, [0, -thigh_half_length], [0.0, thigh_half_length], -1, s)
    add_spring(5, 6, [0, -thigh_half_length], [-foot_half_length, 0], -1, s)

    return objects, springs, 3

def setup_robot(objects, springs, h_id):
    global head_id
    head_id = h_id
    global n_objects, n_springs
    n_objects = len(objects)
    n_springs = len(springs)

    print('n_objects=', n_objects, '   n_springs=', n_springs)

    # real = ti.f32
    # vec = lambda: ti.Vector.field(2, dtype=real)
    # x = vec()
    # # for i in range(n_objects):
    # #     x.append([i])
    # print("x",x)
    # print("x[0, 0]",x[0, 0])
    x = []
    halfsize = []
    rotation = []


    for i in range(n_objects):
        print("objects[i][0]",objects[i][0])
        # x[i] = objects[i][0]
        # halfsize[i] = objects[i][1]
        # rotation[0, i] = objects[i][2]
        x.append(objects[i][0])
        halfsize.append(objects[i][1])
        rotation.append(objects[i][2])

    # for i in range(n_springs):
    #     s = springs[i]
    #     spring_anchor_a[i] = s[0]
    #     spring_anchor_b[i] = s[1]
    #     spring_offset_a[i] = s[2]
    #     spring_offset_b[i] = s[3]
    #     spring_length[i] = s[4]
    #     spring_stiffness[i] = s[5]
    #     if s[6]:
    #         spring_actuation[i] = s[6]
    #     else:
    #         default_actuation = 0.05
    #         spring_actuation[i] = default_actuation
    return x, halfsize, rotation



def forward(x, halfsize, rotation,output=None, visualize=True):
    gui = ti.GUI('Rigid Body Simulation', (512, 512), background_color=0xFFFFFF)
    for j in range(1000):
    # while True:
        for i in range(n_objects):
            points = []
            for k in range(4):
                offset_scale = [[-1, -1], [1, -1], [1, 1], [-1, 1]][k]
                # rot = rotation[0,i]
                rot = rotation[i]
                rot_matrix = np.array([[math.cos(rot), -math.sin(rot)],
                                        [math.sin(rot),
                                        math.cos(rot)]])

                # pos = np.array([x[0, i][0], x[0, i][1]
                #                 ]) + offset_scale * rot_matrix @ np.array(
                #                     [halfsize[i][0], halfsize[i][1]])
                pos = np.array([x[i][0], x[i][1]
                                ]) + offset_scale * rot_matrix @ np.array(
                                    [halfsize[i][0], halfsize[i][1]])
                

                points.append((pos[0], pos[1]))

            for k in range(4):
                gui.line(points[k],
                            points[(k + 1) % 4],
                            color=0x0,
                            radius=2)

        # for i in range(n_springs):

        #     def get_world_loc(i, offset):
        #         rot = rotation[t, i]
        #         rot_matrix = np.array([[math.cos(rot), -math.sin(rot)],
        #                                [math.sin(rot),
        #                                 math.cos(rot)]])
        #         pos = np.array([[x[t, i][0]], [
        #             x[t, i][1]
        #         ]]) + rot_matrix @ np.array([[offset[0]], [offset[1]]])
        #         return pos

        #     pt1 = get_world_loc(spring_anchor_a[i], spring_offset_a[i])
        #     pt2 = get_world_loc(spring_anchor_b[i], spring_offset_b[i])

        #     color = 0xFF2233

        #     if spring_actuation[i] != 0 and spring_length[i] != -1:
        #         a = actuation[t - 1, i] * 0.5
        #         color = ti.rgb_to_hex((0.5 + a, 0.5 - abs(a), 0.5 - a))

        #     if spring_length[i] == -1:
        #         gui.line(pt1, pt2, color=0x000000, radius=9)
        #         gui.line(pt1, pt2, color=color, radius=7)
        #     else:
        #         gui.line(pt1, pt2, color=0x000000, radius=7)
        #         gui.line(pt1, pt2, color=color, radius=5)

        ground_height = 0.1
        gui.line((0.05, ground_height - 5e-3),
                    (0.95, ground_height - 5e-3),
                    color=0x0,
                    radius=5)

        file = None
        gui.show(file=file)


if __name__ == '__main__':
    objects, springs, h_id = robotLeg()
    x, halfsize, rotation = setup_robot(objects, springs, h_id)
    forward(x, halfsize, rotation)