import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from numpy.linalg import inv



pos = []
pos_x = []
pos_y = []
pos_z = []


ground_x = []
ground_y = []
ground_z = []

pos_xx = []
pos_yy = []
pos_zz = []
# for i in range(100):
#     pos.append(i)
# ax.scatter(pos, pos, pos)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()

f = open('../Model/blue_car/rotate_30_model.obj', 'r')
out = open("../Model/blue_car/calib_model3.obj", 'w')
vertex_count = 0
normal_count = 0
face_count = 0
vt_count = 0
V = []
Normal = []
normal_map = {}
vindx_map = {}
g_indx = []

alpha = np.pi / 6

# rot_alpha = np.array([[math.cos(alpha), -math.sin(alpha), 0],
#                       [math.sin(alpha), math.cos(alpha), 0],
#                       [0, 0, 1]])

# Wall
# rot_alpha = np.array([[0.8660254,  0.5, 0],
#                       [-0.5,        0.8660254,  0       ],
#                       [ 0,         0,         1       ]])

rot_alpha = np.array([[ 0.9022309,   0.42968398,  0.03675711],
                      [-0.42306469,  0.89840925, -0.11780107],
                      [-0.08364016, 0.09073313,  0.9923567 ]])



translate_para = np.array([1.196927120953121, -2.5, -0.9165016454286453])

new_normal = rot_alpha @ np.array([0, 1, 0])

print('new normal : ' + str(new_normal))

# A = np.array([5, 2, 1 ])
# print(inv(A.T))

for line in f:
    # print(line)

    # v
    if line[0] == 'v' and line[1] == ' ':
        # print(line)
        # print(line[2:-1].split(' '))
        vertex = line[2:-1].split(' ')

        # print(float(vertex[0]))
        # if float(vertex[1]) < -2.5 and float(vertex[1]) > -2.8:
        #     ground_x.append(float(vertex[0]))
        #     ground_y.append(float(vertex[1]))
        #     ground_z.append(float(vertex[2]))
        #     g_indx.append(vertex_count)
        #     out.write(line)
        # else:
            # pos_xx.append(float(vertex[0]))
            # pos_yy.append(float(vertex[1]))
            # pos_zz.append(float(vertex[2]))

        # pos_x.append(float(vertex[0]))
        # pos_y.append(float(vertex[1]))
        # pos_z.append(float(vertex[2]))
        V.append(np.array([float(vertex[0]), float(vertex[2]), float(vertex[1])]))
        vindx_map[tuple([float(vertex[0]), float(vertex[2]), float(vertex[1])])] = vertex_count


        # Transform:

        # Rotate
        calib_v = rot_alpha @ np.array([float(vertex[0]), float(vertex[1]), float(vertex[2])])

        # Translate
        calib_v += translate_para


        new_line = 'v'

        for coord in calib_v:
            new_line += ' ' + str(coord)

        new_line += '\n'
        # print(calib_v)
        out.write(new_line)

        vertex_count += 1
    # vn
    elif line[0] == 'v' and line[1] == 'n':
        # print(line)
        normal_count += 1
        normal_para = line[3:-1].split(' ')
        Normal.append(np.array([float(normal_para[0]), float(normal_para[2]), float(normal_para[1])]))

        # Transform:

        calib_v = rot_alpha @ np.array([float(normal_para[0]), float(normal_para[1]), float(normal_para[2])])

        new_line = 'vn'

        for coord in calib_v:
            new_line += ' ' + str(coord)

        new_line += '\n'
        # print(calib_v)
        out.write(new_line)

    else:
        out.write(line)

        # vt
        if line[0] == 'v' and line[1] == 't':
            vt_count += 1

        # f
        if line[0] == 'f':
            # print(line)
            face_para = line[2:-1].split(' ')
            # print(face_para)
            for v_info in face_para:
                v_idx, uv_idx, n_idx = v_info.split('/')
                # print(v_idx)
                if v_idx not in normal_map:
                    # print(n_idx)
                    normal_map[int(v_idx)] = Normal[int(n_idx) - 1]

            # face_indices = face_para.split('/')
            # print(face_indices)
            face_count += 1



print('vertex count = ' + str(vertex_count))
print('normal count = ' + str(normal_count))
print('face count = ' + str(face_count))
print('vt count = ' + str(vt_count))
f.close
out.close

# PLOT
