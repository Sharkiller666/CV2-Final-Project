import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import time

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

f = open('../Model/Scene/street/untitled.obj', 'r')
out = open("../Model/Scene/street/ground.obj", 'w')
vertex_count = 0
normal_count = 0
face_count = 0
vt_count = 0
V = []
Normal = []
normal_map = {}
vindx_map = {}
g_indx = []
for line in f:
    # print(line)

    # v
    if line[0] == 'v' and line[1] == ' ':
        # print(line)
        # print(line[2:-1].split(' '))
        vertex = line[2:-1].split(' ')

        # print(float(vertex[0]))
        # wall
        if float(vertex[1]) < -2.5 and float(vertex[1]) > -2.8:
            ground_x.append(float(vertex[0]))
            ground_y.append(float(vertex[1]))
            ground_z.append(float(vertex[2]))
            g_indx.append(vertex_count)
            out.write(line)
        # else:
            # pos_xx.append(float(vertex[0]))
            # pos_yy.append(float(vertex[1]))
            # pos_zz.append(float(vertex[2]))

        # pos_x.append(float(vertex[0]))
        # pos_y.append(float(vertex[1]))
        # pos_z.append(float(vertex[2]))
        V.append(np.array([float(vertex[0]), float(vertex[2]), float(vertex[1])]))
        vindx_map[tuple([float(vertex[0]), float(vertex[2]), float(vertex[1])])] = vertex_count
        vertex_count += 1
    # vn
    if line[0] == 'v' and line[1] == 'n':
        # print(line)
        normal_count += 1
        normal_para = line[3:-1].split(' ')
        Normal.append(np.array([float(normal_para[0]), float(normal_para[2]), float(normal_para[1])]))

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


# ax = plt.figure().add_subplot(projection='3d')
# ax.scatter(pos_x, pos_y, pos_z, color='green')
# ax.scatter(pos_xx, pos_yy, pos_zz, color='red')
# ax.quiver(0, 0, -5, 0, 0, 8, color='blue')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
# plt.show()

# X = []
# for i in range(len(pos_x)):
#     X.append(np.array([pos_x[i], pos_z[i], pos_y[i]]))
#
# X = np.array(X)

X = np.array(V)
X[:, 2] = X[:, 2] * 2
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
# print(kmeans.labels_)



# print(kmeans.labels_)
# print(sum(kmeans.labels_==0))
#
# print(sum(kmeans.labels_==1))

normal_avg1 = np.array([0, 0, 0], dtype=float)
normal_avg2 = np.array([0, 0, 0], dtype=float)
count1 = 0
count2 = 0
for i in range(len(V)):
    if kmeans.labels_[i] == 0:
        # print(np.array(Normal[i]))
        if i in normal_map:
            normal_avg1 += np.array(normal_map[i])
            count1 += 1
    elif kmeans.labels_[i] == 1:
        if i in normal_map:
            normal_avg2 += np.array(normal_map[i])
            count2 += 1

ground_normal = normal_avg1/count1
print(normal_avg1/count1)


wall_normal = normal_avg2/count2
print(normal_avg2/count2)

normal_avg3 = np.array([0, 0, 0], dtype=float)
count3 = 0
for i in range(len(g_indx)):
    if g_indx[i] in normal_map:
        normal_avg3 += np.array(normal_map[g_indx[i]])
        count3 += 1

ground_normal2 = normal_avg3 / count3
print(ground_normal2)


ax = plt.figure().add_subplot(projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], c=kmeans.labels_)

ax.quiver(0, 0, 0, ground_normal2[0], ground_normal2[1], ground_normal2[2], length=5)
# ax.quiver(0, 0, 0, wall_normal[0], wall_normal[1], wall_normal[2], length=5)



location = [sum(ground_x)/len(ground_x), sum(ground_z)/len(ground_z), -2.5]
print('Location : ' + str(location))


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()