import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from random import randrange

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

f = open('../Model/scene/campus/untitled.obj', 'r')
out = open("../Model/scene/campus/ground.obj", 'w')

# f = open('../Model/dorm/untitled.obj', 'r')
# out = open("../Model/dorm/ground.obj", 'w')
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
kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(X)
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
print('Ground V num = ' + str(len(g_indx)))
for i in range(len(g_indx)):
    if g_indx[i] in normal_map:
        normal_avg3 += np.array(normal_map[g_indx[i]])
        count3 += 1

ground_normal2 = normal_avg3 / count3
print(ground_normal2)

# RANSAC
ransac_normal = np.array([0, 0, 1], dtype=float)
tmp_normal = np.array([0, 0, 1], dtype=float)
ransac_iter = 0
ransac_max_iter = 5000
consistency = 0
ratio = 0.8
N = len(g_indx)
group_num = int(N * ratio)
print(group_num)

begin = time.time()
consist = []
while ransac_iter < ransac_max_iter:

    while tmp_normal.all() == 0:
        pick = randrange(0, N)

        if g_indx[pick] in normal_map:
            tmp_normal = normal_map[g_indx[pick]]
            # tmp_normal /= np.linalg.norm(tmp_normal)

    # Vote
    tmp_consist = 0
    tmp_count = 0
    tmp_dist = 0
    for _ in range(group_num):
        pick = randrange(0, N)
        if g_indx[pick] in normal_map:
            p_normal = normal_map[g_indx[pick]]
            # p_normal /= np.linalg.norm(p_normal)
            tmp_consist += np.dot(p_normal, tmp_normal)
            # tmp_dist += numpy.linalg.norm(tmp_normal)
            tmp_count += 1

    tmp_consist /= tmp_count

    if tmp_consist > consistency or ransac_normal.all() == 0:
        consist.append((ransac_iter, tmp_consist))
        consistency = tmp_consist
        ransac_normal = tmp_normal

    ransac_iter += 1
    tmp_normal = np.array([0, 0, 0], dtype=float)

print('consistency : ' + str(consistency))
print(ransac_normal)


end = time.time()

# total time taken
print(f"Total runtime of the program is {end - begin}")

# PLOT consist
consist = np.array(consist)
# print(consist)
# print(consist[0, :])
plt.plot(consist[:, 0], consist[:, 1])
plt.xlabel("Iteration")
plt.ylabel("Consistency")
plt.show()


# GROUND CHECK
# mean1 = np.array([0, 0, 0], dtype=float)
# mean2 = np.array([0, 0, 0], dtype=float)
#
# max_ground = float('-inf')
# min_ground = float('inf')
# for i in range(len(X)):
#     if kmeans.labels_[i] == 0:
#         mean1 += X[i]
#     elif kmeans.labels_[i] == 1:
#         mean2 += X[i]
#
#         max_ground = max(max_ground, X[i, 2])
#         min_ground = min(min_ground, X[i, 2])
#
# mean1 /= sum(kmeans.labels_ == 0)
# mean2 /= sum(kmeans.labels_ == 1)
#
# print('mean1 = ' + str(mean1))
# print('mean2 = ' + str(mean2))
#
# print('max ground = ' + str(max_ground))
# print('min ground = ' + str(min_ground))

# PLOT Model, Surface Normal
# ax = plt.figure().add_subplot(projection='3d')
# ax.scatter(X[:,0], X[:,1], X[:,2], c=kmeans.labels_)
# g_indx = np.array(g_indx)
# # ax.scatter(X[g_indx,0], X[g_indx,1], X[g_indx,2], color='green')
# ax.quiver(0, -2, 2, ransac_normal[0], ransac_normal[1], ransac_normal[2], color='red', length=5, zorder=10)
# # ax.quiver(0, 0, 0, ground_normal2[0], ground_normal2[1], ground_normal2[2], length=5, color='green')
#
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()