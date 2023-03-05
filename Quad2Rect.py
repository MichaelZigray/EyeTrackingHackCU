import numpy as np
from skimage.transform import ProjectiveTransform
import matplotlib.pyplot as plt


# bottom_left = [58.6539, 31.512]
# top_left = [27.8129, 127.462]
# top_right = [158.03, 248.769]
# bottom_right = [216.971, 84.2843]


# t = ProjectiveTransform()
# src = np.asarray(
#     [bottom_left, top_left, top_right, bottom_right])
# dst = np.asarray([[0, 0], [0, 1], [1, 1], [1, 0]])
# t.estimate(src, dst)

# data = np.asarray([
#     [69.1216, 51.7061], [72.7985, 73.2601], [75.9628, 91.8095],
#     [79.7145, 113.802], [83.239, 134.463], [86.6833, 154.654],
#     [88.1241, 163.1], [97.4201, 139.948], [107.048, 115.969],
#     [115.441, 95.0656], [124.448, 72.6333], [129.132, 98.6293],
#     [133.294, 121.731], [139.306, 155.095], [143.784, 179.948],
#     [147.458, 200.341], [149.872, 213.737], [151.862, 224.782],
# ])
# data_local = t(data)

# print(data_local)


# plt.figure()
# plt.plot(src[[0,1,2,3,0], 0], src[[0,1,2,3,0], 1], '-')
# plt.plot(data.T[0], data.T[1], 'o')
# plt.figure()
# plt.plot(dst.T[0], dst.T[1], '-')
# plt.plot(data_local.T[0], data_local.T[1], 'o')
# plt.show()


# turns the above into a function. Returns a function that takes a point and returns the transformed point
from skimage.transform import ProjectiveTransform

def quadToRect(botleft, topleft, topright, botright, width, height):
    t = ProjectiveTransform()
    src = np.asarray([botleft, topleft, topright, botright])
    dst = np.asarray([[0, 0], [0, height], [width, height], [width, 0]])
    t.estimate(src,dst)
    
    def func(x,y):
        return t((x,y))
    
    return func

# test using the same points as above
# quadMap1 = quadToRect([58.6539, 31.512], [27.8129, 127.462], [158.03, 248.769], [216.971, 84.2843], 200, 200)
# print(quadMap1(69.1216, 51.7061))
# print(quadMap1(72.7985, 73.2601))
# print(quadMap1(75.9628, 91.8095))
# print(quadMap1(79.7145, 113.802))
# print(quadMap1(83.239, 134.463))
# print(quadMap1(86.6833, 154.654))
# print(quadMap1(88.1241, 163.1))
# print(quadMap1(97.4201, 139.948))
