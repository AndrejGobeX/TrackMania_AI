from matplotlib import pyplot as plt
import numpy as np


def ClosestPointOnLine(a, b, p):
    ap = p-a
    ab = b-a
    result = a + np.dot(ap,ab)/np.dot(ab,ab) * ab
    return result


x1 = np.random.rand()*10
y1 = np.random.rand()*10
z1 = np.random.rand()*10

x2 = np.random.rand()*10
y2 = np.random.rand()*10
z2 = np.random.rand()*10

x = np.random.rand()*10
y = np.random.rand()*10
z = np.random.rand()*10

projection = ClosestPointOnLine(
    np.array([x1, y1, z1]),
    np.array([x2, y2, z2]),
    np.array([x, y, z]),
)

# 3D
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot([x1, x2], [y1, y2], [z1, z2])
ax.plot([x, projection[0]], [y, projection[1]], [z, projection[2]], color='gray', linestyle='dashed')
ax.scatter(x, y, z, color='orange')

color = 'green'
check_out_of_bounds = lambda p, p1, p2 : p < min(p1, p2) or p > max(p1, p2)
if check_out_of_bounds(projection[0], x1, x2):
    color = 'red'

ax.scatter(projection[0], projection[1], projection[2], color=color)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.axis("equal")
plt.show()
