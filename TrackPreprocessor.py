from matplotlib import pyplot as plt
import numpy as np

track_left = []
track_right = []
tracks = [track_left, track_right]
centerline = []

file_name = "Maps/OxideStation.Map."
file_names = [file_name+"left", file_name+"right"]

for filename, track in zip(file_names, tracks):
    with open(filename, 'r') as file:
        for line in file:
            x, y, z, d = eval(line)
            track.append(np.array([x, y, z]))

i_left = 0
i_right = 0

for i_left in range(len(track_left)):
    while np.linalg.norm(track_right[i_right+1]-track_left[i_left]) < \
        np.linalg.norm(track_right[i_right]-track_left[i_left]):
        if i_right < len(track_right)-2:
            i_right += 1
        else:
            break
    centerline.append(
        (track_left[i_left] + track_right[i_right])/2
    )

# 3D
x_left, y_left, z_left = np.array(track_left).T
x_right, y_right, z_right = np.array(track_right).T
x_centerline, y_centerline, z_centerline = np.array(centerline).T

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(x_left, y_left, z_left, color='b')
ax.plot(x_right, y_right, z_right, color='orange')
ax.plot(x_centerline, y_centerline, z_centerline, color='g', linestyle='dashed')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.axis("equal")
plt.gca().invert_xaxis()
plt.show()

np.save("Maps/OxideStation.Map.centerline", np.array(centerline))