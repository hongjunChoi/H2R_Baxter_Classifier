
import numpy as np


#returns rotation matrix M from quaterniion
def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    m = np.zeros((3, 3))
    m[0][0] = 1- 2*qy*qy - 2*qz*qz
    m[0][1] = 2*qx*qy - 2*qw*qz
    m[0][2] = 2*qx*qz + 2*qw*qy

    m[1][0] = 2*qx*qy + 2*qw*qz
    m[1][1] = 1- 2*qx*qx - 2*qz*qz
    m[1][2] = 2*qy*qz - 2*qw*qx

    m[2][0] = 2*qx*qz - 2*qw*qy
    m[2][1] = 2*qy*qz + 2*qw*qx
    m[2][2] = 1- 2*qx*qx - 2*qy*qy
    return m




if __name__ == "__main__":
    qw = 1
    qx = 0
    qy = 0
    qz = 0
    rotation_matrix = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    direction_vector = np.dot(rotation_matrix, np.array([0, 0, 1]))
    direction_vector = direction_vector/np.sum(direction_vector)

    print {'x': direction_vector[0], 'y': direction_vector[1],'z': direction_vector[2]}
