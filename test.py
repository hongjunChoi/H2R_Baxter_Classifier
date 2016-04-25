import numpy as np

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

    print quaternion_to_rotation_matrix(0.7071 ,  0.7071 , 0,  0)