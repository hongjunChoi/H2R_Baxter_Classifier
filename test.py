if __name__ == "__main__":
    import numpy as np
    qw = 1
    qx = 0
    qy = 0
    qz = 0  
    test = qx*qy + qz*qw;
    heading = 0
    attitude = 0
    bank = 0

    if test > 0.499:  # singularity at north pole
        heading = 2 * np.arctan2(qx, qw);
        attitude = np.pi/2;
        bank = 0;

    
    elif test < -0.499: # singularity at south pole
        heading = -2 * np.arctan2(qx, qw)
        attitude = - np.pi/2
        bank = 0

    else:
        sqx = qx*qx
        sqy = qy*qy
        sqz = qz*qz
        heading = np.arctan2(2*qy*qw-2*qx*qz , 1 - 2*sqy - 2*sqz);
        attitude = np.arcsin(2*test);
        bank = np.arctan2(2*qx*qw-2*qy*qz , 1 - 2*sqx - 2*sqz)

    x = np.cos(heading) * np.cos(attitude)
    y = np.sin(heading) * np.cos(attitude)
    z = np.sin(attitude)
    print "x : " + str(x) + " y :" + str(y) + "z : " + str(z)
