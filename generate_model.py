import yaml
import sys
from base64 import b64decode
import zlib
import struct
import json
import numpy as np
import math
from scipy.ndimage.filters import gaussian_filter

class Observation:
    observationCount = 0
    occupancyCount = 0
    occupancyConfidence = 0.0
    r = 0.0
    g = 0.0
    b = 0.0

    def __init__(self):
        return

class GaussianMapChannel:
    binarySize = 40
    unpacker = struct.Struct('ddddd')

    @staticmethod
    def fromBinary(data):
        assert len(data) == GaussianMapChannel.binarySize, (len(data), GaussianMapChannel.binarySize)
        unpackedTuple = GaussianMapChannel.unpacker.unpack(data)
        #print "tuple", unpackedTuple
        return GaussianMapChannel(*unpackedTuple)

    def __init__(self, counts, squaredcounts, mu, sigmasquared, samples):
        self.counts = counts
        self.squaredcounts = squaredcounts
        self.mu = mu
        self.sigmasquared = sigmasquared
        self.samples = samples
    def __repr__(self):
        return "GaussianMapChannel(%.10f, %.10f, %.10f, %.10f, %.10f)" % (self.counts, self.squaredcounts,
                                         self.mu, self.sigmasquared,
                                         self.samples)

class GaussianMapCell:
    binarySize = GaussianMapChannel.binarySize * 4

    @staticmethod
    def fromBinary(data):
        assert len(data) == GaussianMapCell.binarySize
        return GaussianMapCell(GaussianMapChannel.fromBinary(data[0:GaussianMapChannel.binarySize]),
                               GaussianMapChannel.fromBinary(data[GaussianMapChannel.binarySize:GaussianMapChannel.binarySize * 2]),
                               GaussianMapChannel.fromBinary(data[GaussianMapChannel.binarySize * 2:GaussianMapChannel.binarySize * 3]),
                               GaussianMapChannel.fromBinary(data[GaussianMapChannel.binarySize * 3:GaussianMapChannel.binarySize * 4]))

    def __init__(self, red, green, blue, z):
        self.red = red
        self.green = green 
        self.blue = blue
        self.z = z
    def __repr__(self):
        return "GaussianMapCell(%s, %s, %s, %s)" % (repr(self.red), repr(self.green), repr(self.blue), repr(self.z))

class GaussianMap:
    @staticmethod
    def fromYaml(yamlGmap):


        cellData = readBinaryFromYaml(yamlGmap["cells"])
        print "cellData", len(cellData)

        expectedSize = yamlGmap["width"] * yamlGmap["height"] * GaussianMapCell.binarySize
        assert len(cellData) == expectedSize, (len(cellData), expectedSize)
        
        cells = []
        for i in range(yamlGmap["width"] * yamlGmap["height"]):
            cell = GaussianMapCell.fromBinary(cellData[i * GaussianMapCell.binarySize:(i + 1) * GaussianMapCell.binarySize])
            cells.append(cell)
        return GaussianMap(yamlGmap["width"], yamlGmap["height"],
                           yamlGmap["x_center_cell"], yamlGmap["y_center_cell"],
                           yamlGmap["cell_width"], cells)

    def __init__(self, width, height, x_center_cell, y_center_cell, cell_width, cells):
        self.width = width
        self.height = height
        self.x_center_cell = x_center_cell
        self.y_center_cell = y_center_cell
        self.cell_width = cell_width
        self.cells = cells
        



def readBinaryFromYaml(yamlList):
    data = "".join(yamlList)
    decoded = b64decode(data)
    decompressed = zlib.decompress(decoded)
    return decompressed

#hit_points = []
#old_planes = []
#new_planes = []

doubleunpacker = struct.Struct('d')
uintunpacker = struct.Struct('I')
def readMatFromYaml(fs):
    rows = fs["rows"]
    cols = fs["cols"]
    imgtype = fs["type"]
    print "type", imgtype
    m = cv2.cv.CreateMat(rows, cols, imgtype)
    
    if imgtype == 6:
        numpytype = na.float32
        unpacker = doubleunpacker
        #numpytype = na.uint32
        #unpacker = uintunpacker
    else:
        raise ValueError("Unknown image type: " + repr(imgtype))
    array = na.zeros(m.rows * m.cols * m.channels, dtype=numpytype)
    binary = readBinaryFromYaml(fs["data"])
    size = unpacker.size
    for i in range(len(array)):
        data = unpacker.unpack(binary[i*size:(i+1)*size])
        assert len(data) == 1
        array[i] = data[0]
    array = na.transpose(array.reshape((m.rows, m.cols, m.channels)), axes=(1,0,2))
    return array


##############################################################################################
##############################################################################################
#############                           CUSTOM FUNCTIONS                        ##############
##############################################################################################
##############################################################################################
def variance_filter_rgb(r_var, g_var, b_var, r, g, b):
    if (r_var-r)*(r_var-r)+(g_var-g)*(g_var-g)+(b_var-b)*(b_var-b) > 100:
        return False
    return True


def variance_filter_z(z_var, z):
    if (z_var-z)*(z_var-z) > 200:
        return False
    return True

# given a cube/3D grid of confidence interval and a slug from a single perspective, returns a hashtable of info 
# INPUT 
# filename : name of the slug yml file
# max_length : the length of cube to which we are mapping confidence score 
# this input was added to prevent large values of Z for slug scanning from the side(there is no table so we need to limit max z value)
#
# OUTPUT
# returns a hashtable with keys max_x, max_y, max_z, min_x, min_y, min_z, and position info
def get_slug_info(filename, max_length):
    info = {}
    f = open(filename) 
    lines = []
    f.readline() 

    pose_info = ""
    while True:
        c = f.read(1)
        pose_info = pose_info + c
        if c == "}":
            break   

    background_pose = pose_info.split('{')[1].split('}')[0]
    background_pose = background_pose.split(",")
    
    for i in range(len(background_pose)):
        background_pose[i] = background_pose[i].strip()


    start_line = pose_info.split("background_pose")[0].split("\n")
    for line in start_line:
        lines.append(line)

    for line in f:
        if "background_pose" in line:
            continue
        lines.append(line)


    data = "\n".join(lines)
    ymlobject = yaml.load(data)
    scene = ymlobject["Scene"]
    observed_map = GaussianMap.fromYaml(scene["observed_map"])

    row = observed_map.height
    col = observed_map.width
    cell_width = observed_map.cell_width
    width_len = observed_map.width*cell_width
    height_len = observed_map.height*cell_width

    x_max = 0
    y_max = 0
    y_min = float('inf')
    x_min = float('inf')
    z_min = float('inf')
    z_max = 0
    x_avg = 0
    y_avg = 0
    z_avg = 0
    count = 0

    info['position'] = { 'x' : float(background_pose[0].split(':')[1]), 'y' : float(background_pose[1].split(':')[1]), 
                        'z': float(background_pose[2].split(':')[1]), 'qw' : float(background_pose[3].split(':')[1]), 
                        'qx': float(background_pose[4].split(':')[1]), 'qy':  float(background_pose[5].split(':')[1]), 
                        'qz':  float(background_pose[6].split(':')[1])}

    for x in range((-1 * col / 2), (col / 2)):
        for y in range((-1 * row / 2), (row / 2)):
            index = (x + (col / 2)) + col * (y + (row / 2));
            cell = observed_map.cells[index]
            z_mu = float(observed_map.cells[index].z.mu)

            if z_mu > 0 and z_mu < max_length:
                #ray info
                ray_origin = get_ray_origin(info, x, y, cell_width)
                ray_direction = get_ray_direction(info)
                ray_x = ray_origin['x']
                ray_y = ray_origin['y']
                ray_z = ray_origin['z']
                direction_x = ray_direction['x']
                direction_y = ray_direction['y']
                direction_z = ray_direction['z']
                
                final_x = ray_x + direction_x * z_mu
                final_y = ray_y + direction_y * z_mu
                final_z = ray_z + direction_z * z_mu

                x_avg = x_avg + final_x
                y_avg = y_avg + final_y
                z_avg = z_avg + final_z
                count = count + 1

                if final_x > x_max:
                    x_max = final_x
                if final_x < x_min:
                    x_min = final_x

                if final_y > y_max:
                    y_max = final_y
                if final_y < y_min:
                    y_min = final_y
                
                if final_z < z_min:
                    z_min = final_z
                if final_z > z_max:
                    z_max = final_z


    info['cell_len'] = cell_width
    info['rows'] = row
    info['cols'] = col
    info['x_max'] = round(x_max, 3)
    info['x_min'] = round(x_min, 3)
    info['y_max'] = round(y_max, 3)
    info['y_min'] = round(y_min, 3)
    info['z_max'] = round(z_max, 3)
    info['z_min'] = round(z_min, 3)
    info['x_avg'] = round(float(x_avg/count), 3)
    info['y_avg'] = round(float(y_avg/count), 3)
    info['z_avg'] = round(float(z_avg/count), 3)
    

    return info

def get_view_info(filename):
    info = {}
    f = open(filename) 
    lines = []
    f.readline() 

    pose_info = ""
    while True:
        c = f.read(1)
        pose_info = pose_info + c
        if c == "}":
            break   

    background_pose = pose_info.split('{')[1].split('}')[0]
    background_pose = background_pose.split(",")
    
    for i in range(len(background_pose)):
        background_pose[i] = background_pose[i].strip()

    
    info['position'] = { 'x' :  float(background_pose[0].split(':')[1]), 'y' : float(background_pose[1].split(':')[1]), 
                        'z': float(background_pose[2].split(':')[1]), 'qw' : float(background_pose[3].split(':')[1]), 
                        'qx': float(background_pose[4].split(':')[1]), 'qy': float(background_pose[5].split(':')[1]), 
                        'qz': float(background_pose[6].split(':')[1])}


    return info



def get_info_from_top_view(file_name):
    return get_slug_info(file_name, float('inf'))



def get_old_ray_origin(slug_info, x, y, cell_length):
        
    default_pos = slug_info['position']

    rotation_matrix = quaternion_to_rotation_matrix(slug_info['position']['qw'],
                                                    slug_info['position']['qx'], 
                                                    slug_info['position']['qy'], 
                                                    slug_info['position']['qz'])

    #rotation matrix  has property  inverse = transpose
    vector = np.array([x*cell_length, y*cell_length, 0])
    current_vector = np.dot(rotation_matrix, vector)
    

    x = round(float(current_vector[0] + default_pos['x']), 3)
    y = round(float(current_vector[1] + default_pos['y']), 3)
    z = round(float(current_vector[2] + default_pos['z']), 3)

    return {'x' : x, 'y':  y  , 'z' : z}





def get_ray_origin(slug_info, x, y, cell_length):
    z_max = 0.388
    default_pos = slug_info['position']

    rotation_matrix = quaternion_to_rotation_matrix(slug_info['position']['qw'],
                                                    slug_info['position']['qx'], 
                                                    slug_info['position']['qy'], 
                                                    slug_info['position']['qz'])

    #rotation matrix  has property  inverse = transpose
    z_max = 0.388
    vector = np.array([x*cell_length, y*cell_length, z_max])
    current_vector = np.dot(rotation_matrix, vector)
    

    x = round(float(current_vector[0] + default_pos['x']), 3)
    y = round(float(current_vector[1] + default_pos['y']), 3)
    z = round(float(current_vector[2] + default_pos['z']), 3)
    return {'x' : x, 'y':  y  , 'z' : z}


#Given a yml file, read all the data, put the confidence score in the sparse map and return the updated sparse map
#INPUT : info hashtable obtained from get_slug_info(), yml file name, and sparse map hashtable 
#
#OUTPUT : updated sparse hash map 
def read_from_yml(file_name, sparse_map, slug_info, cube_info):
    f = open(file_name) 

    lines = []
    f.readline() 

    for line in f:
        if "background_pose" in line:
            continue
        lines.append(line)

    data = "\n".join(lines)

    ymlobject = yaml.load(data)
    scene = ymlobject["Scene"]
    observed_map = GaussianMap.fromYaml(scene["observed_map"])

    row = observed_map.height
    col = observed_map.width

    cell_length = observed_map.cell_width
    ray_direction = get_ray_direction(slug_info)

    print " ====== starting ray casting ========"

    for x in range((-1 * col / 2), (col / 2)):
        for y in range((-1 * row / 2), (row / 2)):
            index = (x + (col / 2)) + col * (y + (row / 2));
            cell = observed_map.cells[index]
            r = float(cell.red.mu)
            g = float(cell.green.mu)
            b = float(cell.blue.mu)
            z_mu = float(observed_map.cells[index].z.mu)

            #old_ray_origin = get_old_ray_origin(slug_info, x, y, cell_length)
            ray_origin = get_ray_origin(slug_info, x, y, cell_length)
            
            #if x % 14 == 0 and y % 14 == 0:
            #    data = { 'x' : ray_origin['x'] , 'y' : ray_origin['y'] , 'z' : ray_origin['z'], 'plane' : True}
            #    new_planes.append(data)

            #    new_data = {"x" : old_ray_origin['x'], 'y' : old_ray_origin['y'], 'z' : old_ray_origin['z']}
            #    old_planes.append(new_data)

            #print "z : :" + str(z_mu) + " insied the loop !"
            if z_mu > 0:  
                sparse_map = ray_cast(sparse_map, ray_origin, ray_direction, z_mu, cube_info, r, g, b)


    print "===== end of ray casting ========= "
    print "length of sparse map is " + str(len(sparse_map))

    return sparse_map


# Multiply 2 quaternions, r and q, together
# INPUT - r : {qx, qy, qz, qw}, q: {qx, qy, qz, qw}
# OUTPUT - {qx, qy, qz, qw}
def multiply_quaternion(r, q):
    t0 = r['qw']*q['qw'] - r['qx']*q['qx'] - r['qy']*q['qy'] - r['qz']*q['qz']
    t1 = r['qw']*q['qx'] + r['qx']*q['qw'] - r['qy']*q['qz'] + r['qz']*q['qy']
    t2 = r['qw']*q['qy'] + r['qx']*q['qz'] + r['qy']*q['qw'] - r['qz']*q['qx'] 
    t3 = r['qw']*q['qz'] - r['qx']*q['qy'] + r['qy']*q['qx'] + r['qz']*q['qw'] 
    return {'qw': t0, 'qx': t1, 'qy': t2, 'qz': t3}

def get_ray_direction(slug_info):
    quaternion = slug_info['position']
    quaternion = multiply_quaternion(quaternion , {"qx": 0, "qy" : 1, "qz" : 0, "qw" : 0})
    qw = quaternion['qw']
    qx = quaternion['qx']
    qy = quaternion['qy']
    qz = quaternion['qz']

    rotation_matrix = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    direction_vector = np.dot(rotation_matrix, np.array([0, 0, 1]))

    return {'x': direction_vector[0], 'y': direction_vector[1],'z': direction_vector[2]}



def encode_key(x, y, z):
    return str(int(x)) + "_" + str(int(y)) + "_" + str(int(z))


def decode_key(key):
    temp = key.split('_')
    return {'x': (int(temp[0])) , 'y': (int(temp[1])), 'z':(int(temp[2]))} 


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
    #return np.linalg.inv(m)
    return m

def convertYCrCB_BGR(y,cr,cb):
    data = []
    delta = 0.5
    r = min(1, max(0, y + 1.403*(cr - delta)))
    g = min(1, max(0, y - 0.714*(cr - delta) - 0.344*(cb - delta)))
    b = min(1, max(0, y + 1.733*(cb - delta)))
    r *= 255.0
    g *= 255.0
    b *= 255.0
    data.append(b)
    data.append(g)
    data.append(r)
    return data

# TODO : CODY SHOULD IMPLEMENT THIS FUNCTION
# INPUTS
#  - sparse_map : hashtable of data key being "x_y_z" and value being data that we store including confidence score. 
#                  use encode_key(), decode_key() function
#  - origin : hashmap with key x, y, z => global location where the ray starts
#  - direction : hashmap with key x, y, z => represents the unit direction vector of the ray
#  - z : float => represents the length of the ray
#  - cube info => hashtable of key 'size' => length of each edge of the cube 
#                 and 'cube_origin' => global location of (0, 0, 0) of the cube
#                 cube_info['cube_origin'] is also a dictionary with keys "x_origin", "y_origin", "z_origin"
#
# OUTPUT : an updated sparse_map hashtable with updated confidence score 
def ray_cast(sparse_map, origin, direction, z_len, cube_info, r, g, b):
    #ray info
    ray_x = origin['x']
    ray_y = origin['y']
    ray_z = origin['z']
    direction_x = direction['x']
    direction_y = direction['y']
    direction_z = direction['z']
    
    #cube info
    cube_origin_x = cube_info['cube_origin']['x_origin']
    cube_origin_y = cube_info['cube_origin']['y_origin']
    cube_origin_z = cube_info['cube_origin']['z_origin']
    grid_size = cube_info['grid_size']
    cell_width = cube_info['cell_width']

    delta_z = 0.001
    cumulative_z = 0.001
    previous = ""
    
    #final_x = ray_x + direction_x * z_len 
    #final_y = ray_y + direction_y * z_len
    #final_z = ray_z + direction_z * z_len 

    #data = {'x' : final_x, 'y' : final_y, 'z' : final_z }
    #hit_points.append(data)

    while cumulative_z <= z_len:
        
        curr_x = ray_x + direction_x * cumulative_z
        curr_y = ray_y + direction_y * cumulative_z
        curr_z = ray_z + direction_z * cumulative_z

        x = curr_x - cube_origin_x
        y = curr_y - cube_origin_y
        z = curr_z - cube_origin_z

        if (x >= 0) and (x <= grid_size * cell_width):
            if (y >= 0) and (y <= grid_size * cell_width):
                if (z >= 0) and (z <= grid_size * cell_width):
                    key_x = math.floor(x / cell_width)
                    key_y = math.floor(y / cell_width)
                    key_z = math.floor(z / cell_width)
                    key = encode_key(key_x, key_y, key_z)
                    
                    # add an occupied observation to this cell's observation object
                    if cumulative_z == z_len:
                        # in this case, there should already be something in sparse_map, so a null pointer shouldn't be thrown
                        if key == previous:
                            observation = sparse_map[key]
                            observation.occupancyCount = observation.occupancyCount + 1
                            observation.r = (observation.r * observation.observationCount + r) / (observation.observationCount + 1)
                            observation.g = (observation.g * observation.observationCount + g) / (observation.observationCount + 1)
                            observation.b = (observation.b * observation.observationCount + b) / (observation.observationCount + 1)
                            observation.occupancyConfidence = observation.occupancyCount / observation.observationCount
                            sparse_map[key] = observation
                            print "ray cast type 1"
                        # in this case, we need to check whether or not there's an observation object in sparse_map already
                        else:
                            if key in sparse_map:
                                observation = sparse_map[key]
                                observation.r = (observation.r * observation.observationCount + r) / (observation.observationCount + 1)
                                observation.g = (observation.g * observation.observationCount + g) / (observation.observationCount + 1)
                                observation.b = (observation.b * observation.observationCount + b) / (observation.observationCount + 1)
                                observation.observationCount = observation.observationCount + 1
                                observation.occupancyCount = observation.occupancyCount + 1
                                observation.occupancyConfidence = observation.occupancyCount / observation.observationCount
                                sparse_map[key] = observation
                                print "ray cast type 2"
                            else:
                                observation = Observation()
                                observation.r = (observation.r * observation.observationCount + r) / (observation.observationCount + 1)
                                observation.g = (observation.g * observation.observationCount + g) / (observation.observationCount + 1)
                                observation.b = (observation.b * observation.observationCount + b) / (observation.observationCount + 1)
                                observation.observationCount = 1
                                observation.occupancyCount = 1
                                observation.occupancyConfidence = observation.occupancyCount / observation.observationCount
                                sparse_map[key] = observation
                                print "ray cast type 3"
                            previous = key

                    # add an unoccupied observation to this cell's observation object
                    elif key != previous:
                        if key in sparse_map:
                            observation = sparse_map[key]
                            observation.observationCount = observation.observationCount + 1
                            observation.occupancyConfidence = observation.occupancyCount / observation.observationCount
                            sparse_map[key] = observation
                            print "ray cast type 4"
                        else:
                            observation = Observation()
                            observation.occupancyConfidence = 0
                            observation.observationCount = 1
                            sparse_map[key] = observation
                            print "ray cast type 5"
                        previous = key

        #ensures that we don't skip over checking the exact location of intersection
        if (cumulative_z != z_len) and ((cumulative_z + delta_z) >= z_len):
            print " \n\n\n======= IS THIS HAPPENING   ============\n\n\n"
            cumulative_z = z_len
        else:
            cumulative_z = cumulative_z + delta_z

    return sparse_map


def set_cube_dimension(top_down_view_info, padding, grid_size):

    x_size = max(abs(top_down_view_info['x_min'] - top_down_view_info['x_avg']), 
         abs(top_down_view_info['x_max'] - top_down_view_info['x_avg']) )

    y_size = max(abs(top_down_view_info['y_min'] - top_down_view_info['y_avg']), 
         abs(top_down_view_info['y_max'] - top_down_view_info['y_avg']) )

    z_size = max(abs(top_down_view_info['z_min'] - top_down_view_info['z_avg']), 
         abs(top_down_view_info['z_max'] - top_down_view_info['z_avg']) )

    half = max(x_size, y_size, z_size) * padding
    cube_size = float(2*half)
    cube_origin = { 'x_origin': top_down_view_info['x_avg'] - half,
                    'y_origin': top_down_view_info['y_avg'] - half, 
                    'z_origin': top_down_view_info['z_avg'] - half}

    #x_size = abs(top_down_view_info['x_min'] - top_down_view_info['x_max']) 
    #y_size = abs(top_down_view_info['y_min'] - top_down_view_info['y_max']) 
    #z_size = abs(top_down_view_info['z_min'] - top_down_view_info['z_max']) 
    

    #the global location of  (0, 0, 0) cell in the cube  
    #cube_origin = { 'x_origin': top_down_view_info['position']['x'] + top_down_view_info['x_min'], 
    #                'y_origin': top_down_view_info['position']['y'] + top_down_view_info['y_min'], 
    #                'z_origin': top_down_view_info['position']['z'] + top_down_view_info['z_min'] }

    cube_info = {'size' : cube_size, 'cube_origin' : cube_origin, 'grid_size' : grid_size, 'cell_width' : float(cube_size/grid_size)}
    return cube_info




################################################################################################
################################################################################################
################################################################################################

def main(top_view, other_views, file_name):
    # PAREMETER TUNING 
    GRID_SIZE = 1000 # there are GRID_SIZE^3 cells in the cube / number of smalls cubes in one edge
    PADDING_RATE = 1.2 # how much more space are we going to consider other than (min - max)
    THRESHOLD  = 0.2

    sparse_map = {}
    top_down_view_info = get_info_from_top_view(top_view)

    cube_info = set_cube_dimension(top_down_view_info, PADDING_RATE, GRID_SIZE)
    print "cube info : " + str(cube_info)

    # 1. RAY CAST FROM TOP DOWN VIEW SLUG
    print "===== reading from top down view ====== "
    sparse_map = read_from_yml(top_view, sparse_map, top_down_view_info, cube_info)


    # 2. RAY CAST FROM OTHER VIEWS 
    for other_view in other_views:
        view_info = get_view_info(other_view)
        print " other view info : " + str(view_info) + "\n\n"
        sparse_map = read_from_yml(other_view, sparse_map, view_info, cube_info)


    # 3. WRITE SPARSE MAP INTO JSON FILE
    data = []
    n_count = 0
    print " ======  writing to file ======"
    for key in sparse_map:
        position = decode_key(key)
        y = float(sparse_map[key].b)/255.0
        cr = float(sparse_map[key].g)/255.0
        cb = float(sparse_map[key].r)/255.0
        bgr_array = convertYCrCB_BGR(y, cr, cb)
        b_mu = int(bgr_array[0])
        g_mu = int(bgr_array[1])
        r_mu = int(bgr_array[2])
        
        print sparse_map[key].occupancyConfidence 

        if sparse_map[key].occupancyConfidence < THRESHOLD:
            n_count = n_count + 1

        if sparse_map[key].occupancyConfidence >= 0:
            data.append({'x': position['x']*cube_info["cell_width"] , 'y': position['y']*cube_info["cell_width"] , 'z': position['z']*cube_info["cell_width"] , 
                'score': sparse_map[key].occupancyConfidence , 'r' : r_mu, 'g': g_mu, 'b': b_mu})


    print " noise count is : " + str(n_count)

    out_file = open(file_name, "w")

    # Save the dictionary into this file
    # (the 'indent=4' is optional, but makes it more readable)

    #final_data = {"data" : data , "info" : top_down_view_info}
    #json.dump(final_data, out_file, indent=4)                                    

    #temp = new_planes + old_planes

    print " ================== ABOUT TO WRTIE TO FILE =============="
    print "length of final sparse map  : " + str(len(sparse_map))
    print "length of data written to json after threshold  : " + str(len(data))
    json.dump(data, out_file, indent=4) 
    
    # Close the file
    out_file.close()

    return 


if __name__ == "__main__":
    top_down_file = sys.argv[1]
    other_views = []
    l = len(sys.argv)

    if l < 2:
        print "at least one slug file and output file name is needed"
    else: 

        file_name = sys.argv[l-1];

        for i in range(2,l-1):
            other_views.append(sys.argv[i])


        print "processing " + str(l-2) + " yaml files and creating " +  str(file_name) + "  json file..." 

        main(top_down_file, other_views, file_name )

   