import yaml
import sys
from base64 import b64decode
import zlib
import struct
import json


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
    f = open(fname) 
    lines = []
    f.readline() 
    background_pose  = None
    for line in f:
        if "background_pose" in line:
            background_pose = line.split('{')[1].split('}')[0]
            background_pose = background_pose.replace(".", "")
            background_pose = background_pose.split(",")
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

    for x in range(0, col):
        for y in range(0, row):
            index = x + col * y;
            cell = observed_map.cells[index]
            z_mu = float(observed_map.cells[index].z.mu)
            
            if z_mu > 0 and z_mu < max_length:
                x_len = x*(cell_width) 
                y_len = y*(cell_width)
                if x_len > x_max:
                    x_max = x_len
                if y_len > y_max:
                    y_max = y_len
                if y_len < y_min:
                    y_min = y_len
                if x_len < x_min:
                    x_min = x_len
                if z_mu < z_min:
                    z_min = z_mu
                if z_mu > z_max:
                    z_max = z_mu

    info['cell_len'] = cell_width
    info['rows'] = row
    info['cols'] = col
    info['x_max'] = round(x_max, 3)
    info['x_min'] = round(x_min, 3)
    info['y_max'] = round(y_max, 3)
    info['y_min'] = round(y_min, 3)
    info['z_max'] = round(z_max, 3)
    info['z_min'] = round(z_min, 3)
    info['position'] = { 'x' : float(background_pose[0].split(':')[1]), 'y' : float(background_pose[1].split(':')[1]), 
                        'z': float(background_pose[2].split(':')[1]), 'qw' :float(background_pose[3].split(':')[1]), 
                        'qx': float(background_pose[4].split(':')[1]), 'qy':float(background_pose[5].split(':')[1]), 
                        'qz': float(background_pose[6].split(':')[1])}
    return info



def get_info_from_top_view(file_name):
    return get_slug_info(file_name, float('inf'))



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


    for x in range(0, col):
        for y in range(0, row):
            index = x + col * y;
            cell = observed_map.cells[index]
            z_mu = float(observed_map.cells[index].z.mu)
            
            if z_mu > 0 and z_mu < cube_info['size']:
                ray_origin = get_ray_origin(slug_info, x, y, cell_length)
                ray_direction = get_ray_direction(slug_info)
                sparse_map = ray_cast(sparse_map, ray_origin, ray_direction, z_mu)

    return sparse_map



def q_to_euler(qw, qx, qy, qz):




def ray_cast(sparse_map, origin, direction, z):



def get_ray_direction(slug_info):



def get_ray_origin(slug_info, x, y, cell_length):




def encode_key(x, y, z):
    return str(round(x, 3)) + "_" + str(round(y, 3)) + "_" + str(round(z, 3))

def decode_key(key):
    temp = key.split('_')
    return {'x': round(float(temp[0]),3) , 'y': round(float(temp[1]),3), 'z':round(float(temp[2]),3)} 





################################################################################################
################################################################################################
################################################################################################

def main(top_view, other_views, file_name):
    # 0. CREATE CUBE DIMENSION AND SPARSE MAP 
    # setting spase map and size of the cube from the top down view 
    # from the top down view, obtain the dimensions of the cube
    # for now, I set (0, 0, 0) location of the cube as 0.8 * (min x, min y, min z) 
    # also the length (size) of the cube is set to be 1.5 * max(x_max - x_min, y_max - y_min, z_max - z_min)
    # TODO: CODY needs to change this as necessary

    sparse_map = {}
    top_down_view_info = get_info_from_top_view(top_view)
    #the global location of  (0, 0, 0) cell in the cube  
    cube_origin = {'x_origin': 0.8*top_down_view_info['x_min'], 'y_origin': 0.8*top_down_view_info['y_min']
                    'z_origin' : 0.8*top_down_view_info['z_min']}

    # the size of the cube edge (length of the edge of cube)
    cube_size = 1.5 * max(top_down_view_info['x_max']-top_down_view_info['x_min'], 
                        top_down_view_info['y_max']-top_down_view_info['y_min'],
                        top_down_view_info['z_max']-top_down_view_info['z_min'])

    cube_info = {'size' : cube_size, 'cube_origin' : cube_origin}

    # 1. RAY CAST FROM TOP DOWN VIEW SLUG
    sparse_map = read_from_yml(top_view, sparse_map, top_down_view_info, cube_info)

    # 2. RAY CAST FROM OTHER VIEWS 
    for other_view in other_views:
        view_info = get_slug_info(other_view, cube_size)
        sparse_map = read_from_yml(other_view, sparse_map, view_info, cube_info)


    # 3. WRITE SPARSE MAP INTO JSON FILE
    data = []
    for key in sparse_map:
        position = decode_key(key)
        data.append({'x': position['x'] , 'y': position['y'] , 'z': position['z'], 'score': sparse_map[key]})

    out_file = open(datafile, "w")

    # Save the dictionary into this file
    # (the 'indent=4' is optional, but makes it more readable)
    json.dump(data,file_name, indent=4)                                    

    # Close the file
    out_file.close()




if __name__ == "__main__":
    top_down_file = sys.argv[1]
    other_views = []
    l = len(sys.argv)

    if l < 2:
        print "at least one slug file and output file name is needed"
        return 

    file_name = sys.argv[l-1];

    for i in range(2,l-1):
        other_views.append(sys.argv[i])


    print "processing " + str(l-2) + " yaml files and creating " +  str(file_name) + "  json file..." 

    main(top_down_file, other_views, file_name )

   