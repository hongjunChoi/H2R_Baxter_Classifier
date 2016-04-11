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



def variance_filter_rgb(r_var, g_var, b_var, r, g, b):
    if (r_var-r)*(r_var-r)+(g_var-g)*(g_var-g)+(b_var-b)*(b_var-b) > 100:
        return False
    return True


def variance_filter_z(z_var, z):
    if (z_var-z)*(z_var-z) > 200:
        return False
    return True


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


def main(fname, datafile):

    f = open(fname) 

    lines = []
    f.readline() 
    background_pose = None
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

    data = []

    x_max = 0
    y_max = 0
    y_min = float('inf')
    x_min = float('inf')
    z_min = float('inf')
    z_max = 0
    count1 = 0
    count2 = 0

    for x in range(0, col):
        for y in range(0, row):
            index = x + col * y;
            cell = observed_map.cells[index]
            z_mu = float(observed_map.cells[index].z.mu)
            

            if z_mu == 0.0:
                count1 = count1 + 1

            if z_mu > 0:
                count2 = count2 + 1
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

                r = float(observed_map.cells[index].red.mu)/255.0
                g = float(observed_map.cells[index].green.mu)/255.0
                b = float(observed_map.cells[index].blue.mu)/255.0
                bgr_array = convertYCrCB_BGR(b, g, r)
                b_mu = int(bgr_array[0])
                g_mu = int(bgr_array[1])
                r_mu = int(bgr_array[2])
                z_var = float(observed_map.cells[index].z.sigmasquared)
                r_var = float(observed_map.cells[index].red.sigmasquared)
                g_var = float(observed_map.cells[index].green.sigmasquared)
                b_var = float(observed_map.cells[index].blue.sigmasquared)

                point = {"x" : x*(cell_width), "y" : y*(cell_width), "z" : z_mu, "r": r_mu, 
                        "g" : g_mu, "b": b_mu, "z_var" : z_var, "r_var": r_var, "g_var": g_var, "b_var":b_var}
                data.append(point)

    print "z min : " + str(z_min) + "  / z max : " + str(z_max)
    print "the number of z = 0 are : " + str(count1) + " /  the number of positive z are : " + str(count2)

    data.append({"x_max" : x_max, "y_max":y_max, "x_min": x_min, "y_min": y_min, "z_min" : z_min, "z_max": z_max})   

    # Open a file for writings
    out_file = open(datafile, "w")

    # Save the dictionary into this file
    # (the 'indent=4' is optional, but makes it more readable)
    json.dump(data,out_file, indent=4)                                    

    # Close the file
    out_file.close()





if __name__ == "__main__":

    main(sys.argv[1], sys.argv[2])