#defines the number of small cubes that make the length of each side
NUMBER_SMALL_CUBES_PER_SIDE = 1000
#defines how much margin we give when reading in our the first scan of data points
ENCOMPASS_RATIO = 0.75
#defines how confident we want to be in order to display a point
CONFIDENCE_THRESHOLD = 0.25

class Observation:
	#the double, occupancyConfidence, should range from -1 to 1, where -1 is absolutely certain there's nothing there and 1 is absolutely certain that something is there
	def __init__(self, occupancyConfidence):
		self.occupancyConfidence = occupancyConfidence

class Cube:
	#map of coordinate strings to list of observation objects
	occupancyMap = {}

	def __init__(self, initialJSONFile):
		#using the intial json file to center our cube
		#TODO: use initialJSONFile data to determine the below values
		avg_x = 
		avg_y = 
		avg_z = 
		min_x = 
		min_y = 
		min_z = 
		max_x = 
		max_y = 
		max_z = 
		initialSmallCubeSpan = ENCOMPASS_RATIO * NUMBER_SMALL_CUBES_PER_SIDE
		#Using the max guarantees us at least a margin of size defined by 1 - ENCOMPASS_RATIO
		self.smallCubeLength = max(((max_x - min_x) / initialSmallCubeSpan), ((max_y - min_y) / initialSmallCubeSpan), ((max_z - min_z) / initialSmallCubeSpan))
		self.largeCubeLength = NUMBER_SMALL_CUBES_PER_SIDE * self.smallCubeLength
		self.xLeftPlane = avg_x - (NUMBER_SMALL_CUBES_PER_SIDE / 2) * self.smallCubeLength
		self.xRightPlane = avg_x + (NUMBER_SMALL_CUBES_PER_SIDE / 2) * self.smallCubeLength
		self.yBottomPlane = avg_y - (NUMBER_SMALL_CUBES_PER_SIDE / 2) * self.smallCubeLength
		self.yTopPlane = avg_y + (NUMBER_SMALL_CUBES_PER_SIDE / 2) * self.smallCubeLength
		self.zFarPlane = avg_z - (NUMBER_SMALL_CUBES_PER_SIDE / 2) * self.smallCubeLength
		self.zNearPlane = avg_z + (NUMBER_SMALL_CUBES_PER_SIDE / 2) * self.smallCubeLength

	def rayCaster(originX, originY, originZ, dirX, dirY, dirZ, z):
		#TODO: find all small cubes in Cube that this ray intersects with, and add observations to occupancyMap.
		#QUESTION: Where is the best place to incorporate confidence levels? Maybe take in a distribution for "z" as opposed to just the mean "z"

	def getOccupancyMap():
		return self.occupancyMap

#topDownViewJSONFile is used to instantiate our cube, allJSONFiles is something like *.json which includes the topDownViewJSONFile
def main(topDownViewJSONFile, allJSONFiles):	
	cube = Cube(topDownViewJSONFile)
	for jsonFile in allJSONFiles:
		#TODO: go through each JSON file and call cube.rayCaster on each scan's (x, y) grid square, feeding the appropriate origin position, direction, and z distribution
		
	#TODO: after all data is added to the occupancyMap, go through the cube's map and write all points that have an occupancyConfidence higher than CONFIDENCE_THRESHOLD
	# maybe a csv file with columns: "x,y,z,confidence"
	#QUESTION: do we want to also write color, and if so, where do we incorporate it in this data-parsing work-flow?
	occupancyMap = cube.getOccupancyMap()

if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2])
