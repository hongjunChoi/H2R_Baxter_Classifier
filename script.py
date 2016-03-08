import yaml
import sys

def main(fname):

        f = open(fname) 

        lines = []
        f.readline() 

        for line in f:
            if "background_pose" in line:
                continue
            lines.append(line)
        data = "\n".join(lines)
        
        ymlobject = yaml.load(data)

	print ymlobject


if __name__ == "__main__":
	main(sys.argv[1])
