import sys
import numpy as np
import math


if __name__ == "__main__":

    l = len(sys.argv)

    if l < 2:
        print "at least one slug file and output file name is needed"
        return
