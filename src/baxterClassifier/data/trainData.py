import glob
from xml.dom import minidom
import urllib
import socket
import re
import imghdr
import os




def main():
    socket.setdefaulttimeout(3)
    class1 = "hammer/hammer/"
    class2 = "scissor/scissor/"
    spoonList = glob.glob(class1+"*.xml")
    forkList = glob.glob(class2+"*.xml")
    spoonUrl = []
    forkUrl = []
    spoonSet = set()
    forkSet = set()
    for i in range(len(spoonList)):
        xml = spoonList[i].split("/")[2]
        img_id = xml.split(".")[0]
        print(img_id)
        spoonList[i] = img_id
        spoonSet.add(img_id)
    for j in range(len(forkList)):
        line = forkList[j]
        xml = line.split("/")
        img_id = xml[2].split(".")[0]
        print(img_id)
        forkList[j] = img_id
        forkSet.add(img_id)
    print(forkSet)
    print(spoonSet)
    # GET URLs
    with open("object_data_url.txt", "r") as ins:
        # OPEN WRTE FOR CSV FILE
        # Open a file
        fo = open("data1.csv", "wb")
        count = 0
        for line in ins:
            arr = line.split()
            img_id = arr[0]
            url = ""
            if img_id in spoonSet or img_id in forkSet:
                print("---")
                url = arr[1]
                # DOWNLAOD IMAGE
                localFileName = "images/" + img_id + ".png"
                print(url)
                try:
                    a = urllib.urlretrieve(url, localFileName)
                    # if (a[1]["Content-Type"])
                    if imghdr.what(localFileName) == None:
                        os.remove(localFileName)
                        continue
                except Exception:
                    continue
                if img_id in spoonSet:
                    print("A")
                    xmldoc = minidom.parse(class1 + img_id + ".xml")
                    classId = "a"
                elif img_id in forkSet:
                    print("B")
                    xmldoc = minidom.parse(class2 + img_id + ".xml")
                    classId = "b"
                # FIND THE ANNOTATION FILE
                ymin = xmldoc.getElementsByTagName(
                    'ymin')[0].firstChild.nodeValue
                ymax = xmldoc.getElementsByTagName(
                    'ymax')[0].firstChild.nodeValue
                xmin = xmldoc.getElementsByTagName(
                    'xmin')[0].firstChild.nodeValue
                xmax = xmldoc.getElementsByTagName(
                    'xmax')[0].firstChild.nodeValue
                data = classId + "," + \
                    str(ymin) + " , " + str(ymax) + \
                    "," + str(xmin) + "," + str(xmax) + \
                    "," + str(localFileName)
                # APPEND TO CSV FILE WITH DOWNLOADED URL & ANNOTATION (WRITE)
                print(data)
                fo.write(data + "\n")
            else:
                continue
        # CLOSE WRITE
        fo.close()
if __name__ == '__main__':
    main()
