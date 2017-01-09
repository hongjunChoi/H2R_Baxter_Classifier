import glob
from xml.dom import minidom
import urllib.request as myurl
import socket
import re
import imghdr
import os


def main():
    # TODO: change this part depending on class label
    class1 = "data/hammer/hammer/"
    class2 = "data/scissor/scissor/"

    class1List = glob.glob(class1 + "*.xml")
    class2List = glob.glob(class2 + "*.xml")
    class1Url = []
    class2Url = []
    class1Set = set()
    class2Set = set()

    for i in range(len(class1List)):
        xml = class1List[i].split("/")[3]
        img_id = xml.split(".")[0]
        class1List[i] = img_id
        class1Set.add(img_id)

    for j in range(len(class2List)):
        xml = class2List[j].split("/")[3]
        img_id = xml.split(".")[0]
        class2List[j] = img_id
        class2Set.add(img_id)

    print(class1Set)
    print(class2Set)

    # GET URLs
    socket.setdefaulttimeout(3)
    fo = open("data/data.csv", "w")
    with open("data/object_data_url.txt", "r") as urlFile:
        count = 0
        for line in urlFile:
            arr = line.split()
            img_id = arr[0]

            if img_id in class1Set or img_id in class2Set:
                url = arr[1]
                # DOWNLAOD IMAGE
                localFileName = "data/images/" + img_id + ".png"

                try:
                    print("fetching image .....", url)
                    myurl.urlretrieve(url, localFileName)

                    if imghdr.what(localFileName) == None:
                        if os.path.exists(localFileName):
                            os.remove(localFileName)
                        continue

                except Exception as e:
                    print("error in downloading image ...",
                          url, "  / error : ", e)
                    if os.path.exists(localFileName):
                        os.remove(localFileName)
                    continue

                if img_id in class1Set:
                    xmldoc = minidom.parse(class1 + img_id + ".xml")
                    classId = "0"
                elif img_id in class2Set:
                    xmldoc = minidom.parse(class2 + img_id + ".xml")
                    classId = "1"

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
