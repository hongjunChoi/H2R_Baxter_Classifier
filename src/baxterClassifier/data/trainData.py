import glob
from xml.dom import minidom
import urllib
import socket
import Image


def main():
    socket.setdefaulttimeout(10)
    spoonList = glob.glob("spoon_annotation/n04284002/*.xml")
    forkList = glob.glob("fork_annotation/n03384167/*.xml")

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
        fo = open("data.csv", "wb")

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
                    urllib.urlretrieve(url, localFileName)

                except Exception as e:
                    print("invalid url")
                    continue

                print("...... ")
                if img_id in spoonSet:
                    print("---spoon")
                    xmldoc = minidom.parse(
                        "spoon_annotation/n04284002/" + img_id + ".xml")

                elif img_id in forkSet:
                    print("---fork")
                    xmldoc = minidom.parse(
                        "fork_annotation/n03384167/" + img_id + ".xml")

                # FIND THE ANNOTATION FILE

                ymin = xmldoc.getElementsByTagName(
                    'ymin')[0].firstChild.nodeValue
                ymax = xmldoc.getElementsByTagName(
                    'ymax')[0].firstChild.nodeValue
                xmin = xmldoc.getElementsByTagName(
                    'xmin')[0].firstChild.nodeValue
                xmax = xmldoc.getElementsByTagName(
                    'xmax')[0].firstChild.nodeValue

                data = "n04284002" + "," + \
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
