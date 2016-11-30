import glob


def main():

    spoonList = glob.glob("data/spoon_annotation/n04284002/*.xml")
    forkList = glob.glob("data/fork_annotation/n03384167/*.xml")

    spoonUrl = []
    forkUrl = []

    spoonSet = set()
    forkSet = set()

    for i in range(len(spoonList)):
        xml = spoonList[i].split("/")[3]
        if len(xml) is not 4:
            continue

        img_id = xml.split(".")[0]
        spoonList[i] = (img_id)
        spoonSet.add(img_id)

    for j in range(len(forkList)):
        line = forkList[j]
        xml = line.split("/")

        if len(xml) is not 4:
            continue

        img_id = xml[3].split(".")[0]
        forkList[j] = (img_id)
        forkSet.add(img_id)

    # GET URLs
    with open("data/object_data_url.txt", "r") as ins:
        # OPEN WRTE FOR CSV FILE

        for line in ins:
            arr = line.split(" ")
            img_id = arr[0]
            url = ""
            if img_id in spoonSet | | img_id in forkSet:
                url = arr[1]

                # DOWNLAOD IMAGE

                # RENAMGE IMG TO IMG_ID.jpg

                # PUT IN GIVEN DIR

            else:
                continue

            # FIND THE ANNOTATION FILE

            # GET ANNOTATION DATA

            # APPEND TO CSV FILE WITH DOWNLOADED URL & ANNOTATION (WRITE)

        # CLOSE WRITE

    # WRITE IN FORMAT

if __name__ == '__main__':
    main()
