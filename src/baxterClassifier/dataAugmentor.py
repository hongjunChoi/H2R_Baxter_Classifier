import random
import cv2
import time
import math
import numpy as np
import glob
import random
import os
import imutils
from PIL import Image
import PIL.ImageOps


def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def randomCropImage(image):
    # given an image, return an randomly cropped image
    # get lower bound and upper bound for x and y
    height, width = image.shape[:2]
    xLowerBound = int(width * (random.random() / 5.0))
    xUpperBound = int(width * (0.8 + random.random() / 5.0))
    yLowerBound = int(height * (random.random() / 5.0))
    yUpperBound = int(height * (0.8 + random.random() / 5.0))

    croppedImage = image[yLowerBound:yUpperBound, xLowerBound:xUpperBound]
    return croppedImage


def perturbateColor(image):
    return 255 - image


def augmentColor(image):
    rand = random.random() * 10 + 20
    multiplier = 1
    if random.random > 0.5:
        multiplier = -1

    index = int(round(random.random() * 2))

    shape = image.shape
    x = shape[0]
    y = shape[1]
    for i in range(x):
        for j in range(y):
            image[i][j][index] = int(
                image[i][j][index] + int(multiplier * rand))

    return image


def augmentImages(path):
    # get list of all image paths in directory
    imagePathList = glob.glob(path + "/*.jpg")

    # TODO : Loop through each image

    for imagePath in imagePathList:
        original_image = cv2.imread(imagePath)
        original_height, original_width = original_image.shape[0:2]
        max_length = original_height
        if original_width > max_length:
            max_length = original_width

        ratio = int(max_length / 256)

        # TODO : reshape the size if greater
        count = 0
        image = cv2.resize(original_image, (int(original_height / ratio),
                                            int(original_width / ratio)), interpolation=cv2.INTER_AREA)

        # TODO : For each image, rotate and color permutate image
        for i in np.arange(0, 360, 45):
            if i is not 0:
                image_height, image_width = image.shape[0:2]
                image_orig = np.copy(image)
                image_rotated = rotate_image(image, i)
                image_rotated_cropped = crop_around_center(
                    image_rotated,
                    *largest_rotated_rect(
                        image_width,
                        image_height,
                        math.radians(i)
                    )
                )

                saveImage(image_rotated_cropped, imagePath, count)
                count += 1

                if i % 90 == 0:
                    saveImage(randomCropImage(
                        image_rotated_cropped), imagePath, count)
                    count += 1

                    saveImage(randomCropImage(
                        image_rotated_cropped), imagePath, count)
                    count += 1

                    saveImage(randomCropImage(
                        image_rotated_cropped), imagePath, count)
                    count += 1

                saveImage(augmentColor(
                    image_rotated_cropped), imagePath, count)
                count += 1

                saveImage(augmentColor(
                    image_rotated_cropped), imagePath, count)
                count += 1

    return


def divideImageSet(classPath1, classPath2):
    # For directory containing images of each class, combine all images path
    # create custom_training_set.csv with training image path and label
    # create custom_test_set.csv with test image path and label
    class1List = glob.glob(classPath1 + "/*")
    class2List = glob.glob(classPath2 + "/*")

    combined_list = []
    for class1_path in class1List:
        combined_list.append([class1_path, 0])

    for class2_path in class2List:
        combined_list.append([class2_path, 1])

    data = [(random.random(), l) for l in combined_list]
    data.sort()

    train_data = data[0:int(len(data) * 0.9)]
    test_data = data[int(len(data) * 0.9):]

    with open('data/imagenet_spoon_train_data.csv', 'w') as train_file:
        for data in train_data:
            line = str(data[1][0]) + " , " + str(data[1][1]) + "\n"
            train_file.write(line)
    with open('data/imagenet_spoon_test_data.csv', 'w') as test_file:
        for data in test_data:
            line = str(data[1][0]) + " , " + str(data[1][1]) + "\n"
            test_file.write(line)

    return


def saveImage(image, targetPath, count):
    new_name = targetPath + "_" + str(count) + ".jpg"
    cv2.imwrite(new_name, image)
    return


def create_synthetic_data(path, backgroundPath, targetPath, autoMasking=False):
    backgroundPath = backgroundPath + "/*.png"
    backgrounds_paths = glob.glob(backgroundPath)
    object_id = 0
    paths = glob.glob(path + '/*')

    if autoMasking is True:
        paths = glob.glob(path + '/*.jpg')

    for path in paths:
        maskPath = path + "/mask.png"
        objectPath = path + "/rgb.png"

        if autoMasking is True:
            objectPath = path
            maskPath = objectPath

        object_id = object_id + 1

        for background in backgrounds_paths:

            print("\n\nmask path : " + str(maskPath) +
                  "   / object path : " + str(objectPath) + " / background path : " + str(background) + " \n\n")

            background_id = background.split('/')[3].split('.')[0].strip()
            savePath = targetPath + "/" + \
                str(object_id).strip() + "_" + background_id

            backGroundSelection(background, objectPath, maskPath, savePath)


def backGroundSelection(backgroundPath, imagePath, maskPath, targetPath):
    # Load two images
    bg = cv2.imread(backgroundPath)
    bg_shape = bg.shape
    count = 0

    for angle in np.arange(0, 360, 45):
        for i in range(2):
            # get random portion from background

            if bg_shape[0] < 180 or bg_shape[1] < 180:

                backgroundImage = cv2.resize(
                    bg, (180, 180), interpolation=cv2.INTER_AREA)
            else:
                bg_x = int((bg_shape[0] - 180) * random.random())
                bg_y = int((bg_shape[1] - 180) * random.random())
                print("x : " + str(bg_x) + "/  y : " + str(bg_y))
                backgroundImage = bg[
                    bg_x:bg_x + 180, bg_y:bg_y + 180]

            image = cv2.imread(imagePath)
            object_mask = cv2.imread(maskPath)

            background_x = 180
            background_y = 180

            bImage = np.copy(backgroundImage)
            maskImage = np.copy(object_mask)

            objectImage = imutils.rotate_bound(image, angle)
            maskImage = imutils.rotate_bound(maskImage, angle)

            if objectImage.shape[0] > background_x or objectImage.shape[1] > background_y:
                ratio = math.ceil(max(float(objectImage.shape[
                    0]) / float(background_x), float(objectImage.shape[1]) / float(background_y)))
                ratio = int(ratio)
                objectImage = cv2.resize(
                    objectImage, (int(objectImage.shape[0] / (ratio)), int(objectImage.shape[1] / (ratio))), interpolation=cv2.INTER_AREA)
                maskImage = cv2.resize(
                    maskImage, (int(maskImage.shape[0] / (ratio)), int(maskImage.shape[1] / (ratio))), interpolation=cv2.INTER_AREA)

            img2gray = cv2.cvtColor(maskImage, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 50, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            rows, cols, channels = objectImage.shape
            x = int((background_x - rows) * random.random())
            y = int((background_y - cols) * random.random())
            roi = bImage[x:x + rows, y:y + cols]

            # Now black-out the area in ROI
            img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            # Take only region of object from object image.
            img2_fg = cv2.bitwise_and(objectImage, objectImage, mask=mask)

            # Put cropped object in ROI and modify the main image
            dst = cv2.add(img1_bg, img2_fg)
            bImage[x:x + rows, y:y + cols] = dst

            saveImage(bImage, targetPath, count)
            count += 1

            # saveImage(randomCropImage(
            #     bImage), targetPath, count)
            # count += 1

            # saveImage(augmentColor(
            #     bImage), targetPath, count)
            # count += 1

            # saveImage(augmentColor(
            #     bImage), targetPath, count)
            # count += 1


def create_anti_label_image(data_path, target_path, label):
    label_paths = glob.glob(data_path + "/*")
    count = 0

    for path in label_paths:
        temp = path.split('/')
        if label.strip().lower() not in temp[len(temp) - 1].strip().lower():
            image_path = path + "/*.jpg"
            images = glob.glob(image_path)
            for i in range(30):
                img = cv2.imread(images[i])
                saveImage(img, target_path, count)
                count = count + 1

    return


if __name__ == "__main__":
    # Create Dataset of 'non spoons'
    # create_anti_label_image("data/caltech_dataset",
    #                         "data/TRAIN/not_spoon", "spoon")

    # create_synthetic_data("data/TRAIN",
    # "data/fromJohn/deepBackgrounds", "data/TRAIN/spoon", autoMasking=True)
    divideImageSet("data/TRAIN/imagenet_spoon", "data/TRAIN/not_spoon")

    # create_synthetic_data("data/fromJohn/deepWoodenSpoons",
    #                       "data/fromJohn/deepBackgrounds", "data/synthetic_spoon")

    # create_synthetic_data("data/fromJohn/deepMarkers",
    #                       "data/fromJohn/deepBackgrounds", "data/synthetic_marker")

    # divideImageSet("data/synthetic_spoon", "data/synthetic_marker")
