import random
import cv2
import time
import math
import numpy as np
import glob
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


def demo():
    """
    Demos the largest_rotated_rect function
    """

    image = cv2.imread("data/test_caltech/both1.jpeg")
    image_height, image_width = image.shape[0:2]

    cv2.imshow("Original Image", image)

    print("Press [enter] to begin the demo")
    print("Press [q] or Escape to quit")

    key = cv2.waitKey(0)
    if key == ord("q") or key == 27:
        exit()

    for i in np.arange(0, 360, 0.5):
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

        key = cv2.waitKey(2)
        if(key == ord("q") or key == 27):
            exit()

        # cv2.imshow("Original Image", image_orig)
        # cv2.imshow("Rotated Image", image_rotated)
        cv2.imshow("Cropped Image", image_rotated_cropped)

    print("Done")


def rotateSaveDemo():
    image = cv2.imread("data/test_caltech/both1.jpeg")
    image_height, image_width = image.shape[0:2]

    rotationDegree = 45
    image_orig = np.copy(image)
    image_rotated = rotate_image(image, rotationDegree)
    image_rotated_cropped = crop_around_center(
        image_rotated,
        *largest_rotated_rect(
            image_width,
            image_height,
            math.radians(rotationDegree)
        )
    )

    cv2.imwrite("test_image_save.jpeg", image_rotated_cropped)
    return


def randomCropImage(image):
    # TODO : given an image, return an randomly cropped image
    # get lower bound and upper bound for x and y
    height, width = image.shape[:2]
    xLowerBound = int(width * (random.random() / 4.0))
    xUpperBound = int(width * (0.75 + random.random() / 4.0))
    yLowerBound = int(height * (random.random() / 4.0))
    yUpperBound = int(height * (0.75 + random.random() / 4.0))

    croppedImage = image[yLowerBound:yUpperBound, xLowerBound:xUpperBound]
    return croppedImage


def perturbateColor(image):
    return 255 - image


def saveImage(image, originalPath, count):
    name = originalPath.split(".")
    new_name = name[0] + "_" + str(count) + ".jpg"
    cv2.imwrite(new_name, image)
    return


def augmentImages(path):
    # get list of all image paths in directory
    imagePathList = glob.glob(path + "/*.jpg")

    # TODO : Loop through each image
    for imagePath in imagePathList:
        count = 0
        image = cv2.imread(imagePath)
        # TODO: 5 random crop image and save
        for x in range(2):
            croppedImage = randomCropImage(image)
            saveImage(croppedImage, imagePath, count)
            count += 1
            saveImage(perturbateColor(croppedImage), imagePath, count)
            count += 1

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
                saveImage(perturbateColor(
                    image_rotated_cropped), imagePath, count)
                count += 1

    return


def divideImageSet(classPath1, classPath2):
    # For directory containing images of each class, combine all images path
    # create custom_training_set.csv with training image path and label
    # create custom_test_set.csv with test image path and label
    class1List = glob.glob(classPath1 + "/*.jpg")
    class2List = glob.glob(classPath2 + "/*.jpg")

    combined_list = []
    for class1_path in class1List:
        combined_list.append([class1_path, 0])

    for class2_path in class2List:
        combined_list.append([class2_path, 1])

    data = [(random.random(), l) for l in combined_list]
    data.sort()

    train_data = data[0:int(len(data) * 0.9)]
    test_data = data[int(len(data) * 0.9):]

    with open('data/custom_train_data.csv', 'w') as train_file:
        for data in train_data:
            line = str(data[1][0]) + " , " + str(data[1][1]) + "\n"
            train_file.write(line)
    with open('data/custom_test_data.csv', 'w') as test_file:
        for data in test_data:
            line = str(data[1][0]) + " , " + str(data[1][1]) + "\n"
            test_file.write(line)

    return


if __name__ == "__main__":
    augmentImages("data/umbrella_caltech")
    augmentImages("data/laptop_caltech")
    divideImageSet("data/umbrella_caltech", "data/laptop_caltech")
