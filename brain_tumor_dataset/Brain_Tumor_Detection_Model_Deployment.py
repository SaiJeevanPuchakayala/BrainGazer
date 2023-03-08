from tensorflow.keras.models import load_model
import cv2
import imutils
import numpy as np


############################################################
def crop_brain_contour(image, plot=False):
    # import imutils
    # import cv2
    # from matplotlib import pyplot as plt

    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1] : extBot[1], extLeft[0] : extRight[0]]

    #     if plot:
    #         plt.figure()

    #         plt.subplot(1, 2, 1)
    #         plt.imshow(image)

    #         plt.tick_params(axis='both', which='both',
    #                         top=False, bottom=False, left=False, right=False,
    #                         labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    #         plt.title('Original Image')

    #         plt.subplot(1, 2, 2)
    #         plt.imshow(new_image)

    #         plt.tick_params(axis='both', which='both',
    #                         top=False, bottom=False, left=False, right=False,
    #                         labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    #         plt.title('Cropped Image')

    #         plt.show()

    return new_image


############################################################


def load_data(filename, image_size):
    """
    Read images, resize and normalize them.
    Arguments:
        dir_list: list of strings representing file directories.
    Returns:
        X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
        y: A numpy array with shape = (#_examples, 1)
    """

    # load all images in a directory
    X = []
    image_width, image_height = image_size

    # load the image
    image = cv2.imread(filename)
    # crop the brain and ignore the unnecessary rest part of the image
    image = crop_brain_contour(image, plot=False)
    # resize image
    image = cv2.resize(
        image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC
    )
    # normalize values
    image = image / 255.0
    # convert image to numpy array and append it to X
    X.append(image)

    X = np.array(X)

    return X


############################################################

IMG_WIDTH, IMG_HEIGHT = (240, 240)

X = load_data("./Augmented_Dataset/no/aug_1 no._0_1034.jpg", (IMG_WIDTH, IMG_HEIGHT))
best_model = load_model(
    filepath="./Brain_Tumor_Detection_Models/cnn-parameters-improvement-03-0.91.model"
)
y = best_model.predict(X)
Detection_Result = "No"
if y[0][0] >= 0.5:
    Detection_Result = "Yes"
print(Detection_Result)
