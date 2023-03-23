import streamlit as st
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Define the layout of the app
st.set_page_config(page_title="BrainGazer", page_icon=":camera:")

############################################################
def crop_brain_contour(image, plot=False):
    import imutils
    import cv2
    from matplotlib import pyplot as plt

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


def streamlit_load_data(image, image_size):
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
    # image = cv2.imread(filename)
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


st.header("BrainGazer: A Brain Tumor Detection Algorithm")
st.write(
    "Try uploading an MRI Scan and watch how a CNN based algorithm with 91% accuracy detect the possibility of a brain tumor in the uploaded MRI Scan."
)

st.caption(
    "The application will infer the one label out of 2 labels: The model has detected the presence of a brain tumor in the scan., The model has not detected the presence of a brain tumor in the scan."
)

st.caption(
    "Warning: Do not click Submit Scan button before uploading image. It will result in error."
)

with st.sidebar:
    st.header("BrainGazer")
    img = Image.open("./Images/brain_scans.png")
    st.image(img)
    st.subheader("About BrainGazer")
    st.write(
        "BrainGazer is a state-of-the-art deep learning algorithm developed for the detection of brain tumors using medical imaging. It is based on Convolutional Neural Networks (CNNs), which are known for their exceptional performance in image classification tasks. The algorithm is designed to analyze Magnetic Resonance Imaging (MRI) scans and accurately identify the presence of tumors in the brain."
    )

    st.write(
        "The BrainGazer algorithm has been trained on a large dataset of MRI scans, including both healthy and tumor-afflicted brains. This has allowed it to learn complex patterns and features that are indicative of tumors. The algorithm uses a series of convolutional layers to extract meaningful features from the input images and then passes them through fully connected layers to classify the image as either tumor or healthy."
    )

    st.write(
        "BrainGazer has achieved a remarkable accuracy of 91% in detecting brain tumors, making it a highly effective tool for medical professionals in diagnosing brain tumors early. It has the potential to significantly reduce the number of missed diagnoses and provide a more accurate diagnosis of brain tumors, enabling patients to receive the appropriate treatment in a timely manner."
    )

# Comment down the uploaded_file varaible below to accept camera_input image.
# uploaded_file = st.file_uploader(
#     label="Upload your MRI Scan",
#     accept_multiple_files=False,
#     label_visibility="visible",
#     type=["png", "jpeg", "jpg"],
# )

# Uncomment uploaded_file varaible below to accept camera_input image.
uploaded_file = st.camera_input(
    label="Upload your MRI Scan",
    label_visibility="visible",
)


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="BGR")


def detect_tumor_in_the_scan(img_array):
    IMG_WIDTH, IMG_HEIGHT = (240, 240)

    X = streamlit_load_data(img_array, (IMG_WIDTH, IMG_HEIGHT))
    best_model = load_model(
        filepath="./Brain_Tumor_Detection_Models/cnn-parameters-improvement-03-0.91.model"
    )
    y = best_model.predict(X)
    Detection_Result = (
        "The model has not detected the presence of a brain tumor in this MRI scan."
    )
    if y[0][0] >= 0.5:
        Detection_Result = (
            "The model has detected the presence of a brain tumor in this MRI scan."
        )

    return Detection_Result


submit = st.button(label="Submit Scan")
if submit:
    st.subheader("Output")
    classified_label = detect_tumor_in_the_scan(opencv_image)
    with st.spinner(text="This may take a moment..."):
        st.write(classified_label)


footer = """
<div style="text-align: center; font-size: medium; margin-top:50px;">
    If you find BrainGazer useful or interesting, please consider starring it on GitHub.
    <hr>
    <a href="https://github.com/SaiJeevanPuchakayala/BrainGazer" target="_blank">
    <img src="https://img.shields.io/github/stars/SaiJeevanPuchakayala/BrainGazer.svg?style=social" alt="GitHub stars">
  </a>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)