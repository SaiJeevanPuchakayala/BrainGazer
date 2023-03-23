
# BrainGazer

## A CNN based algorithm with 91% accuracy for brain tumor detection.

![MRI Scan Evaluation of Brain Tumor Detection](./Images/brain_scans.png)

BrainGazer is a state-of-the-art deep learning algorithm developed for the detection of brain tumors using medical imaging. It is based on Convolutional Neural Networks (CNNs), which are known for their exceptional performance in image classification tasks. The algorithm is designed to analyze Magnetic Resonance Imaging (MRI) scans and accurately identify the presence of tumors in the brain.

The BrainGazer algorithm has been trained on a large dataset of MRI scans, including both healthy and tumor-afflicted brains. This has allowed it to learn complex patterns and features that are indicative of tumors. The algorithm uses a series of convolutional layers to extract meaningful features from the input images and then passes them through fully connected layers to classify the image as either tumor or healthy.

BrainGazer has achieved a remarkable accuracy of 91% in detecting brain tumors, making it a highly effective tool for medical professionals in diagnosing brain tumors early. It has the potential to significantly reduce the number of missed diagnoses and provide a more accurate diagnosis of brain tumors, enabling patients to receive the appropriate treatment in a timely manner.

## ⭐  About the Brain MRI Images Dataset<br>
The dataset contains 2 folders: yes and no which contains 253 Brain MRI Images. The folder yes contains 155 Brain MRI Images that are tumorous and the folder no contains 98 Brain MRI Images that are non-tumorous. You can find it [here](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection).

## ⭐  To run these scripts, you need the following installed:

1. Python 3
2. The python libraries listed in requirements.txt
    * Try running "pip3 install -r requirements.txt"

### Step 1: Clone this repository
Run:
```bash
git clone https://github.com/SaiJeevanPuchakayala/BrainGazer
```

### Step 2: Navigate to the BrainGazer directory
Run:
```bash
cd BrainGazer
```
### Step 3: Install the python libraries
Run:
```bash
pip install -r requirements.txt
```
### Step 4: Run the streamlitApp.py file
Run:
```bash
streamlit run streamlitApp.py
```
## ⭐ Lessons Learned

* Data quality is critical: The quality of the data used to train the CNN model can greatly affect its accuracy. Ensuring that the data is labeled correctly and covers a diverse range of cases can improve the model's performance.

* Model optimization is key: Hyperparameter tuning, regularization, and model architecture selection can all play a significant role in improving the model's accuracy. It is important to experiment with different configurations to find the best combination for a given problem.

* Collaboration is essential: Building a CNN-based algorithm for brain tumor detection requires a team effort, involving experts in different fields, such as radiology, computer science, and machine learning. Collaboration helps ensure that the model is clinically relevant and has the potential to make a positive impact on patient outcomes.

* Ethics and transparency are critical: As with any algorithm used in healthcare, it is important to consider ethical and transparency issues. This includes ensuring that the model is fair and unbiased and that its limitations and potential errors are clearly communicated to stakeholders.

## ⭐  Challenges faced:

* The collection and preparation of a large dataset. 
* Dealing with class imbalance.
* Selecting appropriate architectures.
* Fine-tuning hyperparameters.
* Optimizing the algorithm's performance.

## ⭐ Streamlit Deployment Configurations:
```
[theme]
base="dark"

[browser]
gatherUsageStats = false
```

## ⭐ References:
1. https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app
2. https://streamlit-cloud-example-apps-streamlit-app-sw3u0r.streamlit.app/?hsCtaTracking=28f10086-a3a5-4ea8-9403-f3d52bf26184|22470002-acb1-4d93-8286-00ee4f8a46fb
3. https://docs.streamlit.io/library/advanced-features/configuration

## ⭐ Note:
### **If you find my GitHub repository useful, why not give it a star? It's like giving a little virtual high-five that makes my day!**