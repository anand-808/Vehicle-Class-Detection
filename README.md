# Vehicle-Class-Detection

---

# Vehicle Classification Project

This project aims to classify images of vehicles into different categories using deep learning models. The project consists of several components, each responsible for a specific task. The models used in this project include VGG16, InceptionV3, and ResNet50, with one variant of InceptionV3 utilizing CUDA for accelerated training.

## Project Components

1. **01_data_preparation**
   - This component involves preparing the dataset by splitting the images into training and testing sets. The resulting splits are stored in CSV files.
   - **Input**: Directory of images
   - **Output**: `train_df.csv` and `test_df.csv`
   - **Notebook**: `01_data_preparation.ipynb`

2. **02_vgg16_training**
   - This component involves training a VGG16 model on the prepared dataset.
   - **Model Path**: `E:/Vehicle/model/vgg16_model.h5`
   - **Notebook**: `02_vgg16_training.ipynb`

3. **03_inceptionv3_training**
   - This component involves training an InceptionV3 model on the prepared dataset.
   - **Model Path**: `E:/Vehicle/model/inceptionv3_model.h5`
   - **Notebook**: `03_inceptionv3_training.ipynb`

4. **04_inceptionv3_cuda_training**
   - This component involves training an InceptionV3 model using CUDA for faster processing.
   - **Model Path**: `E:/Vehicle/Cuda/models/inceptionv3_cuda_model.pth`
   - **Notebook**: `04_inceptionv3_cuda_training.ipynb`

5. **05_resnet50_training**
   - This component involves training a ResNet50 model on the prepared dataset.
   - **Model Path**: `E:/Vehicle/model/resnet50_model.pth`
   - **Notebook**: `05_resnet50_training.ipynb`

6. **06_compare_and_prediction**
   - This component involves comparing the predictions of all trained models and displaying the results along with the confusion matrices.
   - **Notebook**: `06_compare_and_prediction.ipynb`

## Getting Started

### Prerequisites

- Python 3.7 or later
- TensorFlow 2.x
- PyTorch 1.8 or later
- OpenCV
- Pandas
- Scikit-learn
- Seaborn
- Matplotlib
- Jupyter Notebook

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/vehicle-classification.git
   cd vehicle-classification
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Data Preparation**

   Run the data preparation notebook to split the images into training and testing sets:
   ```bash
   jupyter notebook 01_data_preparation.ipynb
   ```

2. **Train Models**

   - Train the VGG16 model:
     ```bash
     jupyter notebook 02_vgg16_training.ipynb
     ```

   - Train the InceptionV3 model:
     ```bash
     jupyter notebook 03_inceptionv3_training.ipynb
     ```

   - Train the InceptionV3 CUDA model:
     ```bash
     jupyter notebook 04_inceptionv3_cuda_training.ipynb
     ```

   - Train the ResNet50 model:
     ```bash
     jupyter notebook 05_resnet50_training.ipynb
     ```

3. **Compare and Predict**

   Compare the models' predictions and display the results:
   ```bash
   jupyter notebook 06_compare_and_prediction.ipynb
   ```

## Results

The `06_compare_and_prediction.ipynb` notebook will output the predictions from all models for a given sample image, along with confusion matrices for each model. The predicted labels and accuracy scores will be displayed, and confusion matrices will be visualized using Seaborn.


---
