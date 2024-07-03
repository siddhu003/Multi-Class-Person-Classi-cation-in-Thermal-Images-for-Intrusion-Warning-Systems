
# Multi Class Person Classification in Thermal Images for Intrusion Warning Systems

- Developing a multi-class person classiﬁcation system using CNNs, improving intrusion detection in thermal images.

- Proposing a segmentation-driven classiﬁcation method, enhancing thermal image analysis accuracy.


## Dataset

- This is the dataset which I used - PDIWS: Thermal Imaging Dataset for Person Detection in Intrusion Warning Systems
- The PDIWS dataset is available here - https://ieee-dataport.org/documents/pdiws-thermal-imaging-dataset-person-detection-intrusion-warning-systems

#### Download dataset and move it to the folder Dataset.

#### Data description

Each image in the dataset is compounded by an object image and a background using the Poison image editing method (see ./PIE). The dataset consists of two subsets train and test:

- train: 2,000 images under .JPG format, each image contains only one object.
- test: 500 images under .JPG format, each image contains only one object.


## Dataset Splitting

Both the training and testing images were initially stored in single folders named train and test respectively. However, for classification purposes, these images need to be organized into separate folders according to their respective classes.

### Classes

The dataset comprises the following 5 classes: 

    1. Intruder Creeping
    2. Intruder Crawling
    3. Pedestrian Stooping
    4. Intruder Climbing
    5. Intruder Other

To facilitate the classification process, the dataset needs to be reorganized. The following steps were taken to achieve this:

- Download the dataset: The PDIWS dataset can be downloaded from here.
- Move to Data Folder: Place the downloaded dataset in the Data directory.
- Run Splitting Scripts: Execute the provided Python scripts to organize the images into their respective class folders.

### Instructions to Run Splitting Scripts

1 Ensure the dataset is in the Data folder:
```bash
Data/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── test/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

2 Run the 'split_train.py' script:
```bash
python split_train.py
```

3 Run the 'split_test.py' script:
```bash
python split_test.py
```

After running these scripts, the dataset directory structure will be organized as follows:

```bash
Data/
├── train/
│   ├── Intruder_Creeping/
│   ├── Intruder_Crawling/
│   ├── Pedestrian_Stooping/
│   ├── Intruder_Climbing/
│   ├── Intruder_Other/
└── test/
    ├── Intruder_Creeping/
    ├── Intruder_Crawling/
    ├── Pedestrian_Stooping/
    ├── Intruder_Climbing/
    ├── Intruder_Other/
```

This organization allows for effective training and evaluation of the classification model based on the defined classes.
## Segmentation Process

We used a segmentation-driven method to enhance the classification of individuals in thermal images:

1 Labeling:
- 430 thermal images were annotated using CVAT.ai, focusing on five classes: Intruder Creeping, Intruder Crawling, Pedestrian Stooping, Intruder Climbing, and Intruder Other.
- 370 images were used for training, and 60 for validation.

2 Training:
- The annotated images were converted into .txt files and used to train the YOLOv8 nano model for segmentation.

3 Application:
- The trained YOLOv8 nano model segmented regions of interest in the PDIWS dataset, highlighting relevant features for classification.

Annotated images are available here.

### Running the Segmentation
To segment the images using the provided scripts, follow these steps:

1 Ensure the dataset and annotated images are in the correct directories:
- Place the annotated images in the directory specified in config.yaml.
- Ensure the PDIWS dataset is in the Data folder.
2 Run the Segmentation Script:
- Run the 'prediction.py' script to segment the images.
3 Running the Script:
- Place the script in your working directory and run it using Python:

```bash
python prediction.py
```
4 Output Directory:
- The segmented images will be saved in the specified output directory with subfolders for each class.
## CNN Model

### Training and Testing the Model
Use the provided Jupyter notebook to train and test the CNN model.

1 Training the Model:
- Open the cnn-model.ipynb notebook.
- Follow the instructions to train the CNN model on the segmented images.
- Ensure that the segmented images are organized into the appropriate directories as described above.

2 Testing the Model:
- Use the same notebook to test the trained model on the test dataset.
- Evaluate the model’s performance and adjust parameters as necessary.
## Results

### Training and Loss Curves
During the training process, we tracked the model's performance through training and loss curves.
- Training Curves: These curves show the model's accuracy over each epoch, indicating how well the model learns from the training data.

![CNN-Accuracy](https://github.com/siddhu003/Multi-Class-Person-Classi-cation-in-Thermal-Images-for-Intrusion-Warning-Systems/assets/113658076/c88eeb4c-5443-4fbf-94a2-19d2f6d55c6f)

- Loss Curves: These curves depict the loss value over each epoch, showing how well the model's predictions match the actual labels.

![CNN-Loss](https://github.com/siddhu003/Multi-Class-Person-Classi-cation-in-Thermal-Images-for-Intrusion-Warning-Systems/assets/113658076/896353d7-1f81-425c-be62-d4eb3acce930)

### Confusion Matrix

- The confusion matrix was used to evaluate the final performance of the model on the test dataset
- Confusion Matrix: This matrix shows the true positive, true negative, false positive, and false negative rates for each class, providing a comprehensive view of the model's classification performance.
  
![Confusion_matrix](https://github.com/siddhu003/Multi-Class-Person-Classi-cation-in-Thermal-Images-for-Intrusion-Warning-Systems/assets/113658076/a5dfe15e-46b2-4201-86a0-575fc570cf53)

  
## Prediction

The prediction process involves using the YOLO model to segment the thermal images and then using the trained CNN model to classify the segmented images.

### Running the Prediction

1 Ensure the model is trained:

- Ensure that the YOLO and CNN models are trained and the weights are saved.

2 Run the Prediction Script:

- Open the prediction.ipynb notebook.
- Run each of the cells in the notebook to make a prediction.
## Conclusion

The goal of this project was to enhance the classification of individuals in thermal images for intrusion warning systems using a segmentation-driven classification method. This approach significantly improves over traditional techniques, advancing thermal image classification for security purposes.

The proposed model, leveraging a combination of YOLO for segmentation and a CNN for classification, achieved an accuracy of 80.25%. This performance is substantially better than most pre-trained models when applied to this specific dataset. The results demonstrate the effectiveness of integrating segmentation with classification, providing a robust solution for real-world security applications.
