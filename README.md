## 🚀 Malaria Cell Detection using Deep Learning

A computer vision system that automatically detects malaria parasites in blood cell images using Convolutional Neural Networks (CNN). This project helps medical professionals quickly identify parasitized cells from microscopic blood smear images.

## 📷 Project Overview

This system analyzes blood cell images and classifies them as either "Parasitized" (infected with malaria) or "Uninfected" using a custom-trained CNN model. The project includes both the training pipeline and a web application for real-time predictions.

## 🧬 Dataset Information

- **Source**: Kaggle Cell Images for Detecting Malaria dataset
- **Total Images Used**: 10,000 images (5,000 per class)
- **Classes**: Parasitized, Uninfected
- **Training Split**: 80% training, 20% validation
- **Image Size**: 128x128 pixels

## 🏗️ Model Architecture

The CNN model consists of:
- 3 Convolutional blocks with BatchNormalization
- Conv2D layers with filters: 32, 64, 128
- MaxPooling2D layers for downsampling
- Flatten layer followed by Dense layers
- Dropout (0.5) for regularization
- Sigmoid activation for binary classification

## 📊 Model Performance

- **Training Epochs**: 10
- **Final Validation Accuracy**: 93.9%
- **Final Validation Loss**: 0.1875
- **Batch Size**: 64

## 📁 Project Structure

'''
malaria-detection/
├── app.py                          # Streamlit web application
├── malariya-detection.ipynb        # Training notebook (generates model)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
'''

**Note**: The 'malaria_cnn.h5' model file is generated when you run the training notebook.

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook or access to Kaggle
- Internet connection for dataset download

### Quick Start
1. Clone this repository
2. Install dependencies: 'pip install -r requirements.txt'
3. Open and run 'malariya-detection.ipynb' to train the model
4. Once training completes, update the MODEL_PATH in 'app.py'
5. Run the web app: 'streamlit run app.py'

## 🖥️ Usage

### Step 1: Train the Model

1. Open 'malariya-detection.ipynb' in Jupyter Notebook or upload to Kaggle
2. Run all cells to:
   - Download and process the malaria dataset
   - Train the CNN model for 10 epochs
   - Generate 'malaria_cnn.h5' model file
3. Download the generated model file to your local project directory

### Step 2: Run the Web Application

1. Update the MODEL_PATH in 'app.py' to point to your model location
2. Run the Streamlit app:
'''bash
streamlit run app.py
'''
3. Upload a blood smear image (JPG, JPEG, PNG)
4. View the prediction and confidence score

## 🔧 Data Preprocessing

The training pipeline includes:
- **Data Sampling**: Random selection of 5,000 images per class
- **Data Augmentation**: Rotation, shifts, shear, zoom, horizontal flip
- **Normalization**: Pixel values scaled to [0,1]
- **Image Resizing**: All images resized to 128x128 pixels

## 🎯 Model Training Details

**Callbacks Used**:
- ModelCheckpoint: Save best model based on validation accuracy
- ReduceLROnPlateau: Reduce learning rate when validation loss plateaus
- EarlyStopping: Stop training if no improvement for 4 epochs

**Optimizer**: Adam
**Loss Function**: Binary crossentropy
**Metrics**: Accuracy

## 🌐 Web Application Features

- Simple drag-and-drop image upload interface
- Real-time image preprocessing and prediction
- Visual display of uploaded image
- Clear prediction results with confidence scores
- Support for common image formats (JPG, JPEG, PNG)

## ⚙️ Technical Requirements

- Python 3.7+
- TensorFlow/Keras for deep learning
- Streamlit for web interface
- PIL for image processing
- NumPy for numerical operations

## ⚕️ Medical Disclaimer

This tool is intended for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult qualified medical professionals for accurate malaria diagnosis and treatment.

## File Descriptions

**app.py**: Streamlit web application that loads the trained model and provides a user interface for uploading images and getting predictions.

**malariya-detection.ipynb**: Complete training pipeline including data handling, model building, training, and evaluation.

## Model Confidence Interpretation

- Confidence > 80%: Classified as "Uninfected"
- Confidence ≤ 80%: Classified as "Parasitized"

Critical Design Choice: The 80% threshold is intentionally conservative to minimize false negatives (missing malaria cases). In medical diagnosis, a false negative (predicting "Uninfected" when the patient actually has malaria) can be
life-threatening as it delays critical treatment. By using a lower threshold, the model errs on the side of caution, potentially flagging more cases for further examination rather than missing infected patients.
The model outputs a probability score between 0 and 1, where values closer to 1 indicate higher confidence in the "Uninfected" class.

## 📷 Images
<img width="1179" height="1052" alt="image" src="https://github.com/user-attachments/assets/90d30d5b-ef78-461a-97de-525e6c44c515" />
<img width="1180" height="1097" alt="image" src="https://github.com/user-attachments/assets/38ffac28-effb-4fe9-9f04-ef6481838e40" />
<img width="1176" height="1114" alt="image" src="https://github.com/user-attachments/assets/b701ce42-fc7c-4f52-9dbd-dcaa98522726" />



## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Dataset: Cell Images for Detecting Malaria (Kaggle)
- TensorFlow/Keras for deep learning framework
- Streamlit for web application framework
