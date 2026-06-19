# 🚖 Taxi Driver Trajectory Classification using Deep Neural Networks (DNN)

---

## 🛠️ Tech Stack & Skills Depth

Here is the tech stack and depth of Machine Learning (ML) & Deep Learning (DL) engineering demonstrated in this project:

### 🚀 Core Technologies
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

### 📊 Deep Learning & ML Skill Depth Showcase
* **🧠 Deep Neural Architectures (DNN)**: Design and fine-tuning of multi-layered feedforward networks (up to 1024-wide Relu hidden layers) built with `Keras Sequential API` and modern `Input()` layers.
* **⚡ Optimization & Regularization**: Expertise in preventing model overfitting and vanishing/exploding gradients using **Batch Normalization**, **Dropout Regularization (0.3 - 0.4)**, and **Early Stopping** callbacks.
* **🌍 Geospatial Feature Engineering**: Complex coordinate data manipulation using spatial calculations (Great Circle geodesic distance calculation using `geopy` distance APIs) and temporal elapsed calculation (`time_diff`).
* **📈 Model Validation & Metrics**: Robust split-validation techniques (`train_test_split`) coupled with detailed model classification metrics (`classification_report`, F1-Score, precision-recall curve analysis, and visual `ConfusionMatrixDisplay`).

---

## 🚀 SEO Keywords & Search Terms
*Trajectory Mining, Taxi Driver Classification, GPS Data Analysis, Deep Learning Spatial-Temporal, Geospatial Feature Engineering, Driver Profiling, TensorFlow Keras Trajectory Prediction, Geopy Distance Classification, Deep Neural Network Trajectory, Keras Deep Learning classification, Taxi GPS stream parsing.*

---

## 📌 Project Overview
Urban mobility data contains rich signatures of individual driving behavior. This project aims to classify unique taxi drivers (identified by their license plate IDs) by profiling their spatial-temporal footprints. By preprocessing raw GPS streams containing coordinates, timestamps, and passenger status, the model learns distinct route choices, schedules, and spatial patterns characteristic of individual drivers.

### 🌟 Key Features
- 🧹 **Geospatial Processing**: Cleans and filters out coordinates outside valid boundary ranges.
- 📐 **Advanced Feature Engineering**: Leverages time features (hour, day of the week, month) and computes spatial metrics such as geodesic distance (using the Great Circle formula) and trajectory elapsed time.
- 🧠 **Deep Learning Classifier**: Employs a multi-layer Deep Neural Network (DNN) in Keras/TensorFlow optimized with Batch Normalization, Dropout regularization, Adam optimizer, and Early Stopping.
- 📊 **Rigorous Evaluation**: Validates predictions using Precision, Recall, F1-Score, Accuracy, and visual Confusion Matrices.

---

## 📊 Dataset & Feature Engineering

### Raw Data Structure
The model processes high-frequency GPS streams with the following baseline columns:
- `plate`: The unique identifier/label for the taxi driver (Target Variable).
- `longitude` / `latitude`: GPS coordinates.
- `time`: Timestamp of coordinate collection.
- `status`: Binary occupancy indicator (e.g., 0 = Empty, 1 = Passenger Onboard).

### Preprocessing & Data Cleaning
1. **Anomaly Filtering**: Removes coordinate outliers (ensures longitude is between -180 and 180, and latitude is between -90 and 90).
2. **Deduplication**: Drops duplicate trajectory pings based on space and time coordinates.
3. **Missing Value Management**: Drops rows with incomplete spatial-temporal records.

### Feature Extraction
- **Temporal Features**: Extracted `hour` of the day, `day_of_week`, and `month` from the raw timestamps.
- **Geospatial Metrics**: Geodesic distance calculation via Great Circle distance to compute spatial coverage.
- **Temporal Differences**: Calculated the elapsed time (`time_diff`) between pings to infer driver speed and congestion levels.
- **Scaling**: All input features are normalized using `StandardScaler` to ensure uniform gradient propagation during backpropagation.

---

## 🧠 Model Architecture (Keras Deep Learning)

The final architecture selected for classification is a deep fully-connected network built to prevent overfitting while handling massive trajectory matrices:

```python
model = Sequential([
    Input(shape=(input_dim,)),
    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(num_classes, activation='softmax')
])
```

### Key Training Strategy
- **Optimizer**: Adam (Adaptive Moment Estimation)
- **Loss Function**: Categorical Crossentropy (Multiclass target)
- **Regularization**: High Dropout rates (up to 0.4) and Batch Normalization layers for fast convergence.
- **Early Stopping**: Monitored validation accuracy with a patience of 15 epochs to automatically stop training and restore the best weights.

---

## 📈 Evaluation & Results
The model is evaluated using a held-out test dataset to generate key classification metrics:
- **Accuracy**: Tracks overall correct driver classification rate.
- **Precision, Recall, F1-Score**: Measures per-driver classification quality to account for class imbalances in routes taken.
- **Confusion Matrix**: Shows inter-driver confusion patterns to identify drivers with highly overlapping route routes.

---

## ⚙️ How to Run
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook "Ml project 3.ipynb"
   ```
2. Set the path to the raw driver tracking data directory in the first cell:
   ```python
   data_folder = "/path/to/extracted/CSV_files/"
   ```
3. Run all cells to process the data, train the DNN model, and output the evaluation metrics.
4. The trained model will be exported automatically as:
   - `taxi_driver_classification_model2.keras`

---

## 🔮 Future Enhancements
- 🔄 **RNN / LSTM integration**: Incorporate sequence models (like LSTM or GRU) to classify drivers based on *temporal route sequences* rather than static feature vectors.
- 🗺️ **Spatial Embedding**: Apply geospatial embeddings (e.g., Uber's H3 Hexagonal spatial index) to capture neighborhood-specific route preferences.
- 💻 **Web Dashboard**: Create a lightweight Streamlit interface to visualize trajectory paths alongside driver classifications in real time.
