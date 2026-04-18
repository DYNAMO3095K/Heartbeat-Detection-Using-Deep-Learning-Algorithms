Heartbeat Detection using AI/ML
​An end-to-end machine learning pipeline designed to classify Electrocardiogram (ECG) time-series signals into normal and abnormal categories. This project addresses the challenge of automated cardiac anomaly detection by comparing traditional machine learning algorithms with deep learning architectures.
​Dataset
​The project utilizes the PTB Diagnostic ECG Database, consisting of:

​ptbdb_normal.csv
​ptbdb_abnormal.csv

​The raw data presents a severe class imbalance, which is handled dynamically within the pipeline using a custom resampling algorithm to downsample the majority class and upsample minority classes prior to model training.
​Technologies Used

​Programming Language: Python
​Deep Learning: TensorFlow, Keras
​Machine Learning: Scikit-learn (K-Nearest Neighbors, Logistic Regression)
​Hyperparameter Tuning: Keras Tuner, GridSearchCV
​Data Processing & Visualization: Pandas, NumPy, Matplotlib, Seaborn
​Model Serialization: Joblib

​Features

​Time-Series Signal Processing: Feature amplitude scaling using MinMaxScaler and variance standardization via StandardScaler.
​Deep Learning Architecture: Custom 1D Convolutional Neural Network (CNN) specifically tailored for extracting spatial and temporal features from non-image sequential data.
​Automated Optimization: Dynamic hyperparameter tuning to prevent overfitting and maximize accuracy (RandomSearch for the CNN, GridSearch for traditional ML).
​Advanced Evaluation Metrics: Performance validation utilizing multi-class Receiver Operating Characteristic (ROC) curves, AUC calculations, and Precision-Recall (PR) curves.
​Deployment Ready: Automated serialization pipeline that bundles pre-processing scalers alongside predictive models (.pkl and .keras formats) for stable batch inference.

​How to Run
​1. Install Dependencies
​Ensure you have Python installed, then install the required libraries:
2. Prepare the Data
​Ensure the dataset files (ptbdb_normal.csv and ptbdb_abnormal.csv) are located in the correct working directory or update the file paths in the main script.
​3. Execute the Training Pipeline
​Run the main Python script or Jupyter Notebook. The script will:

​Ingest and clean the data.
​Balance the target classes.
​Train the baseline and tuned models.
​Output evaluation metrics and graphs.
​Save the serialized model artifacts into an /artifacts directory.

​4. Run Inference
​Use the included inference logic to automatically load the latest serialized scaler and model bundle from the /artifacts directory to test predictions on new ECG signal batches.
