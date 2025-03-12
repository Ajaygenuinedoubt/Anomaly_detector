import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import IsolationForest, BaggingRegressor
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import load_wine, load_iris, load_breast_cancer, load_diabetes
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import plotly.express as px

# Function to load prebuilt datasets
def load_data(dataset_name):
    if dataset_name == 'Iris':
        data = load_iris(as_frame=True)
    elif dataset_name == 'Wine':
        data = load_wine(as_frame=True)
    elif dataset_name == 'Breast Cancer':
        data = load_breast_cancer(as_frame=True)
    elif dataset_name == 'Diabetes':
        data = load_diabetes(as_frame=True)
    
    df = pd.DataFrame(data.data, columns=data.feature_names)
    return df

# Function to handle non-numeric columns
def preprocess_data(df):
    # Check if there are any non-numeric columns
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    
    if len(non_numeric_columns) > 0:
        st.write(f"Non-numeric columns found: {list(non_numeric_columns)}")
        # Option 1: Drop non-numeric columns (if not important)
        df = df.drop(non_numeric_columns, axis=1)
        st.write(f"Dropped non-numeric columns: {list(non_numeric_columns)}")
        
        # Option 2 (if you need categorical data): Encode categorical columns
        # Uncomment this if you want to use encoding instead of dropping
        # for col in non_numeric_columns:
        #     le = LabelEncoder()
        #     df[col] = le.fit_transform(df[col].astype(str))
    
    return df

# Function to apply anomaly detection classifiers
def apply_anomaly_detection(classifier_name, X):
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Choose classifier
    if classifier_name == 'Isolation Forest':
        clf = IsolationForest(contamination=0.05, random_state=42)
    elif classifier_name == 'One-Class SVM':
        clf = OneClassSVM(kernel='rbf', nu=0.05)
    elif classifier_name == 'Local Outlier Factor':
        clf = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    elif classifier_name == 'Bagging (Ensemble)':
        base_clf = IsolationForest(contamination=0.05, random_state=42)
        clf = BaggingRegressor(base_estimator=base_clf, n_estimators=10, random_state=42)

    # Fit the model and predict anomalies
    if classifier_name == 'Local Outlier Factor':
        y_pred = clf.fit_predict(X_scaled)
    else:
        clf.fit(X_scaled)
        y_pred = clf.predict(X_scaled)

    # Convert prediction to 0 for inliers and 1 for outliers
    anomalies = np.where(y_pred == -1, 1, 0)

    return anomalies

# Streamlit UI layout
st.title("Anomaly Detection Dashboard By :- Ajay Kumar Jha")
st.write("""
### Choose dataset and anomaly detection algorithm:
Detect anomalies in the dataset using various classifiers.
""")

# Sidebar for dataset and classifier selection
dataset_option = st.sidebar.selectbox("Select Dataset Source", ("Preloaded Dataset", "Upload CSV"))

if dataset_option == "Preloaded Dataset":
    dataset_name = st.sidebar.selectbox("Select Preloaded Dataset", ("Iris", "Wine", "Breast Cancer", "Diabetes"))
    df = load_data(dataset_name)
else:
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV file)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file to continue.")
        st.stop()

# Preprocess the dataset (drop non-numeric columns)
df = preprocess_data(df)

classifier_name = st.sidebar.selectbox("Select Classifier", ("Isolation Forest", "One-Class SVM", "Local Outlier Factor", "Bagging (Ensemble)"))

st.write(f"### Dataset Overview:")
st.write(df.head())

# Apply anomaly detection
anomalies = apply_anomaly_detection(classifier_name, df)

# PCA for 2D visualization
pca = PCA(2)
X_projected = pca.fit_transform(df)

# Convert to DataFrame for Plotly
df_projected = pd.DataFrame(X_projected, columns=["PC1", "PC2"])
df_projected['Anomaly'] = anomalies

# Label for hover display
df_projected['Anomaly Label'] = df_projected['Anomaly'].apply(lambda x: 'Outlier' if x == 1 else 'Inlier')

# Plotly interactive plot for Anomaly Detection Visualization (PCA)
st.write("### Anomaly Detection Visualization (PCA) with Hover Information")
fig = px.scatter(
    df_projected, x="PC1", y="PC2", 
    color="Anomaly", 
    color_continuous_scale=['blue', 'red'],
    hover_data=["Anomaly Label"],  # Show whether it's an outlier or not on hover
    labels={"Anomaly": "Anomaly (1 = Outlier, 0 = Inlier)"}
)
fig.update_layout(title="Anomaly Detection with PCA", legend_title="Anomaly Status")
st.plotly_chart(fig)

# Display detected anomalies count
anomaly_count = sum(anomalies)
st.write(f"Number of anomalies detected: {anomaly_count}")

# Bar chart showing the distribution of anomalies
st.write("### Distribution of Anomalies:")
fig_bar = px.histogram(df_projected, x='Anomaly', nbins=2, labels={"Anomaly": "Anomaly Status"})
fig_bar.update_layout(xaxis_title="Anomaly (1 = Outlier, 0 = Inlier)", yaxis_title="Count")
st.plotly_chart(fig_bar)
