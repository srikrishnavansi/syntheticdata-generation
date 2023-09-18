import streamlit as st
import pandas as pd
import numpy as np
import random
from faker import Faker
from io import BytesIO
from sklearn.neighbors import NearestNeighbors

# Function to calculate DCR (Distance to Closest Record)
def calculate_dcr(data1, data2):
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(data1)
    distances, _ = nn.kneighbors(data2)
    return distances.mean()

# Function to calculate NNDR (Nearest Neighbour Distance Ratio)
def calculate_nndr(data1, data2):
    nn1 = NearestNeighbors(n_neighbors=2)
    nn2 = NearestNeighbors(n_neighbors=2)
    nn1.fit(data1)
    nn2.fit(data2)
    distances1, _ = nn1.kneighbors(data2)
    distances2, _ = nn2.kneighbors(data1)
    return (distances1.mean() / distances2.mean())

# Function to generate synthetic data and return similarity scores
def generate_synthetic_data(real_data):
    # Identify continuous and categorical features
    continuous_features = []
    categorical_features = []

    for column in real_data.columns:
        if real_data[column].dtype in [np.float64, np.int64]:
            continuous_features.append(column)
        else:
            categorical_features.append(column)

    # Generate synthetic data
    num_samples = len(real_data)
    synthetic_data = []

    fake = Faker()

    for _ in range(num_samples):
        synthetic_sample = {}

        # Generate synthetic values for continuous features (within original data range)
        for feature in continuous_features:
            if real_data[feature].dtype == np.float64:
                synthetic_sample[feature] = np.random.uniform(real_data[feature].min(), real_data[feature].max())
            elif real_data[feature].dtype == np.int64:
                synthetic_sample[feature] = random.randint(real_data[feature].min(), real_data[feature].max())

        # Generate synthetic values for categorical features respecting original values
        for feature in categorical_features:
            unique_values = real_data[feature].unique()
            synthetic_sample[feature] = random.choice(list(unique_values))

        synthetic_data.append(synthetic_sample)

    synthetic_data = pd.DataFrame(synthetic_data, columns=real_data.columns)

    # Calculate statistics for continuous features
    mean_real = real_data[continuous_features].mean()
    std_real = real_data[continuous_features].std()

    mean_synthetic = synthetic_data[continuous_features].mean()
    std_synthetic = synthetic_data[continuous_features].std()

    # Calculate the overall similarity score for numerical features
    mean_difference = (mean_synthetic - mean_real).abs().mean()
    std_difference = (std_synthetic - std_real).abs().mean()

    # Normalize the similarity score to a 0-100% scale
    max_possible_difference_mean = (real_data[continuous_features].max() - real_data[continuous_features].min()).mean()
    max_possible_difference_std = (real_data[continuous_features].max() - real_data[continuous_features].min()).std()
    overall_similarity_score_mean = ((max_possible_difference_mean - mean_difference) / max_possible_difference_mean) * 100
    overall_similarity_score_std = ((max_possible_difference_std - std_difference) / max_possible_difference_std) * 100

    # Calculate the average similarity score for mean and std
    overall_similarity_score = (overall_similarity_score_mean + overall_similarity_score_std) / 2

    return synthetic_data, overall_similarity_score

# Function to calculate privacy metrics (IMS, DCR, NNDR) based on numeric columns
def calculate_privacy_metrics(real_data, synthetic_data):
    # Filter out only numeric columns
    numeric_columns = real_data.select_dtypes(include=[np.number]).columns.tolist()

    # Initialize privacy metrics to zero
    training_ims, synthetic_ims, training_dcr, synthetic_dcr, training_nndr, synthetic_nndr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    if len(numeric_columns) > 0:
        # Calculate the number of common records between real and synthetic data
        common_records = len(pd.concat([real_data[numeric_columns], synthetic_data[numeric_columns]]))

        # Calculate privacy metrics
        training_ims = (common_records / len(real_data[numeric_columns])) * 100
        synthetic_ims = (common_records / len(synthetic_data[numeric_columns])) * 100

        if common_records > 0:
            training_dcr = calculate_dcr(real_data[numeric_columns], synthetic_data[numeric_columns])
            synthetic_dcr = calculate_dcr(synthetic_data[numeric_columns], real_data[numeric_columns])

            training_nndr = calculate_nndr(real_data[numeric_columns], synthetic_data[numeric_columns])
            synthetic_nndr = calculate_nndr(synthetic_data[numeric_columns], real_data[numeric_columns])

    return training_ims, synthetic_ims, training_dcr, synthetic_dcr, training_nndr, synthetic_nndr

def normalize(value, min_value, max_value):
    if max_value != min_value:
        return (value - min_value) / (max_value - min_value)
    else:
        return 0.0

def main():
    st.title("Synthetic Data Generator")

    # Upload real data Excel file
    st.header("Upload Real Data")
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xls", "xlsx", "csv"])
    real_data = None
    synthetic_data = None  # Initialize synthetic_data

    if uploaded_file is not None:
        real_data = pd.read_excel(uploaded_file)

        st.header("Real Data Preview")
        st.write(real_data.head())

        similarity_score = 0.0  # Initialize similarity_score
        training_ims, synthetic_ims, training_dcr, synthetic_dcr, training_nndr, synthetic_nndr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        training_samples = 0
        synthetic_samples = 0

        if st.button("Generate Synthetic Data"):
            st.write("Generating synthetic data...")
            synthetic_data, similarity_score = generate_synthetic_data(real_data)

            st.header("Synthetic Data Preview")
            st.write(synthetic_data.head())
            synthetic_data.to_csv("synthetic_data.csv", index=False)
            st.success("Synthetic data generated and saved to CSV.")
            st.subheader("Quality Score")
            st.write(f"Similarity Score: {similarity_score:.2f}%")

            # Calculate privacy metrics
            training_ims, synthetic_ims, training_dcr, synthetic_dcr, training_nndr, synthetic_nndr = calculate_privacy_metrics(real_data, synthetic_data)

            # Display Privacy and Dataset metrics using layout
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Privacy Metrics")
                privacy_metrics_data = {
                    "Metric": ["Training IMS", "Synthetic IMS", "Training DCR", "Synthetic DCR", "Training NNDR", "Synthetic NNDR"],
                    "Value": [normalize(training_ims, 0, training_samples), normalize(synthetic_ims, 0, synthetic_samples), normalize(training_dcr, 0, 1), normalize(synthetic_dcr, 0, 1), training_nndr, synthetic_nndr]
                }
                privacy_metrics_df = pd.DataFrame(privacy_metrics_data)
                st.table(privacy_metrics_df)

            with col2:
                st.subheader("Dataset Metrics")
                dataset_metrics_data = {
                    "Metric": ["No of Training Samples", "No of Synthetic Samples", "Data Columns"],
                    "Value": [len(real_data), len(synthetic_data), len(real_data.columns)]
                }
                dataset_metrics_df = pd.DataFrame(dataset_metrics_data)
                st.table(dataset_metrics_df)

        if real_data is not None:
            training_samples = len(real_data)

        if synthetic_data is not None:
            synthetic_samples = len(synthetic_data)

        if st.button("Download Synthetic Data"):
            with open("synthetic_data.csv", "rb") as file:
                st.download_button("Download Synthetic Data CSV", file.read(), file_name="synthetic_data.csv")

if __name__ == "__main__":
    main()

