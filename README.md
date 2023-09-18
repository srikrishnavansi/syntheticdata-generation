Synthetic Data Generator
The Synthetic Data Generator is a Python application built with Streamlit that allows you to generate synthetic data based on a provided real dataset. It also calculates similarity scores and privacy metrics to evaluate the quality of the generated synthetic data.

Table of Contents
Introduction
Features
Installation
Usage
Dependencies
License
Features
Upload a real dataset in Excel format.
Preview the real dataset.
Generate synthetic data that mimics the statistical properties of the real dataset.
Calculate and display a similarity score between the real and synthetic datasets.
Calculate privacy metrics such as IMS (Intersection of Marginal Scores), DCR (Distance to Closest Record), and NNDR (Nearest Neighbour Distance Ratio).
Display privacy and dataset metrics in a user-friendly interface.
Download the generated synthetic dataset in CSV format.
Installation
Clone the repository to your local machine:

bash
Copy code
git clone https://github.com/srikrishvansi/syntheticdata-generation.git
cd syntheticdata-generation
Create a virtual environment (optional but recommended):

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Run the application:

bash
Copy code
streamlit run Hello.py
Upload a real dataset in Excel format using the provided file uploader.

Preview the real dataset to ensure it's loaded correctly.

Click the "Generate Synthetic Data" button to generate synthetic data based on the real dataset. The application will display the synthetic dataset and calculate a similarity score between the real and synthetic datasets.

View privacy metrics and dataset metrics in the user interface. Privacy metrics include Training IMS, Synthetic IMS, Training DCR, Synthetic DCR, Training NNDR, and Synthetic NNDR. Dataset metrics include the number of training samples, the number of synthetic samples, and the number of data columns.

Optionally, download the generated synthetic dataset in CSV format using the "Download Synthetic Data" button.

Dependencies
The Synthetic Data Generator relies on the following Python libraries and packages:

Streamlit
Pandas
NumPy
Random
Faker
Scikit-Learn
All dependencies are listed in the requirements.txt file and can be installed using pip.

License
This Synthetic Data Generator is open-source software released under the MIT License. You are free to use, modify, and distribute it as per the terms of the license.

