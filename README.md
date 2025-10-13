# 🎯 ING Datathon Churn Prediction ~ Team Güney Kampüs

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-orange.svg)](https://www.kaggle.com/competitions/acquire-valued-shoppers-challenge/data)

## Overview
This project is designed to build an end-to-end machine learning pipeline for churn prediction project, for ING Hubs Türkiye Datathon Kaggle stage.

## Project Structure
```
my-project
├── data          # Datasets and raw data files
├── notebooks     # Jupyter notebooks for analysis and visualizations
├── utils         # Utility scripts and functions
├── logs          # Log files for debugging and monitoring
├── config.yml    # Configuration settings for the project
├── .gitignore    # Files and directories to be ignored by Git
├── Makefile      # Automation tasks for build and management
├── requirements.txt # List of required Python packages
├── setup.py      # Installation script for the package
└── README.md     # Documentation for the project
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone https://github.com/bthnsrts/ing-datathon-team-guney-kampus.git
   ```
2. Navigate to the project directory (make sure you are at the root repo)
   ```
   cd ing-datathon-team-guney-kampus
   ```
3. Create a virtual environment:
   ```
   python -m venv venv
   # Activate the virtual environment
   # On Windows:
   # venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
   - If you see "command not found: kaggle" after installation, install the Kaggle package explicitly:
   ```
   pip install kaggle
   ```

5. Set up Kaggle API credentials:
   - Check if you already have the Kaggle credentials set up:
     ```
     ls -la ~/.kaggle
     ```
   - If you see a file named `kaggle.json` with permissions like `-rw-------`, you already have the credentials set up and can skip to step 6.
   - If not, create a Kaggle account if you don't have one at [kaggle.com](https://www.kaggle.com/)
   - Go to your Kaggle account settings > API section and click "Create New API Token"
   - This will download a `kaggle.json` file containing your API credentials
   - If the directory doesn't exist, create it:
     ```
     mkdir -p ~/.kaggle
     ```
   - Copy your API credentials to the Kaggle directory:
     ```
     cp path/to/downloaded/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json  # Set permissions so only you can read/write the file
     ```

6. Download the competition dataset:
   ```
   make download_kaggle_dataset
   ```
   
   This will:
   - Download the competition data from Kaggle
   - Extract it to the `data/` directory
   - Remove the zip file after extraction
   
   If you prefer to run the commands manually:
   ```
   kaggle competitions download -c ing-hubs-turkiye-datathon
   unzip ing-hubs-turkiye-datathon.zip -d data/
   rm ing-hubs-turkiye-datathon.zip
   ```

7. Notebooks and other scripts use the utils package, in order to enable this;
   ```bash
   pip install -e .
   ```
   This will:
   - Install the utils as a package in development mode
   - Allow you to make changes to the code without reinstalling

## Dataset Explanation

This project uses a banking customer dataset for churn prediction:

- **customer_history.csv**: Monthly transaction history (EFT, credit card usage)
- **customers.csv**: Demographic data (age, gender, work info)
- **reference_data.csv**: Training data with churn labels
- **reference_data_test.csv**: Test customers requiring churn prediction
- **sample_submission.csv**: Submission format template

The goal is to predict customer churn (whether a customer will leave) within 6 months after the reference date. Churn is indicated as 1 (customer left) or 0 (customer stayed).

For detailed column descriptions, see [DATASET_EXPLANATION.md](./DATASET_EXPLANATION.md)

## Usage
- To run the Jupyter notebooks, navigate to the `notebooks` folder and open the desired notebook.
- Choose the Python Interpreter for notebooks from virtual environment created.
- Use the utility functions in the `utils` folder as needed in your analysis.
- Check the `logs` folder for any log files generated during execution.

## Configuration
Adjust the parameters in the `config.yml` file to customize the project settings.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
