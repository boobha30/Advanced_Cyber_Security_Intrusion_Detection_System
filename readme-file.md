# Advanced Intrusion Detection System

This project implements an advanced network intrusion detection system using deep learning techniques with GAN enhancement. The application provides both manual and automatic detection modes with explainable AI features.

## System Requirements

- **Python Version**: 3.10.0 (Strict requirement)
- **IDE**: Visual Studio Code (recommended)
- **Operating System**: Windows/Linux/MacOS

## Installation Guide

### Step 1: Prepare your environment

If you have a different Python version installed, you need to uninstall it first:

#### Windows:
1. Go to Control Panel > Programs > Programs and Features
2. Find Python in the list and click Uninstall
3. Download and install Python 3.10.0 
4. Make sure to check "Add Python to PATH" during installation

#### Linux/MacOS:
```bash
# Check existing Python versions
which python
which python3

# Install Python 3.10.0 using pyenv or your package manager
```

### Step 2: Download and extract the project

1. Download the project ZIP file
2. Extract to a location of your choice
3. Open the extracted folder in VS Code

### Step 3: Set up virtual environment

Creating a virtual environment is recommended to avoid package conflicts:

```bash
# Navigate to the project directory
cd path/to/project

# Create virtual environment
python -m venv venv

# Activate virtual environment
```

#### On Windows:
```bash
venv\Scripts\activate
```

If you encounter an error about scripts being disabled, run PowerShell as administrator and execute:
```powershell
Set-ExecutionPolicy RemoteSigned
```
Then try activating the virtual environment again.

#### On Linux/MacOS:
```bash
source venv/bin/activate
```

### Step 4: Install dependencies

Once your virtual environment is activated, install the required packages:

```bash
pip install -r requirements.txt
```

If you're still encountering issues with the virtual environment, you can try installing the packages globally:

```bash
pip install streamlit numpy pandas tensorflow scikit-learn joblib lime
```

### Step 5: Verify Python version

```bash
python --version
```

Ensure it shows `Python 3.10.0`. If not, revisit Step 1.

### Step 6: Run the application

```bash
python -m streamlit run app1.py
```

The application should open in your default web browser. If it doesn't, navigate to the URL shown in the terminal (typically http://localhost:8501).

## Using the Application

1. **Login/Register**: Use the default admin account (username: admin, password: password) or register a new account
2. **Dashboard**: View system status and recent activity
3. **Manual Detection**: Input network traffic parameters manually
4. **Automatic Detection**: Upload a CSV file or use the included sample dataset
5. **Model Performance**: Compare different approaches and see the impact of GAN enhancement

## Troubleshooting

### Script Execution Issues on Windows
If you encounter errors about scripts not being allowed to run:
1. Open PowerShell as Administrator
2. Run: `Set-ExecutionPolicy RemoteSigned`
3. Try activating the virtual environment again

### Package Installation Problems
If you have trouble installing packages with pip:
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### Model Loading Errors
If the application shows errors loading the models:
1. Ensure all model files are in the correct location:
   - cnn_model.h5 / cnn_model.keras
   - lstm_model.h5 / lstm_model.keras
   - scaler.pkl
2. Try running without virtual environment:
   - Deactivate the virtual environment with `deactivate`
   - Run `python -m streamlit run app1.py`

### Path Issues
If you see "module not found" errors:
```bash
# Windows
set PYTHONPATH=%PYTHONPATH%;.

# Linux/MacOS
export PYTHONPATH=$PYTHONPATH:.
```

## Project Structure

- `app1.py`: Main application file
- `requirements.txt`: Required Python packages
- `cnn_model.keras`: Trained CNN model
- `lstm_model.keras`: Trained LSTM model
- `scaler.pkl`: Feature scaler for input normalization
- `balanced_sample_100_row_per_attack_cat.csv`: Sample dataset for testing

