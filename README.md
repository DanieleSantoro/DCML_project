# DCML 24-25 - Anomaly Detection System

This project implements an **anomaly detection system** for laptops or workstations based on keystroke dynamics and system monitoring.  
It is part of the **DCML 2024-2025** coursework and demonstrates both classical machine learning and recurrent neural network (RNN) approaches for binary classification.

## Project Structure

- **auto_typer.py**: script that prints a standard string ("hello world") or a personalized string for a indefined time.
- **keyboard_tracker.py** → Records keyboard input for a fixed duration.  
- **feature_extractor.py** → Extracts statistical and timing features from raw input data.  
- **feature_extractor_sequential.py** → Builds sequential datasets suitable for RNN training.  
- **monitor_data.py** → Captures system usage parameters (CPU, memory, etc.).  
- **RNN.py** → Defines the recurrent neural network model (PyTorch).  
- **train_model.py** → Trains classical machine learning models (SVM, Random Forest, etc.).  
- **train_rnn.py** → Trains the RNN using sequential data.  
- **predict.py** → Loads a trained model and performs real-time predictions.  
- **auto_scroller.py** / **auto_typer.py** → Utilities to simulate typing for testing purposes.  
- **demo_predict.sh** → End-to-end demo: records input, extracts features, predicts anomalies.  
- **dataset/** → Contains raw and processed data samples.  

## Installation

Clone the repository and install dependencies on the terminal:

git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt

## Usage

1) Activate the virtual environment: source venv_autotyper/bin/activate

# Generating the raw data

2) To run auto_typer.py you can choose some options, like:
        python auto_typer.py --text **something** --delay **number**

2b) In order to collect the human raw data, you must follow the 3) point on one terminal while writing something on a blank file, page, etc.

# Collecting and labeling raw data

3) To run keyboard_tracker.py there are many options, for example: 
        python keyboard_tracker.py --output_file dataset/raw_input.csv --duration 30

# Extracting features from raw data

4) You need to run: 
        python feature_extractor.py --input_dir dataset/ --output_file dataset/features.csv

# Training the RNN model

5) Type: python RNN.py 

6) Then you can train the RNN, so for example you can type: python train_rnn.py --epochs 20 --batch_size 32

# Training the classical models and choosing the best one:

7) Type on you terminal: python train_model.py (or train_classical_model.py ????)

# You can predict data

8) python predict.py --input_file features.csv --model_path model.pkl

 



