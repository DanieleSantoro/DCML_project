#!/bin/bash

echo "Registrazione input tastiera per 10 secondi..."
python keyboard_tracker.py --output_file tmp_input.csv --duration 10

echo "Estrazione feature..."
python feature_extractor.py --input_dir . --output_file tmp_features.csv

echo "Predizione..."
python predict.py --input_file tmp_features.csv

# Pulizia
rm tmp_input.csv tmp_features.csv

