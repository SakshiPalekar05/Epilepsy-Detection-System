from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__,
            static_url_path="",
            static_folder="static")

# Define the function to preprocess a single text file
def preprocess_single_file(filepath, window_size=1024, stride=128):
    # Load the data from the text file
    data = np.loadtxt(filepath)
    
    # Initialize an empty list to store the processed segments
    processed_segments = []
    
    # Calculate the total number of segments
    total_segments = ((data.shape[0] - window_size) // stride) + 1
    
    # Iterate over each segment
    for i in range(total_segments):
        # Calculate the start and end indices of the segment
        start_index = i * stride
        end_index = i * stride + window_size
        
        # Extract the segment from the data
        segment = data[start_index:end_index]
        
        # Append the segment to the list of processed segments
        processed_segments.append(segment)
    
    # Convert the list of segments to a numpy array
    processed_data = np.array(processed_segments)
    
    return processed_data

def binary_classification_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(1024, 1)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    return model

def ternary_classification_model():
    model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(1024, 1)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(128, 3, activation='relu'),  # Increased number of filters
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(128, 3, activation='relu'),  # Additional convolutional layer
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),      # Increased size of dense layer
    tf.keras.layers.Dropout(0.5),                        # Adding dropout for regularization
    tf.keras.layers.Dense(3, activation='softmax')
    ])
    return model

ternary_ab_cd_e = ternary_classification_model()
ternary_ab_cd_e.load_weights("models/ternary_ab_cd_e.h5")

binary_cd_e = binary_classification_model()
binary_cd_e.load_weights("models/binary_cd_e.h5")

def predict_binary_cd_e():
    # Specify the filepath of the text file you want to classify
    # Preprocess the single text file
    preprocessed_data = preprocess_single_file("upload.txt")

    # Reshape the preprocessed data to match the input shape of the model
    preprocessed_data = preprocessed_data.reshape((preprocessed_data.shape[0], preprocessed_data.shape[1], 1))

    # Perform prediction
    predictions = binary_cd_e.predict(preprocessed_data)

    # Calculate the overall prediction for the entire file
    overall_prediction = 1 if np.mean(predictions) >= 0.5 else 0
    return overall_prediction, predictions
    
def predict_ternary_ab_cd_e():
    
    # Preprocess the single text file
    preprocessed_data = preprocess_single_file("upload.txt")

    # Reshape the preprocessed data to match the input shape of the model
    preprocessed_data = preprocessed_data.reshape((preprocessed_data.shape[0], preprocessed_data.shape[1], 1))

    # Perform prediction
    predictions = ternary_ab_cd_e.predict(preprocessed_data)

    # Calculate the overall prediction for the entire file
    overall_prediction = np.argmax(np.mean(predictions, axis=0))

    return overall_prediction, predictions


@app.route('/')
def index():
    return render_template('index.html')


import base64
from io import BytesIO

@app.route('/upload', methods=['POST'])
def upload_file():
    selected_option = request.form['option']

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400


    file.save("upload.txt")

    # Read the uploaded EEG data
    eeg_data = np.loadtxt("upload.txt")
    # Plot the EEG signal
    plt.figure(figsize=(10, 5))
    plt.plot(eeg_data)
    plt.title("Uploaded EEG Data Visualization")
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    plt.grid(True)
    plt.savefig("static/img/upload.png")
    plt.close()
    
    

    if selected_option == "cd_e":
        output, probabilities = predict_binary_cd_e()
        x = "Pre-Ictal" if output == 1 else "Ictal"
        probability = np.mean(probabilities)
        return render_template('result.html', result=x, accuracy=99.5, probability=probability)
    elif selected_option == "ab_cd_e":
        output, probabilities = predict_ternary_ab_cd_e()
        if output == 0:
            x = "Normal"
        elif output == 1:
            x = "Pre-Ictal"
        else:
            x = "Ictal"
        probability = np.mean(probabilities, axis=0)
        return render_template('result.html', result=x, accuracy=99.40, probability=probability)
    else:
        return """Kuch toh gadbad hai"""

    
if __name__ == '__main__':
    app.run(debug=True)
