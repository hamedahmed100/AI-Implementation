# app.py
from flask import Flask, request, jsonify
import torch
import numpy as np
import pandas as pd
from model import NeuralNet  # Import your model architecture
from sklearn.preprocessing import StandardScaler
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
# ... other imports ...

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['STATIC_FOLDER'] = 'static'

# Load the scaler
scaler_filename = "scaler.joblib"  # Replace with the path to your scaler file
scaler = joblib.load(scaler_filename)

# Load the model
model_filename = "best_model.pth"  # Replace with the path to your model file
input_size = 463  # Replace with the correct input size
model = NeuralNet(input_size)
model.load_state_dict(torch.load(model_filename))
model.eval()

# Define a route to render the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for your prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from POST request
        data = request.json
        df = pd.DataFrame([data])
        # One-hot encode the data
        # Make sure to align feature order with training data
        df_encoded = pd.get_dummies(df)
        # Add missing columns with 0 value
        # Assuming 'encoded_columns' is a list of columns after one-hot encoding from your training set
        # Read the CSV file into a DataFrame
        df = pd.read_csv('D:\Hamed-lab\ML-Prj\LR\web\encoded_columns.csv')
        # Extract the values from the DataFrame as a list
        values_list = df['ColumnNames'].tolist()

        # Create a pandas Index object from the extracted values
        encoded_columns = pd.Index(values_list)

        for col in encoded_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded.reindex(columns=encoded_columns, fill_value=0)

        # Scale the features
        features_scaled = scaler.transform(df_encoded)

        # Convert to PyTorch tensor
        features_tensor = torch.tensor(features_scaled.astype(np.float32))

        # Get the model prediction
        with torch.no_grad():
            prediction_tensor = model(features_tensor)
        
        # Convert tensor to list
        # Assuming prediction_tensor contains the prediction value
        prediction = np.exp(prediction_tensor.numpy().tolist()[0][0])  # Convert tensor to float
        prediction_integer = int(prediction)  # Convert float to integer
        formatted_prediction = "{:,}".format(prediction_integer)
        prediction_with_currency = str(formatted_prediction) + " zł"  # Add " zł" to the integer


        # Send response
        return jsonify({'prediction': prediction_with_currency})
    
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)  # Set debug=False for production use

