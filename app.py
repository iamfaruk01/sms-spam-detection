from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Loading the trained model and vectorizer
with open('model.pkl', 'rb') as f:
    model, feature_extraction = pickle.load(f)


# Function to predict spam or ham
def predict_spam_or_ham(message):
    # Transforming the input message into features using the loaded vectorizer
    input_features = feature_extraction.transform([message])
    prediction = model.predict(input_features)
    return prediction[0]  # Return 1 for Ham, 0 for Spam

@app.route('/', methods=['GET', 'POST'])
def predict():
    message = ''
    prediction_text = ''
    
    if request.method == 'POST':
        message = request.form['message']  # Get the message from the form
        
        # Running prediction model here using the `message`
        prediction = predict_spam_or_ham(message)
        
        # Setting the prediction result text
        if prediction == 1:
            prediction_text = "Ham message"
        else:
            prediction_text = "Spam message"
    
    # Rendering the template and pass message and prediction_text to the HTML
    return render_template('index.html', message=message, prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
