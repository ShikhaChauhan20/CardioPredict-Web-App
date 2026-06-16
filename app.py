from flask import Flask, request, render_template
import joblib
import numpy as np
import json

app = Flask(__name__)

# Load model, scaler, columns
model = joblib.load('heart_model.pkl')
scaler = joblib.load('scaler.pkl')
with open('feature_columns.json') as f:
    feature_columns = json.load(f)

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get raw inputs
        age       = float(request.form['age'])
        sex       = int(request.form['sex'])
        trestbps  = float(request.form['trestbps'])
        chol      = float(request.form['chol'])
        fbs       = int(request.form['fbs'])
        thalach   = float(request.form['thalach'])
        exang     = int(request.form['exang'])
        oldpeak   = float(request.form['oldpeak'])
        cp        = int(request.form['cp'])
        restecg   = int(request.form['restecg'])
        slope     = int(request.form['slope'])
        ca        = int(request.form['ca'])
        thal      = int(request.form['thal'])

        # Build a dict matching your dummy-encoded columns
        input_dict = {col: 0 for col in feature_columns}

        # Numeric fields
        input_dict['age']      = age
        input_dict['sex']      = sex
        input_dict['trestbps'] = trestbps
        input_dict['chol']     = chol
        input_dict['fbs']      = fbs
        input_dict['thalach']  = thalach
        input_dict['exang']    = exang
        input_dict['oldpeak']  = oldpeak

        # One-hot encoded fields (drop_first=True was used)
        for col in [f'cp_{cp}', f'restecg_{restecg}', f'slope_{slope}', f'ca_{ca}', f'thal_{thal}']:
            if col in input_dict:
                input_dict[col] = 1

        # Convert to array
        input_array = np.array([list(input_dict.values())])

        # Scale numeric cols
        num_indices = [feature_columns.index(c) for c in ['age','trestbps','chol','thalach','oldpeak']]
        input_array[0, num_indices] = scaler.transform(
            input_array[0, num_indices].reshape(1, -1)
        )[0]

        # Predict
        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0][1] * 100

        if prediction == 1:
            result = f"⚠️ High Risk of Heart Disease ({probability:.1f}% probability)"
            color  = "red"
        else:
            result = f"✅ Low Risk of Heart Disease ({probability:.1f}% probability)"
            color  = "green"

    except Exception as e:
        result = f"Error: {str(e)}"
        color  = "orange"

    return render_template('index.html', result=result, color=color)

if __name__ == '__main__':
    app.run(debug=True)