from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html', results=None)
    else:
        # Gather form data
        data = CustomData(
            CreditScore=int(request.form.get('CreditScore')),
            Geography=request.form.get('Geography'),
            Gender=request.form.get('Gender'),
            Age=int(request.form.get('Age')),
            Tenure=int(request.form.get('Tenure')),
            Balance=float(request.form.get('Balance')),
            NumOfProducts=int(request.form.get('NumOfProducts')),
            HasCrCard=int(request.form.get('HasCrCard')),
            IsActiveMember=int(request.form.get('IsActiveMember')),
            EstimatedSalary=float(request.form.get('EstimatedSalary'))
        )
        pred_df = data.get_data_as_data_frame()
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Map numeric output to label if needed (e.g., 0 = No churn, 1 = Churn)
        prediction_label = "Churn" if results[0] == 1 else "No Churn"

        return render_template('home.html', results=prediction_label)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=True)
