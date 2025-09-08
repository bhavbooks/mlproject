from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app=application

## Route for home page

@app.route('/')
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=['GET',"POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        print("=== FORM SUBMISSION DEBUG ===")
        try:
            # Print all form data
            print("Form data received:")
            for key, value in request.form.items():
                print(f"  {key}: {value}")
            
            # Check if all required fields are present
            required_fields = ['gender', 'race_ethnicity', 'parental_level_of_education', 
                             'lunch', 'test_preparation_course', 'reading_score', 'writing_score']
            
            for field in required_fields:
                value = request.form.get(field)
                print(f"  {field}: {value}")
                if not value:
                    print(f"  WARNING: {field} is missing or empty!")
            
            data = CustomData(
                gender=request.form.get("gender"),
                race_ethnicity=request.form.get("race_ethnicity"),
                parental_level_of_education=request.form.get("parental_level_of_education"),
                lunch=request.form.get("lunch"),
                test_preparation_course=request.form.get("test_preparation_course"),
                reading_score=float(request.form.get("reading_score")),
                writing_score=float(request.form.get("writing_score"))
            )
            
            print("CustomData created successfully")
            
            pred_df = data.get_data_as_data_frame()
            print("DataFrame created:", pred_df)
            
            predict_pipeline = PredictPipeline() 
            results = predict_pipeline.predict(pred_df)
            
            print("Prediction successful:", results[0])
            return render_template('home.html', results=results[0])
            
        except Exception as e:
            print("ERROR:", str(e))
            import traceback
            traceback.print_exc()
            return render_template('home.html', results=f"Error: {str(e)}")
        
if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
