from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained XGBoost model
model = pickle.load(open("model.pkl", "rb"))

# Map model output to disease names
disease_mapping = {
    0: "Thalassemia",
    1: "Hemophilia",
    2: "Breast Cancer",
    3: "Sickle Cell Anemia",
    4: "Cystic Fibrosis"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        Age = float(request.form['Age'])
        Gender = int(request.form['Gender'])
        Family_History = int(request.form['Family_History'])
        Hemoglobin = float(request.form['Hemoglobin'])
        Fetal_Hemoglobin = float(request.form['Fetal_Hemoglobin'])
        RDW_CV = float(request.form['RDW_CV'])
        Serum_Ferritin = float(request.form['Serum_Ferritin'])
        BRCA1_Expression = float(request.form['BRCA1_Expression'])
        p53_Mutation = int(request.form['p53_Mutation'])
        Sweat_Chloride = float(request.form['Sweat_Chloride'])
        Sickled_RBC_Percent = float(request.form['Sickled_RBC_Percent'])
        IL6_Level = float(request.form['IL6_Level'])

        # Prepare input for model
        input_features = np.array([[Age, Gender, Family_History, Hemoglobin, Fetal_Hemoglobin,
                                    RDW_CV, Serum_Ferritin, BRCA1_Expression, p53_Mutation,
                                    Sweat_Chloride, Sickled_RBC_Percent, IL6_Level]])

        # Predict disease
        prediction = model.predict(input_features)[0]
        disease = disease_mapping.get(prediction, "Unknown Disease")

        return render_template('result.html', disease=disease)
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
