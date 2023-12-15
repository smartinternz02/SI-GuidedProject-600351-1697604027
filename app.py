from flask import Flask, render_template, request, redirect
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, template_folder='templates')

model = pickle.load(open('model.pkl', 'rb'))
feature_cols = ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere']

label_dict = {}

label_dict['Age'] = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 60, 61, 62, 65, 72]
label_dict['Gender'] = ['female', 'male', 'trans']
label_dict['family_history'] = ['No', 'Yes']
label_dict['benefits'] = ["Don't know", 'No', 'Yes']
label_dict['care_options'] = ['No', 'Not sure', 'Yes']
label_dict['anonymity'] = ["Don't know", 'No', 'Yes']
label_dict['leave'] = ["Don't know", 'Somewhat difficult', 'Somewhat easy', 'Very difficult', 'Very easy']
label_dict['work_interfere'] = ["Don't know", 'Never', 'Often', 'Rarely', 'Sometimes']

for feature, encoding in label_dict.items():
    le = LabelEncoder()
    le.fit(encoding)
    label_dict[feature] = le

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    input_values = [
        int(request.form['Age']),
        label_dict['Gender'].transform([request.form['Gender']])[0],
        label_dict['family_history'].transform([request.form['family_history']])[0],
        label_dict['benefits'].transform([request.form['benefits']])[0],
        label_dict['care_options'].transform([request.form['care_options']])[0],
        label_dict['anonymity'].transform([request.form['anonymity']])[0],
        label_dict['leave'].transform([request.form['leave']])[0],
        label_dict['work_interfere'].transform([request.form['work_interfere']])[0]
    ]
        
    input_df = pd.DataFrame([input_values], columns=feature_cols)
        
    pred = model.predict(input_df)[0]
        
    if pred:
        return render_template("result.html",y="This person requires mental health treatment ")
    else:
        return render_template("result.html",y="This person doesn't require mental health treatment ")

if __name__ == '__main__':
    app.run(debug=True)