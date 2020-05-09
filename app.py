import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import csv
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method=="POST":
        print('HELLO')
        machine_no=int(request.form['Machine_Number'])
        value=float(request.form['Value'])
        print(machine_no)
        print(value)
        #output=model.xyz(machine_no,value)
        df=pd.read_csv("DS1_signals.csv")
        int_features=[]
        for i in range(1,df.shape[1]):
            sum=0
            for j in range(0,df.shape[0]):
                sum+=df[str(i)][j]
            if(i==machine_no):
                int_features.append(value)
            else:
                temp=float(sum/180)
                int_features.append(temp)
            #int_features[i]=round(int_features[i], 2)
        #print(int_features)
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)
        di={0:"Atrial Fibrillation", 1:"Atrial flutter", 2:"Ventricular Flutter", 3:"Ventricular Tachycardia"}
        output=di[output]
        return render_template('index.html', prediction_text='The patient has {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)