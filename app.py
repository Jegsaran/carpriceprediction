import json
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

## Load the model
dtrmodel=pickle.load(open('dtrmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data) 
    print(np.array(data.values).reshape(1,-1))
    scaled_data = scalar.transform(np.array(data.values).reshape(1,-1))
    new_data = pca.transform(np.array(scaled_data).reshape(1,-1))
    output = dtrmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])


@app.route('/predict',methods=['POST'])
def predict():
    data =[float(x) for x in request.form.values()]
    scaled_input = scalar.transform(np.array(data).reshape(1,-1))
    final_input = pca.transform(np.array(scaled_input).reshape(1,-1))
    print(final_input)
    output = dtrmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text = "The predicted Car price is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)