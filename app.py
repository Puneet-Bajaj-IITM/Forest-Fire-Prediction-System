import pickle, bz2
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from app_logger import log
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)

# Import Classification and Regression model file
with open('linear_regression_model.pkl', 'rb') as file:
    model_C = pickle.load(file)
R_pickle = bz2.BZ2File('Regression.pkl', 'rb')
model_R = pickle.load(R_pickle)



# Route for homepage
@app.route('/')
def home():
    log.info('Home page loaded successfully')
    return render_template('index.html')



# Route for Classification Model

@app.route('/predictC', methods=['POST', 'GET'])
def predictC():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            Temperature=float(request.form['Temperature'])
            Wind_Speed =float(request.form['Ws'])
            FFMC=float(request.form['FFMC'])
            DMC=float(request.form['DMC'])
            ISI=float(request.form['ISI'])
            RH=float(request.form['RH'])
            rain=float(request.form['rain'])


            features = [Temperature, Wind_Speed,FFMC, DMC, ISI, RH, rain]

            Float_features = [float(x) for x in features]
            final_features = [np.array(Float_features)]
            prediction = model_C.predict(final_features)[0]

            log.info('Prediction done for Classification model')

            if prediction < 250:
                text = 'Not Fire'
            else:
                text = 'Fire!'
            return render_template('index.html', prediction_text1="{}".format(text))
        except Exception as e:
            log.error('Input error, check input', e)
        return render_template('index.html', prediction_text1="Check the Input again!!!")

            


# Run APP in Debug mode

if __name__ == "__main__":
    app.run(debug=True, port= 5000)
