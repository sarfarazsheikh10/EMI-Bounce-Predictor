from flask import Flask, render_template, request, url_for, make_response
import pickle
import pandas as pd

#Initialize app
app = Flask(__name__)

imputer = pickle.load(open('Models/Imputer', 'rb'))
model = pickle.load(open('Models/LGBMClassifier', 'rb'))
scaler = pickle.load(open('Models/Scaler', 'rb')) 

result = None

def process(data):

    global result

    # Customer id
    customer_id= data['cust_id']

    # Dropping some features 'v5', 'v13', 'v27', 'v28'
    data.drop(['cust_id', 'v5', 'v13', 'v27', 'v28' ], axis=1, inplace=True )

    # Encoding feature v15
    data['v15']= data['v15'].apply(lambda x: 1 if x=='YES' else 0)

    # Function for binning the salary feature
    def salary_cat(x):
        if(x<250000): return 1
        elif (250000<=x<500000): return 2
        elif (500000<=x<1000000): return 3
        elif (1000000<=x<2500000): return 4
        elif (2500000<=x<5000000): return 5
        elif (5000000<=x<10000000): return 6
        else: return 7 
    
    # Binning the salary feature
    data['v2']= data['v2'].apply(lambda x: salary_cat(x))
    
    # Imputing missing values in test data
    data.iloc[:]= imputer.transform(data)

    # Scaling data
    data= scaler.transform(data)

    # Get predictions
    pred= model.predict_proba(data)

    # Merge predctions with customer_id
    result= pd.concat([customer_id, pd.Series(pred[:,1])], axis=1)
    result.columns=['cust_id', 'bounceProbability']
    result.sort_values('bounceProbability', ascending=False, inplace=True)
    result.reset_index(drop=True, inplace=True,)

    return result
    

#Home page
@app.route('/', methods= ['GET','POST'])
def home():
    return render_template('index.html')

#Upload files    
@app.route('/upload', methods= ['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        df= pd.read_csv(file)
        output= process(df)
        return render_template('index.html', table= output.to_html())

#Download files
@app.route('/download')
def download():
    resp = make_response(result.to_csv(index=False))
    resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp

#Run app
if __name__ == "__main__":
    app.run(debug=True)