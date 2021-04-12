from flask import Flask, render_template,request
import numpy as np
import pickle
from flask_bootstrap import Bootstrap
import matplotlib.pyplot as plt
from io import BytesIO
from tabulate import tabulate
from flask import Markup
import base64
app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index_page():
    return render_template("index.html")
@app.route('/about_us')
def about_us():
    email="sagarhimanshu355@gmail.com"
    phone="9852467064"
    return render_template("about_us.html")
@app.route('/contact_us')
def contact_us():
    return render_template("contact_us.html")

def factorial(n):  
    return 1 if (n==1 or n==0) else n * factorial(n - 1); 


@app.route('/regression',methods=["POST","GET"])
def regression():
    if request.method=="POST":
        num=request.form["n"]
        if num=="":
            return render_template("regression.html")
        else:
            num=int(num)
            fact=factorial(num)
            data="Factorial of "+str(num)+": "+str(fact)
            
            return render_template("regression.html",factorial1=data)
    else:
        return render_template("regression.html")


@app.route('/train_classification',methods=["POST","GET"])
def train_classification():
    global dataset
    global algo
    global m
    global X_t
    global Y
    if request.method== 'POST':
        form_data=request.form.to_dict()
        print(form_data)
        algo=form_data['algo']
        dataset=form_data['dataset']
        if dataset=='wine':
            import wine as model
            code_path="static\code_wine.txt"
            data_head_path="static\images\datahead_wine.png"
        elif dataset=='parkinsons':
            import parkinsons as model
            code_path="static\code_parkinsons.txt"
            data_head_path="static\images\datahead_parkinsons.png"
        elif dataset=='iris':
            import iris as model
            code_path="static\code_iris.txt"
            data_head_path="static\images\datahead_iris.png"
        elif dataset=='phishing_website_detection':
            import phishing_website_detection as model
            code_path="static\code_phishing.txt"
            data_head_path="static\images\datahead_phishing.png"
        model_details,accuracy,class_report,con_matrix,m,plot_url_0,plot_url_1,plot_url_2,plot_url_4,X_t,Y = model.run(algo,dataset)
        
        table_data = class_report
        class_report_html=tabulate(table_data,tablefmt='html')

        table_data_con = con_matrix
        con_matrix_html=tabulate(table_data_con,tablefmt='html')

        return render_template("classification_model.html",model_detail=str(model_details),acc=str(accuracy),report=Markup(class_report_html),con_matrix=con_matrix_html,graph0=plot_url_0,graph1=plot_url_1,graph2=plot_url_2,graph4=plot_url_4,dataset_name=dataset,algo=algo,code_path=code_path,data_head_path=data_head_path)

    
    return render_template("train_classification.html")

@app.route('/prediction', methods = ["POST","GET"])
def prediction():
    if request.method == 'POST':
        actual_header= "Actual Output"
        predict_header="Predicted Output"
        row = request.form['row']
        row=int(row)
        if dataset=='parkinsons':
            loaded_model = pickle.load(open("static/model_parkinsons.pkl", "rb")) 
            predicted = loaded_model.predict(X_t[row:row+1,:])
            actual=Y[row]
            
            prediction=""
            actual1=""
        
            if predicted == 1:
                prediction ='Person is suffering from Parkinsons Disease'
            else: 
                prediction ='Person is Not suffering from Parkinsons Disease'	
            if actual == 1.0:
                actual1 ='Person is suffering from Parkinsons Disease'
            else: 
                actual1 ='Person is Not suffering from Parkinsons Disease'	
        elif dataset=='wine':
            loaded_model = pickle.load(open("static/model_wine.pkl", "rb"))
            predicted = loaded_model.predict(X_t[row:row+1,:])
            actual=Y[row]
            prediction=""
            actual1=""
            if predicted == 1:
                prediction ='Wine Quality is Best'
            elif predicted==2:
                prediction ='Wine Quality is Good'
            elif predicted==3:
                prediction ='Wine Quality is not Good'
            if actual == 1:
                actual1 ='Wine Quality is Best'
            elif actual==2:
                actual1 ='Wine Quality is Good'
            elif actual==3:
                actual1 ='Wine Quality is not Good'

        elif dataset=='iris':
            loaded_model = pickle.load(open("static/model_iris.pkl", "rb")) 
            predicted = loaded_model.predict(X_t[row:row+1,:])
            actual=Y[row]
            
            prediction=""
            actual1=""
        
            if predicted == 0:
                prediction ='Iris-setosa'
            elif predicted==1:
                prediction ='Iris-versicolor'
            elif predicted==2:
                prediction ='Iris-virginica'
            if actual == 0:
                actual1 ='Iris-setosa'
            elif actual==1:
                actual1 ='Iris-versicolor'
            elif actual==2:
                actual1 ='Iris-virginica'
        
        elif dataset=='phishing_website_detection':
            loaded_model = pickle.load(open("static/phishing_website_detection.pkl", "rb")) 
            predicted = loaded_model.predict(X_t[row:row+1,:])
            actual=Y[row]
            
            prediction=""
            actual1=""
        
            if predicted == 0:
                prediction ='Phishing Website'
            elif predicted==1:
                prediction ='Legitimate Website'
            if actual == 0:
                actual1 ='Phishing Website'
            elif actual==1:
                actual1 ='Legitimate Website'
        
        return render_template("prediction.html",actual=actual1,prediction=prediction,actual_header=actual_header,predict_header=predict_header,m=m,dataset_name=dataset,algo=algo)

            
    return render_template("prediction.html",dataset_name=dataset,algo=algo,m=m)

@app.route('/NeuralNetworks', methods = ["POST","GET"])
def NeuralNetworks():
    return render_template("train_NeuralNetwork.html")



@app.route('/test_layout',methods=["POST","GET"])
def test_layout():
    return render_template('test_layout.html')


app.run()