from enum import unique
from flask import Flask,render_template,request,redirect
from flask_sqlalchemy import SQLAlchemy
import os
import pickle
import requests
import socket
from bs4 import BeautifulSoup

# getting the current working directory and creating the database file
project_dir=os.path.dirname(os.path.abspath(__file__))
database_file="sqlite:///{}".format(os.path.join(project_dir,"mydatabase.db"))

def getData(url):
    '''function for getting data from url'''
    r=requests.get(url)
    # print(r.text)
    return  r.text

# Un-pickling
file=open('detectorModel.pkl','rb') 
clf=pickle.load(file)
file.close()

app=Flask(__name__)

# configuring the database_file with the flask app
app.config["SQLALCHEMY_DATABASE_URI"]=database_file
db=SQLAlchemy(app)

class Patient(db.Model):
    # Creating the database model

    id = db.Column(db.Integer , primary_key=True,autoincrement=True)
    name=db.Column(db.String(100),unique=False,nullable=False)
    city=db.Column(db.String(100),unique=False,nullable=False)
    cough=db.Column(db.Integer,unique=False,nullable=False)
    fever=db.Column(db.Integer,unique=False,nullable=False)
    bpain=db.Column(db.Integer,unique=False,nullable=False)
    age=db.Column(db.Integer,unique=False,nullable=False)
    rNose=db.Column(db.Integer,unique=False,nullable=False)
    diffBreath=db.Column(db.Integer,unique=False,nullable=False)
    travelled=db.Column(db.Integer,unique=False,nullable=False)
    infectionProb=db.Column(db.Float,unique=False,nullable=False)


@app.route('/')
def main():
    # Rendering the landing page
    return render_template('index.html')

@app.route('/detector',methods=["POST","GET"])
def detector():
    # Getting the data from the form and using that data to predict using the Logistic Regression Model Object

    # code for inference
    if request.method=="POST":
        formDict=request.form
        if formDict['name']=='' or formDict['name']=='' or formDict['city']=='' or formDict['fever']=='':
            return render_template('detector.html')
        name=formDict['name']
        city=formDict['city']
        cough=int(formDict['cough'])
        fever=int(formDict['fever'])
        pain=int(formDict['pain'])
        age=int(formDict['age'])
        rNose=int(formDict['runnyNose'])
        diffBreath=int(formDict['diffBreath'])
        travelled=int(formDict['travelled'])

        infection=clf.predict_proba([[fever,pain,age,rNose,diffBreath,travelled,cough]])
        infProb=infection[0][1]

        print(name,city,cough)

        # Adding entries to the database
        patientObj=Patient(name=name,city=city,cough=cough,fever=fever,bpain=pain,age=age,rNose=rNose,diffBreath=diffBreath,travelled=travelled,infectionProb=round(infProb,2))
        db.session.add(patientObj)
        db.session.commit()

        print("Data saved into database successfully!")
        
        return render_template('result.html',name=name,inf=round(infProb*100))

    return render_template('detector.html')

@app.route('/status',methods=["POST",'GET'])
def status():
    # For fetching the Real-time Covid Data across India 

    # Checking internet connection (If internet connection exists then fetch the data)
    IPaddress = socket.gethostbyname(socket.gethostname())
    if IPaddress == "127.0.0.1":
        print("No internet")
        return "OOps! No internet connection."
    else:
        myHtmlData = getData("https://prsindia.org/covid-19/cases")

        soup = BeautifulSoup(myHtmlData, 'html.parser')

        record=[]
        if len(soup.find_all('tbody'))==0:
            return "Data not available"
        for tr in soup.find_all('tbody')[0].find_all('tr'):
            li=[]
            for td in tr.find_all('td'):
                li.append(td.get_text())
            record.append(li)


        headings=['S. No.','State','Confimed','Active','Discharged','Deaths']
        total_confirmed=0
        total_active=0
        total_discharged=0
        total_death=0

        for i in record:
            total_confirmed=total_confirmed+int(i[2])
            total_active=total_active+int(i[3])
            total_discharged=total_discharged+int(i[4])
            total_death=total_death+int(i[5])

    return render_template('status.html',headings=headings,record=record,total_confirmed=total_confirmed,total_active=total_active,
                            total_discharged=total_discharged,total_death=total_death)

@app.route('/readdb')
def readdb():
    # For Rendering the html file which will display the Data Stored in the database
    patients=Patient.query.all()
    return render_template('readdb.html',patients=patients)

@app.route('/about')
def about():
    # For rendering the about page
    return render_template('about.html')

@app.route('/pankaj_fb')
def pankaj_fb():
    # This will redirect to Pankaj's fb page
    return redirect("https://www.facebook.com/people/Pankaj-Kumar/100004369415341", code=302)

@app.route('/pankaj_linkedin')
def pankaj_linkedin():
    # This will redirect to Pankaj's LinkedIn page
    return redirect("https://www.linkedin.com/in/pankaj-kumar-353358120/", code=301)

@app.route('/pankaj_twitter')
def pankaj_twitter():
    # This will redirect to Pankaj's Twitter page
    return redirect("https://twitter.com/Pankaj53175102", code=302)

@app.route('/pankaj_github')
def pankaj_github():
    # This will redirect to Pankaj's github page
    return redirect("https://github.com/panks123", code=302)


# (Like main method)
if __name__=='__main__':
    app.run(debug=True)