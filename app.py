from flask import Flask,render_template, request
import LinealRegression

app = Flask(__name__)

@app.route('/')
def home():
    return  "Hello Flask"

@app.route('/FirstPage')
def firstPage(name = ''):
    return render_template('index.html', name = name)


@app.route('/LinearRegression/', methods=["GET","POST"])
def calculateGrade():
    calculateResult=None
    if request.method == "POST":

        hours = float(request.form["hours"])

        calculateResult= LinealRegression.calculateGrade(hours)

    return render_template("LinearRegressionGrades.html", result = calculateResult)