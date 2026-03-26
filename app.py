from flask import Flask, render_template, request
import LinearRegression

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/use-cases')
def use_cases():
    return render_template('cases.html')

@app.route('/concepts')
def concepts():
    return render_template('concepts.html')

@app.route('/predict', methods=["GET", "POST"])
def predict():
    result = None
    hours = None
    if request.method == "POST":
        hours = float(request.form["hours"])
        result = LinearRegression.calculateGrade(hours)
        result = round(float(result), 2)
    return render_template("predict.html", result=result, hours=hours)

if __name__ == '__main__':
    app.run(debug=True)
