from flask import Flask, render_template, request
import LogisticRegressionModel
import MentalHealth

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/use-cases')
def use_cases():
    return render_template('cases.html')

# ==================== Change Menu use Cases ====================
@app.route('/use-cases/case1')
def case1():
    return render_template('case1.html')

@app.route('/use-cases/case2')
def case2():
    return render_template('case2.html')

@app.route('/use-cases/case3')
def case3():
    return render_template('case3.html')

@app.route('/use-cases/case4')
def case4():
    return render_template('case4.html')
# =====================================================================

@app.route('/concepts')
def concepts():
    return render_template('concepts.html')


@app.route('/burnout-concepts')
def burnout_concepts():
    return render_template('burnout_concepts.html')

@app.route('/burnout', methods=["GET", "POST"])
def burnout():
    result = None
    form_data = {}
    model_info = MentalHealth.getModelInfo()

    if request.method == "POST":
        form_data = {
            "stress_level":  float(request.form["stress_level"]),
            "anxiety_score": float(request.form["anxiety_score"]),
            "sleep_hours":   float(request.form["sleep_hours"]),
            "study_hours":   float(request.form["study_hours"]),
        }
        result = MentalHealth.predictBurnout(**form_data)

    return render_template("burnout.html",
                           result=result,
                           form_data=form_data,
                           model_info=model_info)

@app.route('/logistic-concepts')
def logistic_concepts():
    return render_template('logistic_concepts.html')

@app.route('/logistic-application', methods=["GET", "POST"])
def logistic_application():
    result    = None
    form_data = {}
    options    = LogisticRegressionModel.getOptions()
    model_info = LogisticRegressionModel.getModelInfo()

    if request.method == "POST":
        try:
            form_data = {
                "price":      float(request.form["price"]),
                "category":   request.form["category"],
                "payment":    request.form["payment"],
                "month":      int(request.form["month"]),
                "is_weekend": int(request.form["is_weekend"]),
            }
            result = LogisticRegressionModel.predictSavings(**form_data)
        except Exception as e:
            print("Error en formulario:", e)

    return render_template("logistic_application.html",
                           result=result,
                           form_data=form_data,
                           options=options,
                           model_info=model_info)

if __name__ == '__main__':
    app.run(debug=True)