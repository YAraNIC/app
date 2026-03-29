from flask import Flask, render_template, request

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

if __name__ == '__main__':
    app.run(debug=True)