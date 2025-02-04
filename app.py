from flask import Flask, render_template, request
from model import pred_art, model, token

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        article = request.form['article']
        prediction, prediction_label = pred_art(article, model, token)
        return render_template('result.html', prediction=prediction[0][0], prediction_label=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
