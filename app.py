from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    email_text = request.form["email"]

    # Convert text to features
    email_vector = vectorizer.transform([email_text])

    prediction = model.predict(email_vector)

    if prediction[0] == 1:
        result = "Phishing Email Detected"
    else:
        result = "Legitimate Email"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
    