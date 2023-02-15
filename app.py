import openai
import pandas as pd
from flask import Flask, render_template

app = Flask(__name__)

# OpenAI API key
openai.api_key = "YOUR_API_KEY"

# Function to use OpenAI to generate text
def generate_text(prompt):
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        temperature=0.5,
        max_tokens=1000,
        n=1,
        stop=None,
        timeout=10,
    )
    return response.choices[0].text

# Load data into Pandas dataframe
data = pd.read_csv("data.csv")

# Route to display data
@app.route("/")
def index():
    return render_template("index.html", data=data.to_html())

# Route to generate text using OpenAI
@app.route("/openai")
def openai_text():
    prompt = "What is the meaning of life?"
    text = generate_text(prompt)
    return render_template("openai.html", prompt=prompt, text=text)

if __name__ == "__main__":
    app.run()
