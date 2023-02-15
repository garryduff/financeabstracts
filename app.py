from flask import Flask, render_template, request
import openai
import os

app = Flask(__name__, template_folder='templates')

openai.api_key=os.environ.get('OPENAI_API_KEY')

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/', methods=['POST'])
def generate_text():
    prompt = request.form['prompt']
    model_engine = "text-davinci-002"
    completions = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=60,
        n=1,
        stop=None,
        temperature=0.5,
    )
    message = completions.choices[0].text.strip()
    return render_template('input.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)
    
