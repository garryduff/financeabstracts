from flask import Flask, render_template, request
import os
import pandas as pd
import numpy as np
import openai

app = Flask(__name__, template_folder='templates')

openai.api_key=os.environ.get('OPENAI_API_KEY')
from openai import OpenAI
client = OpenAI()


@app.route('/')
def index():
    return render_template('input.html')


@app.route('/', methods=['POST'])
def generate_text():

    from openai import OpenAI
    client = OpenAI()

    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
      ]
    )

    message = completion.choices[0].message.content
    originalquery = message

    return render_template('input.html', message=message, originalquery=originalquery) 
    



if __name__ == '__main__':
    app.run(debug=True)
    
