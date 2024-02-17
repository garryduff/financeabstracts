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

    response = client.completions.create(
      model="ft:babbage-002:personal::8tJOCt1q",
      prompt="We develop a tractable model of systemic bank runs. The market-based banking system features a two-layer structure: banks with heterogeneous fundamentals face potential runs by their creditors while they trade short-term funding in the asset (interbank) market in response to creditor withdrawals. The possibility of a run on a particular bank depends on its assets' interim liquidation value, and this value depends endogenously in turn on the status of other banks in the asset market. The within-bank coordination problem among creditors and the cross-bank price externality feed into each other. A small shock can be amplified into a systemic crisis.->> ad yyyyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy yyy",
      temperature=1,
      max_tokens=25,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    message = response["choices"][0]["text"]
    originalquery = message

    return render_template('input.html', message=message, originalquery=originalquery) 
    



if __name__ == '__main__':
    app.run(debug=True)
    
