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

    prompt = request.form['prompt']
    prompt = prompt.rstrip()
    prompt = prompt + '->>'


    response = client.chat.completions.create(
      model="ft:gpt-3.5-turbo-1106:personal::8tPyu3pp",
      messages=[
        {
          "role": "system",
          "content": "You classify academic abstracts to the most appropriate journal.\n\nYou only return the short code for the most appropriate journal provided in the fine tuning data."
        },
        {
          "role": "user",
          "content": prompt
        },
      ],
      temperature=0.3,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      logprobs=True,
      top_logprobs=3
    )


    top_logprobs = response.choices[0].logprobs.content[0].top_logprobs

    flattened_logprobs = [(top_logprob.token, top_logprob.logprob) for top_logprob in top_logprobs]

    df = pd.DataFrame(flattened_logprobs, columns=['Token', 'LogProb'])

    df.sort_values(by='LogProb', ascending=False, inplace=True)

    top3 = df.head(3)

    results = [{
        "query": prompt,  # Ensure 'prompt' is defined somewhere in your code
        "response1": top3.iloc[0, 0], "value1": top3.iloc[0, 1],
        "response2": top3.iloc[1, 0], "value2": top3.iloc[1, 1],
        "response3": top3.iloc[2, 0], "value3": top3.iloc[2, 1]
    }]

    df_results = pd.DataFrame(results)

    # Calculate the exponent of the log probabilities to get probabilities
    df_results['prob1'] = np.exp(df_results['value1'])
    df_results['prob2'] = np.exp(df_results['value2'])
    df_results['prob3'] = np.exp(df_results['value3'])

    # Calculate the sum of probabilities
    df_results['probsum'] = df_results[['prob1', 'prob2', 'prob3']].sum(axis=1)

    # Calculate the ratio of each probability to the sum of probabilities
    df_results['probratio1'] = df_results['prob1'] / df_results['probsum']
    df_results['probratio2'] = df_results['prob2'] / df_results['probsum']
    df_results['probratio3'] = df_results['prob3'] / df_results['probsum']

    lookup_names = {
    'ad': 'Journal of Finance (4+)',
    'ab': 'Journal of Financial Economics (4+)',
    'ac': 'Review of Financial Studies (4+)',
    }

    lookup_stars = {
    'ad': '5',
    'ab': '5',
    'ac': '5',  
    }

    df_results['prednames1'] = df_results['response1'].map(lookup_names).fillna('Unclassified (0)')
    df_results['prednames2'] = df_results['response2'].map(lookup_names).fillna('Unclassified (0)')
    df_results['prednames3'] = df_results['response3'].map(lookup_names).fillna('Unclassified (0)')

    df_results['predstar1'] = df_results['response1'].map(lookup_stars).fillna(0)
    df_results['predstar2'] = df_results['response2'].map(lookup_stars).fillna(0)
    df_results['predstar3'] = df_results['response3'].map(lookup_stars).fillna(0)
    df_results['predstar1'] = df_results['predstar1'].astype(float)
    df_results['predstar2'] = df_results['predstar2'].astype(float)
    df_results['predstar3'] = df_results['predstar3'].astype(float)

    df_results['predstarall'] = df_results['predstar1']*df_results['probratio1'] + df_results['predstar2']*df_results['probratio2'] + df_results['predstar3']*df_results['probratio3'] 
    df_results['predstarall2dp'] = df_results['predstarall'].round(1)

    df_results['predxb5'] = 2.868937 * df_results['predstarall'] - 13.37742
    df_results['predprob5'] = 1/(1+np.exp(-1*df_results['predxb5']))
    df_results['predprob5pct'] = df_results['predprob5']*100
    df_results['predprob5pct2dp'] = df_results['predprob5pct'].round(1)

    df_results['predxb4'] = 2.262539 * df_results['predstarall'] - 8.31851
    df_results['predprob4'] = 1/(1+np.exp(-1*df_results['predxb4']))
    df_results['predprob4pct'] = df_results['predprob4']*100
    df_results['predprob4pct2dp'] = df_results['predprob4pct'].round(1)

    predstarall2dp = str(df_results.at[0, 'predstarall2dp'])
    predprob5pct2dp = str(df_results.at[0, 'predprob5pct2dp'])
    predprob4pct2dp = str(df_results.at[0, 'predprob4pct2dp'])
    prednames1 = str(df_results.at[0, 'prednames1'])
    prednames2 = str(df_results.at[0, 'prednames2'])
    prednames3 = str(df_results.at[0, 'prednames3'])
    
    df_results['queryx'] = df_results['query'].str.replace(r'->>', '')
    originalquery = str(df_results.at[0, 'queryx'])   
    
    message = "This abstract scores " + predstarall2dp + " out of a maximum of 5.0. Other abstracts of this score have an estimated success rate of " + predprob5pct2dp + "% of being published in a world elite (4+ star) journal, and " + predprob4pct2dp + "% of being published in at least a leading (4 star) journal. The most likely venues for publication are: " + prednames1 + ", " + prednames2 + ", or " + prednames3 + "." 

    return render_template('input.html', message=message, originalquery=originalquery) 
    



if __name__ == '__main__':
    app.run(debug=True)
    
