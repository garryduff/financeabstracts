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


    response1 = client.chat.completions.create(
      model="ft:gpt-3.5-turbo-1106:personal::8tPwFsXu",
      messages=[
        {
          "role": "system",
          "content": "You classify academic abstracts to the most appropriate journal.\n\nYou only return the name of the most appropriate journal provided in the fine tuning data."
        },
        {
          "role": "user",
          "content": prompt
        },
      ],
      temperature=0,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    model1 = response1.choices[0].message.content



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
      temperature=0,
      max_tokens=1,
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
    'ba': 'Review of Finance (4)',
    'bl': 'Journal of Money, Credit and Banking (4)',
    'bc': 'Journal of Financial and Quantitative Analysis (4)',
    'bo': 'Journal of Financial Intermediation (4)',
    'be': 'Journal of Corporate Finance (4)',
    'ca': 'Review of Corporate Finance Studies (3)',
    'ma': 'Review of Asset Pricing Studies (3)',
    'cc': 'Quantitative Finance (3)',
    'cd': 'Mathematical Finance (3)',
    'ce': 'Journal of Portfolio Management (3)',
    'cf': 'Journal of International Financial Markets, Institutions and Money (3)',
    'me': 'Journal of Futures Markets (3)',
    'ch': 'Journal of Financial Stability (3)',
    'mo': 'Journal of Financial Services Research (3)',
    'mu': 'Journal of Financial Research (3)',
    'na': 'Journal of Financial Markets (3)',
    'cl': 'Journal of Financial Econometrics (3)',
    'cm': 'Journal of Empirical Finance (3)',
    'ne': 'Journal of Commodity Markets (3)',
    'co': 'Journal of Banking and Finance (3)',
    'cp': 'International Review of Financial Analysis (3)',
    'no': 'Financial Review (3)',
    'cr': 'Financial Management (3)',
    'cs': 'Financial Analysts Journal (3)',
    'nu': 'European Journal of Finance (3)',
    'cu': 'European Financial Management (3)',
    'da': 'Journal of Economic History (3)',
    'sa': 'Explorations in Economic History (3)',
    'dc': 'European Review of Economic History (3)',
    'se': 'Economic History Review (4)',
    'so': 'Business History Review (4)',
    'df': 'Business History (4)',
    'ta': 'Investment Analysts Journal (1)',
    'te': 'Journal of Asset Management (2)',
    'to': 'Finance Research Letters (2)',
    'tu': 'Journal of Derivatives (2)',
    'tr': 'Research in International Business and Finance (2)',
    'tn': 'Managerial Finance (1)',
    'ka': 'Journal of Management History (1)',
    'ke': 'Financial History Review (2)',
    'ko': 'Journal of European Economic History (1)',
    'ku': 'Cliometrica (2)',
    'k': 'Cliometrica (2)',        
    'ga': 'American Economic Review (4+)',
    'la': 'Quarterly Journal of Economics (4+)',
    'le': 'Review of Economic Studies (4+)',
    'lo': 'Journal of Political Economy (4+)',
    'ge': 'Econometrica (4+)',
    'ha': 'Oxford Economic Papers (3)',
    'ra': 'Economics Letters (3)',
    're': 'Economic Journal (4)',
    'ro': 'Applied Economics (2)',
    'ru': 'Journal of Monetary Economics (4)',   
    'va': 'European Journal of Operational Research (4)',
    'Va': 'European Journal of Operational Research (4)',   
    'Va': 'European Journal of Operational Research (4)',        
    've': 'Management Science (4+)',
    'Ve': 'Management Science (4+)',   
    'vo': 'International Journal of Forecasting (3)',        
    'ud': 'Unclassified (0)',
    'ub': 'Unclassified (0)',
    'un': 'Unclassified (0)', 
    }

    lookup_stars = {
    'ad': '5',
    'ab': '5',
    'ac': '5',
    'ba': '4',
    'bl': '4',
    'bc': '4',
    'bo': '4',
    'be': '4',
    'ca': '3',
    'ma': '3',
    'cc': '3',
    'cd': '3',
    'ce': '3',
    'cf': '3',
    'me': '3',
    'ch': '3',
    'mo': '3',
    'mu': '3',
    'na': '3',
    'cl': '3',
    'cm': '3',
    'ne': '3',
    'co': '3',
    'cp': '3',
    'no': '3',
    'cr': '3',
    'cs': '3',
    'nu': '3',
    'cu': '3',
    'da': '3',
    'sa': '3',
    'dc': '3',
    'se': '4',
    'so': '4',
    'df': '4',
    'ta': '1',
    'te': '2',
    'to': '2',
    'tu': '2',
    'tr': '2',
    'tn': '1',
    'ka': '1',
    'ke': '2',
    'ko': '1',
    'ku': '2',
    'k': '2',        
    'ga': '5',
    'la': '5',
    'le': '5',
    'lo': '5',
    'ge': '5',
    'ha': '3',
    'ra': '3',
    're': '4',
    'ro': '2',
    'ru': '4',        
    'ud': '0',
    'ub': '0',
    'un': '0', 
    'va': '4',
    'Va': '4',          
    've': '5',
    'Ve': '5', 
    'vo': '3',  
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

    # message = "MODEL 1: " + model1 +  "\n\nMODEL 2: This abstract scores " + predstarall2dp + " out of a maximum of 5.0. Other abstracts of this score have an estimated success rate of " + predprob5pct2dp + "% of being published in a world elite (4+ star) journal, and " + predprob4pct2dp + "% of being published in at least a leading (4 star) journal. The most likely venues for publication are: " + prednames1 + ", " + prednames2 + ", or " + prednames3 + "." 

    message = "The base model predicts that the most likely venue is " + model1 +  ". A probabilistic model suggests the following possibilities " + prednames1 + ", " + prednames2 + ", or " + prednames3 + "." 

    return render_template('input.html', message=message, originalquery=originalquery) 
    



if __name__ == '__main__':
    app.run(debug=True)
    
