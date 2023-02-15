from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/calculate')
def calculate():
    # Get the value of x from the request and calculate 3x
    # ...

if __name__ == '__main__':
    app.run()
