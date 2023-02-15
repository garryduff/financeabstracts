from flask import Flask, request, render_template

app = Flask(__name__, template_folder='path/to/templates')

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/calculate')
def calculate():
    x = int(request.args.get('x'))
    result = 3 * x
    return f"The result of 3 times {x} is {result}."

if __name__ == '__main__':
    app.run()

