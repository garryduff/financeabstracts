from flask import Flask, request

app = Flask(__name__)

@app.route('/calculate')
def calculate():
    x = request.args.get('x')
    if x is None:
        return "Please provide a value for x."
    try:
        x = int(x)
    except ValueError:
        return "Invalid value for x: must be an integer."
    result = 3 * x
    return f"The result is {result}."

if __name__ == '__main__':
    app.run()
