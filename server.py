from flask import Flask, request, jsonify
import util
app = Flask(__name__)

@app.route('/api/get_location_names')
def get_location_names():
    print (util.get_location_names())
    response = jsonify({
        'locations':util.get_location_names()
    })
    response.headers.add("Access-Control-Allow-Origin","*")
    return response

@app.route('/api/predict_home_price', methods=['POST'])
def predict_home_price():
    
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    response = jsonify({
        'estimated_price':util.get_estimate_price(location,total_sqft,bhk,bath) 
    })

    response.headers.add("Access-Control-Allow-Origin","*")

    return response



if __name__=="__main__":
    print("Starting PythonFlask Server")
    app.run(port=10000)