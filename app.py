from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from restaurant import CafeAnalyzer
import traceback
import json

app = Flask(__name__)
CORS(app)

analyzer = CafeAnalyzer()

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route('/save_cafe_data', methods=['POST'])
def save_cafe_data():
    try:
        data = request.json
        processed_data = analyzer.process_data(data)
        return jsonify({
            "message": "Veriler başarıyla işlendi ve sıralandı",
            "count": len(data),
            "processed_data": processed_data
        }), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)