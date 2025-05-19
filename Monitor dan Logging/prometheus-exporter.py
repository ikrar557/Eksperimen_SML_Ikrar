from flask import Flask, request, jsonify, Response
import requests
import time
import psutil
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
)

app = Flask(__name__)

# === HTTP & System Metrics ===
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests')
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency')
THROUGHPUT = Counter('http_requests_throughput', 'Total number of requests per second')
CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage Percentage')
RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage Percentage')
DISK_USAGE = Gauge('system_disk_usage', 'Disk Usage Percentage')

# === ML-specific Metrics ===
INFERENCE_SUCCESS = Counter('model_inference_success_total', 'Successful Inferences')
INFERENCE_FAIL = Counter('model_inference_fail_total', 'Failed Inferences')
PREDICTION_LATENCY = Histogram('model_prediction_latency_seconds', 'Prediction Latency')
OUTPUT_CLASS_DISTRIBUTION = Counter('model_output_class_distribution', 'Class distribution', ['class_label'])
MODEL_INVOCATION_INPUT_SIZE = Gauge('model_input_sample_size', 'Number of rows in input data')
MODEL_FEATURE_COUNT = Gauge('model_input_feature_count', 'Number of features per sample')

# === Prometheus Endpoint ===
@app.route('/metrics', methods=['GET'])
def metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=1))
    RAM_USAGE.set(psutil.virtual_memory().percent)
    DISK_USAGE.set(psutil.disk_usage('/').percent)
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

# === Prediction Endpoint ===
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()
    THROUGHPUT.inc()

    data = request.get_json()
    try:
        df_data = data.get("data", []) or data.get("dataframe_split", {}).get("data", [])
        MODEL_INVOCATION_INPUT_SIZE.set(len(df_data))
        if df_data and isinstance(df_data[0], list):
            MODEL_FEATURE_COUNT.set(len(df_data[0]))

        api_url = "http://127.0.0.1:5005/invocations"
        response = requests.post(api_url, json=data)
        duration = time.time() - start_time

        REQUEST_LATENCY.observe(duration)
        PREDICTION_LATENCY.observe(duration)
        INFERENCE_SUCCESS.inc()

        predictions = response.json()

        if isinstance(predictions, list):
            for p in predictions:
                OUTPUT_CLASS_DISTRIBUTION.labels(class_label=str(p)).inc()

        return jsonify(predictions)

    except Exception as e:
        INFERENCE_FAIL.inc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
