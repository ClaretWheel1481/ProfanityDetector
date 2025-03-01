from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from consul import Consul
import time
import threading

# 加载保存的模型和分词器
model_dir = "./model_output"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
class_names = ["normal", "profanity"]

app = Flask(__name__)

def predict_text(text, model, tokenizer, class_names, max_length=128):
    """
    对输入文本进行预测，并返回预测结果
    """
    model.eval()
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()

    return class_names[predicted_class_id]

@app.route('/api/v1/text/predict', methods=['POST'])
def predict():
    """
    接收 POST 请求，要求 JSON 数据中包含 "text" 字段，
    返回预测结果
    """
    data = request.get_json(force=True)
    if not data or 'text' not in data:
        return jsonify({"error": "请求 JSON 中缺少 'text' 字段"}), 400

    text = data['text']
    prediction = predict_text(text, model, tokenizer, class_names)
    return jsonify({
        "text": text,
        "prediction": prediction
    })

@app.route('/api/v1/text/predict', methods=['GET'])
def ping():
    return jsonify({'status': 'ok'})

def register_service_with_consul():
    """
    注册 Flask 服务到 Consul
    """
    consul = Consul()

    service_id = "ProfanityDetector-1"
    service_name = "ProfanityDetector"
    service_port = 37886
    service_address = "127.0.0.1"
    health_check_url = f"http://{service_address}:{service_port}/api/v1/text/predict"

    consul.agent.service.register(
        service_name,
        service_id=service_id,
        port=service_port,
        address=service_address,
        tags=["flask", "predictor"],
        check={
            "http": health_check_url,
            "interval": "20s",
            "timeout": "5s"
        }
    )
    print(f"Service {service_name} registered with Consul")

    try:
        while True:
            time.sleep(100)
    except KeyboardInterrupt:
        consul.agent.service.deregister(service_id)
        print(f"Service {service_name} deregistered from Consul")

if __name__ == '__main__':
    # 若不需要consul服务，则注释掉以下代码
    consul_thread = threading.Thread(target=register_service_with_consul)
    consul_thread.daemon = True
    consul_thread.start()
    ##########################################
    app.run(
        port=37886,
        debug=False,
        threaded=False,
        host='127.0.0.1'
    )
