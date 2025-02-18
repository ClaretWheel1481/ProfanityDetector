from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 加载保存的模型和分词器
model_dir = "./model_output"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
class_names = ["normal", "profanity"]

app = Flask(__name__)

def predict_text(text, model, tokenizer, class_names, max_length=128):
    """
    对输入文本进行预测，并返回预测结果

    参数：
        text: 待预测的文本字符串
        model: 预训练的文本分类模型
        tokenizer: 对应的分词器
        class_names: 类别名称列表，如 ["normal", "profanity"]
        max_length: 分词后序列的最大长度
    """
    # 设置模型为评估模式
    model.eval()

    # 文本预处理：分词、填充、截断，并转换为PyTorch tensor
    inputs = tokenizer(text,
                       truncation=True,
                       padding='max_length',
                       max_length=max_length,
                       return_tensors='pt')

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

if __name__ == '__main__':
    # 启动 Flask 服务
    app.run(
        port=37883,
        debug=False,
        threaded=False,
        host='127.0.0.1'
    )