import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict_text(text, model, tokenizer, class_names, max_length=128):
    """
    对输入文本进行预测，并打印预测结果

    参数：
        text: 待预测的文本字符串
        model: 预训练的文本分类模型
        tokenizer: 对应的分词器
        class_names: 类别名称列表，如 ["类别0", "类别1"]
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

    # 推理得到预测结果
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()

    print(f'预测结果: {class_names[predicted_class_id]}')


# 加载保存的模型和分词器
model_dir = "./model_output"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

class_names = ["normal", "profanity"]

sample_text = ""

predict_text(sample_text, model, tokenizer, class_names)
