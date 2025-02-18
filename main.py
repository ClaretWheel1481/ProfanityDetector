from flask import Flask

app = Flask(__name__)

if __name__ == '__main__':
    # 启动 Flask 服务
    app.run(
        port=37882,
        debug=False,
        threaded=False,
        host='127.0.0.1'
    )