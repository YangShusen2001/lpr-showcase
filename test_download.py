#!/usr/bin/env python3
import urllib.request

urls = [
    'https://paddledet.bj.bcebos.com/pp_yoloe_s.onnx',
    'https://github.com/PaddlePaddle/PaddleDetection/releases/download/v2.7.0/pp_yoloe_s.onnx'
]

for i, url in enumerate(urls):
    print(f'Trying {url}...')
    try:
        urllib.request.urlretrieve(url, f'test_{i}.onnx')
        print('Success!')
        break
    except Exception as e:
        print(f'Failed: {e}')
