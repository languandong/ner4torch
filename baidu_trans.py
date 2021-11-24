# -*- coding: utf-8 -*-

# This code shows an example of text translation from English to Simplified-Chinese.
# This code runs on Python 2.7.x and Python 3.x.
# You may install `requests` to run this code: pip install requests
# Please refer to `https://api.fanyi.baidu.com/doc/21` for complete api document
import pandas as pd
import requests
import random
import json
from hashlib import md5

# Set your own appid/appkey.
appid = '20210415000781369'
appkey = 'gQEO9V2dfAHj6ABsFb0p'

# For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
from_lang = 'zh'
to_lang =  'en'

from_lang1 = 'en'
to_lang1 = 'zh'

endpoint = 'http://api.fanyi.baidu.com'
path = '/api/trans/vip/translate'
url = endpoint + path
# Generate salt and sign
def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()

data_src = pd.read_csv('./data/train_data_public.csv')
text_dst = []

for i in range(data_src.shape[0]):
    # 中译英
    query = data_src.loc[i,'text']

    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)

    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()
    # 英译中
    en_text = eval(json.dumps(result, indent=4, ensure_ascii=False))['trans_result'][0]['dst']
    print(en_text)

    salt = random.randint(32768, 65536)
    sign = make_md5(appid + en_text + str(salt) + appkey)
    payload1 = {'appid': appid, 'q': en_text, 'from': from_lang1, 'to': to_lang1, 'salt': salt, 'sign': sign}
    r = requests.post(url, params=payload1, headers=headers)
    result = r.json()
    print(json.dumps(result, indent=4, ensure_ascii=False))
    result_text = eval(json.dumps(result, indent=4, ensure_ascii=False))['trans_result'][0]['dst']

    # Show response
    print(query)
    print(result_text)

    break
