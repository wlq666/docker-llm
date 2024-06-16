import requests
import json

check_api = ['docs','health-check']
url = 'http://127.0.0.1:5001/predictions'
payload = json.dumps({"input":{
    "inputs": "Who are you? what's your name? How old are you?",
    "do_sample": True,
    "top_k": 40,
    "top_p": 0.9
}})
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json'
}
response = requests.request("POST", url, headers=headers, data=payload)

if response.status_code == 200:
    print('String sent successfully!')
    result = response.text
    content = result
    print('Feedback result:', content)
else:
    print('Error:', response.status_code)
    print('Reason:', response.reason)
    print('Content:', response.content)

for api in check_api:
    url = 'http://127.0.0.1:5001/'+api
    res = requests.request("GET", url, headers=headers)
    if res.status_code == 200:
        print('String sent successfully!')
        result = res.text
        content = result
        print('Feedback result:', content)
    else:
        print('Error:', res.status_code)
        print('Reason:', res.reason)
        print('Content:', res.content)