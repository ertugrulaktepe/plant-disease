import requests

url = "http://localhost:5000/predict"



files = {'image': open('../images/images.jpg', 'rb')}

response = requests.post(url, files=files)

print(response.json())