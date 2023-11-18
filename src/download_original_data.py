import requests
from urllib.parse import urlencode

base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
# public_key = 'https://yadi.sk/d/UJ8VMK2Y6bJH7A'  # Сюда вписываете вашу ссылку
public_key = 'https://disk.yandex.ru/d/IYiSpLob-zAxqg'


# Получаем загрузочную ссылку
final_url = base_url + urlencode(dict(public_key=public_key))
response = requests.get(final_url)
download_url = response.json()['href']

# Загружаем файл и сохраняем его
download_response = requests.get(download_url)
with open('downloaded_file.txt', 'wb') as f:   # Здесь укажите нужный путь к файлу
    f.write(download_response.content)