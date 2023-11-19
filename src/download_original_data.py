import requests
from urllib.parse import urlencode
from io import BytesIO
from zipfile import ZipFile


UNZIP_PATH = './data/data'

if __name__ == '__main__':
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    # public_key = 'https://yadi.sk/d/UJ8VMK2Y6bJH7A'  # Сюда вписываете вашу ссылку
    public_key = 'https://disk.yandex.ru/d/IYiSpLob-zAxqg'


    # Получаем загрузочную ссылку
    final_url = base_url + urlencode(dict(public_key=public_key))
    response = requests.get(final_url)
    download_url = response.json()['href']

    # Загружаем файл
    download_response = requests.get(download_url)

    z = ZipFile(BytesIO(download_response.content))
    z.extractall(UNZIP_PATH)
