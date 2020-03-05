import requests

def download_file():
    url = 'https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed'
    r = requests.get(url, allow_redirects=True)
    open('./univeral-sentence-encoder-v3.tar.gz', 'wb').write(r.content)


if __name__ == '__main__':
    download_file()