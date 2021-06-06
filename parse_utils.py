import requests
from bs4 import BeautifulSoup as bs
def get_response_soup(url):
  response = requests.get(url)
  return bs(response.text, 'html')

accord_list = ['C', 'F', 'Gm', 'Dm', 'Bb', 'B', 'E', 'Em', 'Eb', 'Fm', 'Fb', 'Am', 'G', 'D', 'A', 'Bm', 'G6sus2', '3x223x', 'F#m']
def if_accords(text):
  if '|' in text or '*' in text:
    return True
  list_tmp = text.split(' ')
  for word in list_tmp:
    if word in accord_list:
      return True
  return False

def prepare_song(soup):
  tmp_string = soup.find('div',class_='js-store')['data-content'].split('content":"')[1].split("revision")[0]
  tmp_string = tmp_string.replace('\\r', '').replace('\t', '').replace('[tab]', '').replace('[/tab]', '').replace('[ch]', '').replace('[/ch]', '').split('\\n')
  filtered_data = []
  for tmp in tmp_string:
    if len(tmp) > 0 and tmp[0] != '[' and tmp != ' ':
      filtered_data.append(tmp)
  chords = []
  texts = []
  for tmp in filtered_data:
    if if_accords(tmp):
      chords.append(tmp)
    else:
      texts.append(tmp)
  return {'chords':chords, 'texts':texts, 'all':filtered_data}
