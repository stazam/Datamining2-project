from bs4 import BeautifulSoup 
import numpy as np 
import requests


def get_links(category:str, limit:int) -> list:

    url = 'https://www.dailymail.co.uk/' + category + '/index.html'
    prefix = url[:27]

    page = requests.get(url) 
    soup = BeautifulSoup(page.content, 'html.parser') 

    links = []
    for i, url in enumerate(soup.find_all('a',itemprop="url")):
        if i > limit:
            break
    
        links.append(prefix + url['href'])

    return links


def get_text(links: list) -> dict:

    dict_text = {}
    for link in links:

        page1 = requests.get(link) 
        soup1 = BeautifulSoup(page1.content, 'html.parser') 
        
        header = soup1.find('h2').text

        article_text = []
        for p in soup1.find_all('p', class_ = 'mol-para-with-font'):
            try: 
                article_text.append(p.find('span').text.replace("\xa0", " "))
            except:
                article_text.append(p.text.replace("\xa0", " "))

        dict_text[header] = article_text

    return dict_text


articles = {'sciencetech': r'C:\Users\zamec\Datamining2-project\data\sciencetech_articles.npy',
            'sport': r'C:\Users\zamec\Datamining2-project\data\sport_articles.npy',
            'travel': r'C:\Users\zamec\Datamining2-project\data\travel_articles.npy'}


def main():

    for cat, folder in articles.items():
        links = get_links(cat,100)
        texts = get_text(links)
        np.save(folder, texts) 
        print('Articles for %s category saved' % (cat))


if __name__ == "__main__":
    main()