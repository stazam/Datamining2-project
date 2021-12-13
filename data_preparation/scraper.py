from bs4 import BeautifulSoup 
import numpy as np 
import requests


articles = {'sciencetech': r'C:\Users\zamec\Datamining2-project\data\sciencetech_articles.npy',
            'sport': r'C:\Users\zamec\Datamining2-project\data\sport_articles.npy',
            'travel': r'C:\Users\zamec\Datamining2-project\data\travel_articles.npy'}


links_articles = {'sciencetech': r'C:\Users\zamec\Datamining2-project\data\sciencetech_links.npy',
            'sport': r'C:\Users\zamec\Datamining2-project\data\sport_links.npy',
            'travel': r'C:\Users\zamec\Datamining2-project\data\travel_links.npy'}

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

    try:
        old_links = list(np.load(links_articles[category],allow_pickle='TRUE'))
        all_links = list(set(links + old_links))
        np.save(links_articles[category], all_links)
        for link in old_links:
            if link in links:
                links.remove(link)

    except:
        print('Was a mistake')
        np.save(links_articles[category], links)    

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


def main():

    for cat, folder in articles.items():
        links = get_links(cat,110)
        num_articles = len(links)
        if num_articles > 0:
            try:
                old_texts = np.load(folder, allow_pickle='TRUE').item()
                texts = get_text(links)
                texts.update(old_texts)
                print('skuska')
            except:
                texts = get_text(links)

            np.save(folder, texts) 
            print('%d articles for %s category saved' % (num_articles,cat))
        else:
            print('No new article on the page for %s category' % (cat))

if __name__ == "__main__":
    main()