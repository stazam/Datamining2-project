from scraper import *

folders = list(articles.values())

read_clanky = np.load(folders[0],allow_pickle='TRUE').item()
print(read_clanky)