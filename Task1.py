import requests
from bs4 import BeautifulSoup

URL = "http://books.toscrape.com/"

try:
    response = requests.get(URL)
    response.raise_for_status()  
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    books = []
    for book in soup.select('article.product_pod'):
        title = book.h3.a['title']
        price = book.select_one('p.price_color').text
        books.append(f"{title} - {price}")
    
    print("📚 Top 5 Books:")
    for i, book in enumerate(books[:5], 1):
        print(f"{i}. {book}")

except Exception as e:
    print(f"Error: {str(e)}")
