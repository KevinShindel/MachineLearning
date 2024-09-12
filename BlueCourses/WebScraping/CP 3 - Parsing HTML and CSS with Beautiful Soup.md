## Parsing HTML
- HTML is HyperText Markup Language which is used to create web pages.
- We need something that can make sense of the HTML and extract the information we need.
- Not: RegEx is not the right tool for parsing HTML.
- Convert HTML to structured DOM (Document Object Model) tree.
- Quick selection of elements using CSS selectors or XPath queries, DOM tree navigation.
- We need an HTML parsing library.
- We will use Beautiful Soup.
- Relatively easy to use and learn, though it has some gotchas
- Comes with some overhead, as it itself wraps around another XML\HTML tree pasrser
- Other libraries parse, pyquery, parsel, lxml, html5lib, etc.


## Using Beautiful Soup

```python
import requests
from bs4 import BeautifulSoup

url = 'http://www.bluecourses.com/'

r = requests.get(url)
html_content = r.text
soup = BeautifulSoup(html_content, 'html.parser')
print(soup.prettify())
```

- Two important methods:
- - find(name, attrs, recursive, string, **kwargs) - returns the first matching element
- - find_all(name, attrs, recursive, string, limit, **kwargs) - returns a list of all matching elements
- The **_name_** argument defines the tag name of the element we are looking for.
- The **_attrs_** argument is a dictionary of attributes and their values that we are looking for.
- The **_recursive_** argument is a boolean that defines if we want to search only the top-level elements or all the descendants.
- The **_string_** argument is used to search for strings in the text of the elements.
- The **_limit_** argument is used to limit the number of results ( for find_all() only).
- The **_keyword_** arguments are used to search for attributes that are not valid Python identifiers.
- The children attribute of a tag object returns a list of the tag's children.
- The descendants attribute of a tag object returns a generator that iterates over all the tag's descendants.
- The parent attribute of a tag object returns the tag's parent.
- next_sibling and previous_sibling attributes of a tag object return the next and previous siblings of the tag.
- Use text attribute to get the text of a tag ( use get_text(strip=True) it's simply than text.strip()).
- tag.find('div').find('table').find('thead').find('tr') the same as: tag.div.table.thead.tr
- tag.find_all('h1') the same as tag('h1')

## Beautiful Soup hands-on

### Scraping the BlueCourses web site
```python
import requests
from bs4 import BeautifulSoup as bs

url = 'http://www.bluecourses.com/'
r = requests.get(url)
html_content = r.text
soup = bs(html_content, 'html.parser')

course_info_elements = soup.find_all(class_='courses-listing-item')
print(len(course_info_elements))

# type of the elements: Tag
type(course_info_elements[0])

# Tag name of the element 
print(course_info_elements[0].name)

# A list containing the tag's children as a list 
print(course_info_elements[0].contents)

# Get the textual contents as clear text ( Note the diff between text and string)
    
print(course_info_elements[0].text)
print(course_info_elements[0].string)

# Iterate elements
for element in course_info_elements:
    course_name = element.find(class_='course-title').get_text(strip=True)
    course_desc = element.find(class_='course-description').get_text(strip=True)
    course_link = element.find('a').get('href')
    print(f'{course_name} - {course_desc} - {course_link}')
```
### Scraping Hacker News
```python
import requests
from bs4 import BeautifulSoup as bs
url = 'https://news.ycombinator.com/'

r = requests.get(url)
html_contents = r.text

html_soup = bs(html_contents, 'html.parser')

for post in html_soup.find_all('tr', class_='athing'):
    post_title_element = post.find('a', class_='storylink')
    post_title = post_title_element.get_text(strip=True)
    post_link = post_title_element.get('href')
    post_points = post.find_next(class_='score').get_text(strip=True)
    print(post_title, post_link, post_points)
    print()
```
### Dealing with JS in Beautiful Soup
- Inspect tool in browser shows the page as currently rendered:
- - Including dynamic changes by JS
- - Inpsecting element in browser shows the current state of the DOM
- Requests and BS cannot execute JS:
- - Static view, as the page came in
- - View source in browser

## Cascading Style Sheets
- Common HTML attributes which help to select elements:
- - ID: unique identifier
- - Class: group of elements
- - Tag name: all elements of a certain type
- Originaly HTML was meant to define both the structure and formatting of a website
- Web devs began to argue that the structure and formatting of documents basically relate to two diff concerns.
- CSS to govern how a document should be styled, HTML governs how it should be structured.
- In CSS, style info is written down as a list of colon-separated key-value pairs.

```css
h1 {
    color: red;
    background-color: yellow;
    font-size: 14pt;
    border: 2px solid yellow;
}
```
- The style declarations can be included in a document in three diff ways:
- - Inside a regular HTML attribute
- - Inside a style tag
- - In an external CSS file
- How to determine to which elements the styling should be applied?

- tagname - select all elements with a particular tag name
- .classname - select all elements with a particular class
- #id - select the element with a particular id
- selector1 selector2 - select all elements that are descendants of selector1 and match selector2
- selector1 > selector2 - select all elements that are children of selector1 and match selector2
- selector1 ~ selector2 - select all elements that are siblings of selector1 and match selector2
- selector1 + selector2 - select the element that is immediately after selector1 and matches selector2
- tagname[attribute] - select all elements with a particular attribute
- [attribute=value] - select all elements with a particular attribute value
- [attribute~=value] - select all elements with an attribute value containing a specific word
- [attribute|=value] - select all elements with an attribute value starting with a specific value
- [attribute^=value] - select all elements with an attribute value starting with a specific value
- [attribute$=value] - select all elements with an attribute value ending with a specific value
- [attribute*=value] - select all elements with an attribute value containing a specific value
- p:first-child - select the first child of a parent element
- p:not(selector) - select all elements that do not match a specific selector
- Using the elect method:
- - soup.select('a') - select all a elements
- - soup.select('#info') - select the element with id info
- - soup.select(div.classname) - select all div elements with the class classname
- The CSS selector rule engine in BeautifulSoup is not as powerful as the one found in a modern web browser.
- Some complex selectors might not work
- Use pyquery, parsel or Selenium for more complex CSS selectors

## More Beautiful Soup Examples 

```python
import requests
from bs4 import BeautifulSoup

page = 1
results = []

while True:
    print('Scraping page', page)
    p = requests.get('http://books.toscrape.com/catalogue/page-{}.html'.format(page))
    page += 1
    if p.status_code == 404:
        break
    soup = BeautifulSoup(p.text, 'html.parser')
    books = soup.select('.product_pod')
    for book in books:
        book_title = book.find('img').get('alt')
        book_link = book.find('a').get('href')
        book_rating = book.find(class_='star-rating').get('class')
        book_price = book.find(class_='price_color').get_text(strip=True)
        results.append({
            'book_title': book_title,
            'book_link': book_link,
            'book_rating': book_rating,
            'book_price': book_price
        })
    print(p.encoding, p.headers)
    
page = 1
results = []

ratings = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five']

while True:
    print('Scraping page', page)
    p = requests.get('http://books.toscrape.com/catalogue/page-{}.html'.format(page))
    p.encoding = 'UTF-8'
    page += 1
    if p.status_code == 404:
        break
    soup = BeautifulSoup(p.text, 'html.parser')
    books = soup.select('.product_pod')
    for book in books:
        book_title = book.find('img').get('alt')
        book_link = book.find('a').get('href')
        book_rating = ratings.index(book.find(class_='star-rating').get('class')[1])
        book_price = book.find(class_='price_color').get_text(strip=True)
        results.append({
            'book_title': book_title,
            'book_link': book_link,
            'book_rating': book_rating,
            'book_price': book_price
        })
        
# Scraping Zalando

import re
url = 'https://www.zalando.co.uk/womens-clothing-dresses/'
pages_to_crawl = 2
headers = {
    'User-Agent': 
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.70 Safari/537.36'
}

for p in range(1, pages_to_crawl+1):
    print('Scraping page:', p)
    r = requests.get(url, params={'p' : p}, headers=headers)
    html_soup = BeautifulSoup(r.text, 'html.parser')
    for article in html_soup.find_all('z-grid-item', class_=re.compile('^cat_card')):
        article_info = article.find(class_=re.compile('^cat_infoDetail'))
        if article_info is None:
            continue
        article_name = article.find(class_=re.compile('^cat_articleName')).get_text(strip=True)
        print(' -', article_name, article_info.get('href'))

```