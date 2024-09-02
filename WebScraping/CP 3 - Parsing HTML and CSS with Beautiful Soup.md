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
## Cascading Style Sheets
## CSS selectors in Beautiful Soup
## Further Beautiful Soup examples