## Forms and POST Data
- In the case of a 'GET' method, the form data kust get submitted as a part of the GET request as URL parameters.
- - Easy to handle with Requests
- - Common for search boxes, for instance.
- When a 'POST' is used, the browser will perform an HTTP POST request
- - Used when a request is not idempotent; the browser will warn you try to refresh a POST
- The 'action' argument of the form, determines to which URL the request is submitted to:
- - If left blank, the same URL as the current page will be used
- Input elements themselves are specified using 'input', 'select' tags
- - Name-value pairs will be submitted as form data.

- Example of POST request:
```python
import requests

url = 'http://www.webscrapingfordatascience.com/postform2/'

r = requests.post(url, data={'name': 'Seppe', 'gender': 'm', 'pizza': 'like', 'haircolor': 'brown', 'comments': ''})
print(r.text)
```

- It's here where we see some further 'picky' aspects of web servers:
- - Sometimes servers check duplicate keys
- - The submit button in a web form can have a name and included in parameters or POST data
- - The ordering of parameters or POST data fields can matter
- - Sometimes, servers do not care whether data is in the URL parameter or POST data, and sometimes they do.

- POST Requests with Requests 
```python
import requests

url = 'http://www.webscrapingfordatascience.com/postform2/'

# First perform a normal GET request (we don't have to, but we can do so to take a look at the form)
r = requests.get(url)

print(r.text)

# Next, we submit the form
formdata = {
    'name': 'Seppe',
    'gender': 'M',
    'pizza': 'like',
    'haircolor': 'brown',
    'comments': ''
}

r = requests.post(url, data=formdata)
print(r.text)

# Quotes to Scrape

We first need to import Beautiful Soup as well.

from bs4 import BeautifulSoup
url = 'http://quotes.toscrape.com/search.aspx'
soup = BeautifulSoup(requests.get(url).text, 'html.parser')
Let's first get the list of authors.

authors = [element.get('value') for element in soup.find(id='author').find_all('option') if element.get('value')]
authors[:3]

# Just selecting the tag drop down doesn't work. So we need to figure out what happens if we select a particular author:

soup.find(id='tag').find_all('option')

filter_url = 'http://quotes.toscrape.com/filter.aspx'
r = requests.post(filter_url, data={
    'author': 'Albert Einstein'
})

r.status_code # 500

def get_author_tags(author):
    # First request the search page
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    # Get out the viewstate
    viewstate = soup.find(id='__VIEWSTATE').get('value')
    # Now perform the post
    soup = BeautifulSoup(requests.post(filter_url, data={
        'author': author,
        'tag': '----------',
        '__VIEWSTATE': viewstate
    }).text, 'html.parser')
    # And get out the list of tags
    return [element.get('value') for element in soup.find(id='tag').find_all('option') if element.get('value')]
get_author_tags('Albert Einstein')

# This works, but having to perform the GET request to the main page every time is annoying, and won't always work (i.e. sites will not always have an option to go back to an initial state). As such, the following is even better:

def get_author_tags(author, viewstate=None):
    # If the viewstate is None, get out the first one
    if not viewstate:
        soup = BeautifulSoup(requests.get(url).text, 'html.parser')
        viewstate = soup.find(id='__VIEWSTATE').get('value')
    soup = BeautifulSoup(requests.post(filter_url, data={
        'author': author,
        'tag': '----------',
        '__VIEWSTATE': viewstate
    }).text, 'html.parser')
    viewstate = soup.find(id='__VIEWSTATE').get('value')
    # Return the tags and viewstate for the next request
    return [element.get('value') for element in soup.find(id='tag').find_all('option') if element.get('value')], \
            viewstate
tags, viewstate = get_author_tags('Albert Einstein')

tags, viewstate = get_author_tags('Jane Austen', viewstate)
print(tags)

```

## Other HTTP Methods

- GET requests a representation of the specified URL
- POST method indicates that data is being submitted as part of a request to a particular URL
- GET and POST are most used , though REST APIs might require other methods
- HEAD requests a response just like the GET request does but indicates to the web server that id does not need to send the response body.
- PUT - requests that the submitted data should be stored under the supplied request URL, thereby creating
- DELEte requests that the data listed under the request URL should be removed
- CONNECT, OPTIONS, TRACE and PATCH are less commonly encountered request methods.

## HTTP Headers
```python
import requests

url = 'http://www.webscrapingfordatascience.com/usercheck/'
r = requests.get(url)

print(r.text) # shows : it seems you are using a scraper or something similar

print(r.request.headers) # Headers of the request

# {'User-Agent': 'python-requests/2.24.0', 'Accept-Encoding': 'gzip, deflate', 'Accept': '*/*', 'Connection': 'keep-alive'}

```
- APart from User-Agent header, there is another header that deserves special mention: The Referer header:
- - Browser will include this header to indicate the URL of the web page that linked to the URL being requested
- - Some websites will check this to prevent 'deep links' or 'hotlinking' from working.
- Just as we've seen at various occasions, the server can send back headers as well, which can be accessed through the 'headers' attribute of the response object.
- Just like the data and params arguments, headers can accept an OrdereDict object in case the ordering of the headers is important
- What is allowed is to provide multiple values for the same header by separating them with a comma, as in the line Accept-Encoding: gzip, deflate
- Note that response headers can contain multiple lines with the same name 

## Cookies