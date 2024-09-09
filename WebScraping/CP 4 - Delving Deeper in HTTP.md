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
- - Some websites will check this to prevent 'deep links' or 'hot-linking' from working.
- Just as we've seen at various occasions, the server can send back headers as well, which can be accessed through the 'headers' attribute of the response object.
- Just like the data and params arguments, headers can accept an OrderedDict object in case the ordering of the headers is important
- What is allowed is to provide multiple values for the same header by separating them with a comma, as in the line Accept-Encoding: gzip, deflate
- Note that response headers can contain multiple lines with the same name 

## Cookies
- HTTP is a simple networking protocol
- - Text based on follows a simple request-and-reply based communication scheme
- - In the simplest case, every request-reply cycle in HTTP involves setting up a fresh new underlying network connection as well, though the v1.1 of the HTTP standard allows to 
    set up 'keep alive' connections.
- The simple request-reply based approach poses some problems for websites:
- - From a web server's point of view, every incoming request is completely independent of any previous ones and can be handled on its own
- - As online shop where items can be added to a cart. When visiting the checkout page, we expect the web server to 'remember' the items we selected and added previously.
- How to add a state mechanism to HTTP?
- - We could include a special identifier as a URL parameter that 'links' multiple visits to the same user, 'checkout.html?visitor=20495', but this is easy to leak. Also, what 
    if browser is reopened?
- - For POST requests, we could either use the same URL parameter, or include the 'session' ID in hidden form field.
- - Some older websites do actually use such mechanisms, but they are not very secure.
- Better solution: Cookies
- - Simple header mechanism: server sends cookies the browser should resend with every subequent request to that domain
- - Set-cookie header example
```http request
HTTP/1.1 200 OK
Content-type: text/html
Set-Cookie: sessionid=20495; expires=Wed, 09 Jun 2021 10:18:14 GMT
Set-Cookie: siteTheme=dark; expires=Wed, 09 Jun 2021 10:18:14 GMT
```

- - Or all-in-one line and comma-separated (though less common given that comma can appear in cookie line)
- - Some servers user set-cookie in lowercase - allowed for header names.
- Every cookie is well-defined:
- - A name and value, separated by an equal sign '='
- - An additional attributes, ';'-separated
- At every sub-sequent request, browser checks and sends its cookies (here semicolon separated)
- - Cookie header
```http request
GET /checkout.html HTTP/1.1
Host: www.example.com
Cookie: sessionid=20495; siteTheme=dark
```
- Note: advertisers have come up with many diff ways to perform fingerprinting:
- - JSON Web Tokens (JWT) - a way to encode data in a URL-safe way, ETag headers, web storage, Flash and many more.
- - See 'evercookie' for an example of a cookie that is very hard to delete.
- - Luckily, oftentimes not necessary to 'emulate' all this for web scraping.
- - Mainly used for trackiny g and advertising purposes.

### Cookies and Sessions in Requests 
```python
# In this notebook, we'll explore some examples where we will learn about the concept of cookies in HTTP.

import requests
from bs4 import BeautifulSoup
# Let us first start with a simple example. Navigate to http://www.webscrapingfordatascience.com/cookielogin/. This page asks you to login (any username and password is fine), after which you can access a "secret" page, http://www.webscrapingfordatascience.com/cookielogin/secret.php.

# Let's try whether we can access this secret page directly using Requests:

url = 'http://www.webscrapingfordatascience.com/cookielogin/secret.php'

# Without cookies
r = requests.get(url)
print(r.text)
# Hmm... it seems you are not logged in
# That didn't work. So let's try to attempt to login ourselves using Requests first with a POST request as seen before:

url = 'http://www.webscrapingfordatascience.com/cookielogin/'

r = requests.post(url, data={'username': 'user', 'password': 'pass'})

print(r.text)

# Let's now try to access the secret page:

url = 'http://www.webscrapingfordatascience.com/cookielogin/secret.php'

r = requests.get(url)
print(r.text) # Hmm... it seems you are not logged in
# What happened here? After submitting the login data, the server includes Set-Cookie headers in its HTTP reply, which it will use to identify us in subsequent requests. So we have to do the same here...

url = 'http://www.webscrapingfordatascience.com/cookielogin/'

r = requests.post(url, data={'username': 'user', 'password': 'pass'})

print(r.headers)
{'Date': 'Fri, 31 Jul 2020 10:18:09 GMT', 'Server': 'Apache/2.4.18 (Ubuntu)', 'Set-Cookie': 'PHPSESSID=iuh0i1jq783t1ried5i6m0bo85; path=/', 'Expires': 'Thu, 19 Nov 1981 08:52:00 GMT', 'Cache-Control': 'no-store, no-cache, must-revalidate', 'Pragma': 'no-cache', 'Vary': 'Accept-Encoding', 'Content-Encoding': 'gzip', 'Content-Length': '114', 'Keep-Alive': 'timeout=5, max=100', 'Connection': 'Keep-Alive', 'Content-Type': 'text/html; charset=UTF-8'}
r.cookies # <RequestsCookieJar[Cookie(version=0, name='PHPSESSID', value='iuh0i1jq783t1ried5i6m0bo85', port=None, port_specified=False, domain='www.webscrapingfordatascience.
# com', domain_specified=False, domain_initial_dot=False, path='/', path_specified=True, secure=False, expires=None, discard=True, comment=None, comment_url=None, rest={}, rfc2109=False)]>

url = 'http://www.webscrapingfordatascience.com/cookielogin/secret.php'

r = requests.get(url, cookies=r.cookies)
print(r.text) # This is a secret code: 1234
# Let us now take a look at a trickier example, using http://www.webscrapingfordatascience.com/redirlogin/.

# This page behaves very similar to the example before, so let's try the same approach.

url = 'http://www.webscrapingfordatascience.com/redirlogin/'

r = requests.post(url, data={'username': 'user', 'password': 'pass'})

print(r.headers)
r.cookies  # {'Date': 'Fri, 31 Jul 2020 10:19:23 GMT', 'Server': 'Apache/2.4.18 (Ubuntu)', 'Expires': 'Thu, 19 Nov 1981 08:52:00 GMT', 'Cache-Control': 'no-store, no-cache, 
            # must-revalidate', 'Pragma': 'no-cache', 'Content-Length': '27', 'Keep-Alive': 'timeout=5, max=99', 'Connection': 'Keep-Alive', 'Content-Type': 'text/html; charset=UTF-8'}
# Strange, there are no cookies here. If we inspect the requests using our web browser, we see why: the status code of the HTTP reply here is a redirect (302), which Requests 
# follows by default. Since it is the intermediate page which provides the Set-Cookie header, this gets overridden by the final destination reply headers.

# Luckily, we can override this behavior using allow_redirects.

url = 'http://www.webscrapingfordatascience.com/redirlogin/'

r = requests.post(url, data={'username': 'user', 'password': 'pass'}, allow_redirects=False)

print(r.status_code) # 302
print(r.headers) #  {'Date': 'Fri, 31 Jul 2020 10:21:42 GMT', 'Server': 'Apache/2.4.18 (Ubuntu)', 'Set-Cookie': 'PHPSESSID=vsm1065k77t1qjepijmnr9rjg6; path=/', 'Expires': 'Thu, 
# 19 Nov 1981 08:52:00 GMT', 'Cache-Control': 'no-store, no-cache, must-revalidate', 'Pragma': 'no-cache', 'Location': 'http://www.webscrapingfordatascience.com/redirlogin/secret.php', 'Content-Length': '114', 'Keep-Alive': 'timeout=5, max=100', 'Connection': 'Keep-Alive', 'Content-Type': 'text/html; charset=UTF-8'}

url = 'http://www.webscrapingfordatascience.com/redirlogin/secret.php'

r = requests.get(url, cookies=r.cookies)
print(r.text) # This is a secret code: 1234

# Luckily, Requests provides a handy mechanism which avoids us having to manage cookies ourselves: sessions. By creating a session and using it as the requests module so far, Requests will make sure to keep track of cookies and send them with each subsequent request.

my_session = requests.Session()
# In addition, sessions make it easy to change some headers you want to apply for all requests you make using this session. This helps to avoid having to provide a headers 
# argument for every request, which is particularly helpful for e.g. the User-Agent header (you can still use the headers argument as well to set request-specific headers).

my_session.headers.update({'User-Agent': 'Chrome!'})
# Let's now try out our session on the URL http://www.webscrapingfordatascience.com/trickylogin/. Inspect your browser to see what is happening there.

# Note that we do perform an explicit GET request below to get out the login form first. What happens if you don't use this? Can you figure out using the developer tools what is happening here?

url = 'http://www.webscrapingfordatascience.com/trickylogin/'

# Perform a GET request
r = my_session.get(url)
print(r.request.headers)
print(r.headers)
print()

# Login using a POST
r = my_session.post(url, params={'p': 'login'}, data={'username': 'dummy', 'password': '1234'}) 
print(r.request.headers)
print(r.headers)
print()

# Get the protected page (note that in this example, a URL parameter is used as well)
r = my_session.get(url, params={'p': 'protected'})
print(r.request.headers)
print(r.headers)

{'User-Agent': 'Chrome!', 'Accept-Encoding': 'gzip, deflate', 'Accept': '*/*', 'Connection': 'keep-alive'}
{'Date': 'Fri, 31 Jul 2020 11:11:43 GMT', 'Server': 'Apache/2.4.18 (Ubuntu)', 'Set-Cookie': 'PHPSESSID=tp07889unftqnaabp0jl8bkvt3; path=/, PHPSESSID=op9ldo1bjmc1j02noh7ljrkv13; path=/', 'Expires': 'Thu, 19 Nov 1981 08:52:00 GMT', 'Cache-Control': 'no-store, no-cache, must-revalidate', 'Pragma': 'no-cache', 'Vary': 'Accept-Encoding', 'Content-Encoding': 'gzip', 'Content-Length': '167', 'Keep-Alive': 'timeout=5, max=100', 'Connection': 'Keep-Alive', 'Content-Type': 'text/html; charset=UTF-8'}

{'User-Agent': 'Chrome!', 'Accept-Encoding': 'gzip, deflate', 'Accept': '*/*', 'Connection': 'keep-alive', 'Cookie': 'PHPSESSID=ha1vbffei4bchg2e5i5k3totk3'}
{'Date': 'Fri, 31 Jul 2020 11:11:43 GMT', 'Server': 'Apache/2.4.18 (Ubuntu)', 'Expires': 'Thu, 19 Nov 1981 08:52:00 GMT', 'Cache-Control': 'no-store, no-cache, must-revalidate', 'Pragma': 'no-cache', 'Set-Cookie': 'PHPSESSID=tk2ph9bndp4nffaisvee1b30n5; path=/', 'Content-Length': '31', 'Keep-Alive': 'timeout=5, max=98', 'Connection': 'Keep-Alive', 'Content-Type': 'text/html; charset=UTF-8'}

{'User-Agent': 'Chrome!', 'Accept-Encoding': 'gzip, deflate', 'Accept': '*/*', 'Connection': 'keep-alive', 'Cookie': 'PHPSESSID=tk2ph9bndp4nffaisvee1b30n5'}
{'Date': 'Fri, 31 Jul 2020 11:11:43 GMT', 'Server': 'Apache/2.4.18 (Ubuntu)', 'Expires': 'Thu, 19 Nov 1981 08:52:00 GMT', 'Cache-Control': 'no-store, no-cache, must-revalidate', 'Pragma': 'no-cache', 'Set-Cookie': 'PHPSESSID=9eecg4bl00onra516qb1sdi4c4; path=/', 'Content-Length': '31', 'Keep-Alive': 'timeout=5, max=97', 'Connection': 'Keep-Alive', 'Content-Type': 'text/html; charset=UTF-8'}

print(r.text)
# Here is your secret code: 3838.
# Cookies can also be managed yourself through the session cookiejar, which behaves like a normal Python dictionary.

my_session.cookies # <RequestsCookieJar[Cookie(version=0, name='PHPSESSID', value='9eecg4bl00onra516qb1sdi4c4', port=None, port_specified=False, domain='www.webscrapingfordatascience.com', domain_specified=False, domain_initial_dot=False, path='/', path_specified=True, secure=False, expires=None, discard=True, comment=None, comment_url=None, rest={}, rfc2109=False)]>
my_session.cookies.get('PHPSESSID') # '9eecg4bl00onra516qb1sdi4c4'
my_session.cookies.clear()
my_session.cookies # <RequestsCookieJar[]>
```

## Other Content

- Binary files example (images, PDFs, etc.)
```python
import requests

url = 'http://www.webscrapingfordatascience.com/files/kitten.jpg'

r = requests.get(url)

with open('kitten.jpg', 'wb') as f:
    f.write(r.content)
```

- Binary files streaming

```python
import requests

url = 'http://www.webscrapingfordatascience.com/files/kitten.jpg'

r = requests.get(url, stream=True)

with open('kitten_stream.jpg', 'wb') as f:
    for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)
```

- JSON data
```python
import requests

url = 'http://www.webscrapingfordatascience.com/jsonajax/results.php'   

r = requests.post(url, data={'api_code': 'C123456'})

print(r.request.headers) # {'User-Agent': 'python-requests/2.24.0', 'Accept-Encoding': 'gzip, deflate', 'Accept': '*/*', 'Connection': 'keep-alive', 'Content-Length': '13', 'Content-Type': 'application/x-www-form-urlencoded'}
print(r.json()) # {'results': 'success'}
print(r.json()['results'])
```
- Even if the website you wish to srape doesn't provide a JSON API, you can still use JSON data in your scraping workflow, it's always reccomended to keep an eye on your 
  browser's dev tools networking information to see if you can spot an JS-driven request to URL endpoints which return nicely structured JSON data.
- Even although an API might not be documented, fetching the info directly from such 'internal APIs' is always a clever idea, as this will avoid having to deal with the HTML soup.

### JSON Content with Requests 

```python
# In this notebook, we illustrate how you can work with JSON APIs using Requests. Here, we will get the list of post on the world news subreddit,
# https://www.reddit.com/r/worldnews/. Reddit provides a handy JSON API simply by appending .json to the URL.

import requests
r = requests.get('https://www.reddit.com/r/worldnews/.json')
r.text
'{"message": "Too Many Requests", "error": 429}'
# Reddit doesn't like the Requests user-agent, however...

r = requests.get('https://www.reddit.com/r/worldnews/.json', headers={
    'User-Agent': 
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36'
})
r.text[:200]
'{"kind": "Listing", "data": {"modhash": "", "dist": 25, "children": [{"kind": "t3", "data": {"approved_at_utc": null, "subreddit": "worldnews", "selftext": "", "author_fullname": "t2_612zd", "saved": '
# We could parse this manually, but Requests allows to simply do the following:

j = r.json()
for post in j['data']['children']:
    print(post['data'].get('title'))
```