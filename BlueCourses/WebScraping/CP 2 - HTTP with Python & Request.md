## What happens in a web browser
- HTTP - protocol used to communicate between web servers and clients
- HTML - Language used to create web pages (semi-structured data)
- CSS - Language used to style HTML (CSS style selectors can be used for web scraping)
- JavaScript - Language used to add interactivity to web pages

### Looking Up Domain Names (demo)
- DNS (Domain Name System) - translates domain names to IP addresses
- `nslookup` command to look up domain names
- `whois` command to look up domain registration information
- `traceroute` command to trace the route packets take to a destination
- `ping` command to test network connectivity

### ISO model (International Standards Organization)
- Layer 7: Application (HTTP, FTP, SMTP)
- Layer 6: Presentation (SSL, TLS)
- Layer 5: Session (NetBIOS, PPTP)
- Layer 4: Transport (TCP, UDP)
- Layer 3: Network (IP, ICMP)
- Layer 2: Data Link (Ethernet, Wi-Fi)
- Layer 1: Physical (Copper, Fiber, Radio)

## The HyperText Transfer Protocol
- HTTP request format
```http request
GET /index.html HTTP/1.1       # Request line
Host: www.example.com          # Header
Accept: text/html              # what type of data is accepted
Accept-Language: en-us         # what language is accepted
Accept-Encoding: gzip, deflate # what encoding is accepted
User-Agent: Mozilla/5.0        # what browser is being used
<blank line>
<message body>
```
- Each header includes a cese-insensitive field name and a value, followed by ":"
- Any whitespace before the value is ignored
- Browsers are very chatty in terms of what the like to include in theirs headers
- HTTP standard includes some headers which are standardized and will be utilized by proper we browses
- Duplicate headers are allowed and can be sent as a comma-separated list

HTTP versions:
- 0.9 - Not used anymore (no headers, no status codes, 1991)
- 1.0 - Improved version of 0.9 (1996)
- 1.1 - Current version (1997) TCP, persistent connections, pipelining, chunked transfer encoding, host header, status codes, cache control, etc.
- 2.0 - Latest version (2015) Binary protocol, multiplexing, server push
- 3.0 - In development ( switch to UDP from TCP)

## Talking HTTP with Python

See notes in `Python HTTP with Sockets.md`
```python
import socket

HOST = 'example.com'
PORT = 80

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(b'GET /index.html HTTP/1.1\r\n')
    s.sendall(b'Host: example.com\r\n')
    s.sendall(b'\r\n')
    data = s.recv(1024)
print('Received', repr(data))
```

## Python HTTP libraries
- Recall the main purpose of web scraping: to retrieve data from the web in automated manner
- urllib - built-in Python library for working with URLs
- httplib2 - comprehensive HTTP client library
- urllib3 - powerful, user-friendly HTTP client for Python
- requests - most popular Python library for HTTP requests
- grequests - Google's Requests library
- httpx - next generation HTTP client for Python

## Requests vs Urllib

### urllib
```python
import urllib.request
import urllib.parse

url = 'http://www.webscrapingfordatascience.com/postform2/'

formdata = {
    'name': 'Seppe',
    'gender': 'M',
    'pizza': 'like',
    'haircolor': 'brown',
    'comments': ''
}

data = urllib.parse.urlencode(formdata).encode('utf-8')
req = urllib.request.Request(url, data=data)
response = urllib.request.urlopen(req)
text = response.read()

print(text)
```

### requests
```python
import requests

url = 'http://www.webscrapingfordatascience.com/postform2/'

formdata = {
    'name': 'Seppe',
    'gender': 'M',
    'pizza': 'like',
    'haircolor': 'brown',
    'comments': ''
}

text = requests.post(url, data=formdata).text
print(text)
```

## Requests hands-on
```python

import requests
# Let's now perform a basic GET request to http://www.webscrapingfordatascience.com/basichttp/ -- you can open this web page in your browser. As you can see, it doesn't look 
# like much, but that's fine for now.

url = 'http://www.webscrapingfordatascience.com/basichttp/'
# The easiest way to perform a GET request is:

r = requests.get(url)
# .get and "GET" refer here to the fact that we send a request with a GET verb to the HTTP server. Except for the URL, we don't need to concern ourselves with headers and so on 
# (for now).

# Now let's take a look at this r object we have created:

r, str(r), repr(r), type(r)
(<Response [200]>,
 '<Response [200]>',
 '<Response [200]>',
 requests.models.Response)

# As you can see, this is a requests.models.Response object, and it does not provide any straightforward representation. We can, however, do lots of interesting things with it 
# using the following attributes:

# The HTTP reply status code and status message (reason)
print(r.status_code, r.reason)
200 OK
# The HTTP response headers
print(r.headers)
{'Date': 'Fri, 31 Jul 2020 09:03:11 GMT', 'Server': 'Apache/2.4.18 (Ubuntu)', 'Content-Length': '20', 'Keep-Alive': 'timeout=5, max=100', 'Connection': 'Keep-Alive', 'Content-Type': 'text/html; charset=UTF-8'}
# Note that these headers can be accessed as a dictionary, but they're actually a special type of dictionary:

type(r.headers)
# requests.structures.CaseInsensitiveDict
# Recall that headers in HTTP are case insensitive. Also recall that although multiple headers can share the same name, in which case Requests merges them under the same name, 
# separated by ,.

# The response body as text
print(r.text)

# The response body in raw binary form (helpful in case images or other binary content is retrieved)
print(r.content)
# Hello from the web!

b'Hello from the web!\n'
# E.g. for an image... (note that using .text here will lead to garbled uninterpretable output)
from IPython.display import Image
Image(
    requests.get('http://www.webscrapingfordatascience.com/files/kitten.jpg').content
)

# We can also take a look at the HTTP request, which is a requests.models.PreparedRequest object:

r.request, str(r.request), repr(r.request), type(r.request)
(<PreparedRequest [GET]>,
 '<PreparedRequest [GET]>',
 '<PreparedRequest [GET]>',
 requests.models.PreparedRequest)
# The headers in our request (also a CaseInsensitiveDict)
print(r.request.headers)
{'User-Agent': 'python-requests/2.24.0', 'Accept-Encoding': 'gzip, deflate', 'Accept': '*/*', 'Connection': 'keep-alive'}
# The request URL
r.request.url
'http://www.webscrapingfordatascience.com/basichttp/'
# The request method
r.request.method
'GET'
# The request body (None in this case)
r.request.body
# We've only focused on the GET verb so far. Later, we will take a look at POST and other verbs. Requests also comes with a general purpose method which can be used together 
# with any verb as follows:

requests.request('GET', url)
# We can also introduce URL parameters already. URL parameters are additional parameters added in an URL, following a question mark (?). Each parameter is separated by an 
# ampersand &. Try the following URL for instance: http://www.webscrapingfordatascience.com/paramhttp/?query=BlueCourses.

# This is actually pretty easy to use in Requests, as most of the time, you can just put them in the URL itself (just as shown by your browser)...

url = 'http://www.webscrapingfordatascience.com/paramhttp/?query=BlueCourses'
r = requests.get(url)
print(r.text)
print(r.request.url)
 # I don't have any information on "BlueCourses"
# http://www.webscrapingfordatascience.com/paramhttp/?query=BlueCourses
# The only time when this can become cumbersome is when the parameters contain special characters, i.e. characters that already have some meaning in URLs. This set of reserved 
# characters consists of ! *   '   (   )   ;   :   @   &   =   +   $   ,   /   ?   #   [   ]. A slash for instance is already used to separate paths in an URL.

# So what do we do in case we want to use one of those characters in a URL parameter's name or value? In some cases, the web server is able to parse our mistake correctly...

url = 'http://www.webscrapingfordatascience.com/paramhttp/?query=Blue/Courses'
r = requests.get(url)
print(r.text)
print(r.request.url)
# I don't have any information on "Blue/Courses"
# http://www.webscrapingfordatascience.com/paramhttp/?query=Blue/Courses
url = 'http://www.webscrapingfordatascience.com/paramhttp/?query=Blue?Courses'
r = requests.get(url)
print(r.text)
print(r.request.url)
# I don't have any information on "Blue?Courses"
# http://www.webscrapingfordatascience.com/paramhttp/?query=Blue?Courses
# But not always...

url = 'http://www.webscrapingfordatascience.com/paramhttp/?query=Blue+Courses'
r = requests.get(url)
print(r.text)
print(r.request.url)
# I don't have any information on "Blue Courses"
# http://www.webscrapingfordatascience.com/paramhttp/?query=Blue+Courses
url = 'http://www.webscrapingfordatascience.com/paramhttp/?query=Blue&Courses'
r = requests.get(url)
print(r.text)
print(r.request.url)
# I don't have any information on "Blue"
# http://www.webscrapingfordatascience.com/paramhttp/?query=Blue&Courses
# As such, the cleaner approach is to use the params argument instead:

url = 'http://www.webscrapingfordatascience.com/paramhttp/'
r = requests.get(url, params={'query': 'Blue+Courses'})
print(r.text)
print(r.request.url)
# I don't have any information on "Blue+Courses"
# http://www.webscrapingfordatascience.com/paramhttp/?query=Blue%2BCourses
url = 'http://www.webscrapingfordatascience.com/paramhttp/'
r = requests.get(url, params={'query': 'Blue&Courses'})
print(r.text)
print(r.request.url)
# I don't have any information on "Blue&Courses"
# http://www.webscrapingfordatascience.com/paramhttp/?query=Blue%26Courses
# One thing you might be wondering about is how params would handle parameter ordering or duplicate parameters...

url = 'http://www.webscrapingfordatascience.com/paramhttp/?query=Blue&query=Courses'
r = requests.get(url)
print(r.text)
print(r.request.url)
# I don't have any information on "Courses"
# http://www.webscrapingfordatascience.com/paramhttp/?query=Blue&query=Courses
url = 'http://www.webscrapingfordatascience.com/paramhttp/'
r = requests.get(url, params={'query': 'Blue', 'query': 'Courses'})
print(r.text)
print(r.request.url)
# I don't have any information on "Courses"
# http://www.webscrapingfordatascience.com/paramhttp/?query=Courses
url = 'http://www.webscrapingfordatascience.com/paramhttp/'
r = requests.get(url, params=[('query', 'Blue'), ('query', 'Courses')])
print(r.text)
print(r.request.url)
# I don't have any information on "Courses"
# http://www.webscrapingfordatascience.com/paramhttp/?query=Blue&query=Courses
# Let's now try taking a look at another website...

url = 'https://www.bluecourses.com'
r = requests.get(url)
print(r.text[:200])


# <!DOCTYPE html>
# <!--[if lte IE 9]><html class="ie ie9 lte9" lang="en"><![endif]-->
# <!--[if !IE]><!--><html lang="en"><!--<![endif]-->
# <head dir="ltr">
#     <meta charset="UTF-8">
#     <meta http
# So here we get back a web page formatted using HTML. We'll need a way to properly parse this.
```

## Using Requests
```python
import requests
url = 'http://www.webscrapingfordatascience.com/basichttp/'
r = requests.get(url)
print(r.status_code) # Which HTTPS status code did we get back?
print(r.reason) # What is the reason phrase?
print(r.headers) # What are the HTTP response headers?
print(r.request.headers) # What are the HTTP request headers?
print(r.text) # What is the content of the HTTP response?
```

## Getting the weather with Requests
```python
import requests

url = 'http://www.wttr.in'
headers = {
    'User-Agent': 'curl/7.68.0'
}

response = requests.get(url, headers=headers)
print(response.text[:200])
```