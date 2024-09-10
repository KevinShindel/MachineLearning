## Other Python libraries and tools
- Request libraries
- - urllib - can deal with all things HTTP (included in Python standard library)
- - httplib2 - comprehensive HTTP client library
- - urllib3 - powerful, user-friendly HTTP client for Python
- - requests - HTTP library for Python
- - grequests - a Python library for making HTTP requests
- - httpx - a fully featured HTTP client for Python 3
- - aiohttp - an asynchronous HTTP client for Python
- Parsing libraries
- - parse - a Python library for parsing strings using a specification based on the Python format() syntax
- - pyquery - a jQuery-like library for Python
- - parsel - a library for extracting data from HTML and XML using XPath and CSS selectors
- Scraper tools
- - MechanicalSoup - a Python library for automating interaction with websites
- - Scrapy - a fast, high-level web crawling and web scraping framework
- - BeautifulSoup - a Python library for pulling data out of HTML and XML files
- Caching libraries
- - CacheControl - a Python library for managing cache headers
- - requests-cache - a transparent persistent cache for requests
- Smart retries 
```python
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def requests_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504), session=None):
    session = session or requests.Session()
    retry = Retry(total=retries, read=retries, connect=retries, backoff_factor=backoff_factor, status_forcelist=status_forcelist)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session
```

```python
def retry(exceptions=Exception, tries=-1, delay=0, max_delay=None, backoff=1, jitter=0, logger=logging_loger):
""" Return a retry decorator.
:param exceptions: an exception or a tuple of exceptions to catch. Default: Exception.
:param tries: the maximum number of attempts. Default: -1 (infinite).
:param delay: initial delay between attempts. Default: 0.
:param max_delay: the maximum value of delay. Default: None (no limit).
:param backoff: multiplier applied to delay between attempts. Default: 1 (no backoff).
:param jitter: extra seconds added to delay between attempts. Default: 0.
:param logger: logger to use. Default: logging.getLogger('retrying').
"""
def wrap_function(func):
    @wraps(func)
    def wrapped_function(*args, **kwargs):
        mtries, mdelay = tries, delay
        while mtries:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                mtries -= 1
                if mtries:
                    sleep(mdelay + uniform(0, jitter))
                    mdelay *= backoff
                    if max_delay:
                        mdelay = min(mdelay, max_delay)
                else:
                    logger.exception(e)
                    raise
    return wrapped_function
```



## Other programming languages
- R - rvest, RSelenium
- C# and Java - HtmlUnit, Selenium
- JavaScript - PhantomJS, NightmareJS, Puppeteer
- IE as a browser driver is not recommended

## Command line tools
- Nice as helpers
- - httpie - a user-friendly cURL replacement
- - curl - a command-line tool for transferring data with URL syntax
- - wget - a command-line utility for downloading files from the web

## News articles
- Basically boils down to getting out the main content of a news article
- - Much trickier as it might seem as first sight
- - You might try to iterate the lowest level of the DOM tree and get the text from there
- - Considering all elements does not resolve this issue, as you'll end up simple selecting the top element on the page, as this will always contain the largest amount text
- - The same holds in case you'd rely on the rect attribute Selenium provides to apply a visual approach.
- A large number of libraries and tools have been written in Python to help with web scraping
- - https://github.com/masukomi/ar90-readability
- - https://github.com/misja/python-boilerpipe
- - https://github.com/codelucas/newspaper
- - https://github.com/fhamborg/news-please
- - https://newsapi.org/ or https://webhose.io/ might be a good alternative
- - https://github.com/mozilla/readability
- - Alternative: RSS a web feed which allows users to access updates to online content in a standardized, computer-readable format

```python
from newsplease import NewsPlease

article = NewsPlease.from_url('https://www.bbc.com/news/world-europe-58486391')

print(article.title)
```

- Dragnet (https://github.com/dragnet-org/dragnet) is a Python library for extracting main article content from web pages
```python
import requests
from dragnet import extract_content, extract_content_and_comments

# fetch HTML
url = 'https://moz.com/devblog/dragnet-content-extraction-from-diverse-feature-rich-webpages'

r = requests.get(url)

# get main article without comments
content = extract_content(r.text)

# get article and comments
content_comments = extract_content_and_comments(r.text)
```

## Commercial products

## Web scraping vs. web crawling

## Web scraping vs. AI and ML

## Web scraping vs. RPA

## <...> scraping

## Legal concerns

## Web scraping as part of data science

## The cat and mouse game

## Closing best practices
