_## Other Python libraries and tools
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
- PAAS
- - Import.io - a web-based platform that allows you to extract data from websites without writing any code
- SAAS
- - http://portia.scrapinghub.com/ - a visual scraping tool that lets you extract data from websites without writing any code
- - https://parsehunb.com/ - a web scraping service that allows you to extract data from websites without writing any code
- Full package
- - http://kofax.com/data-integration-extraction - a web scraping tool that lets you extract data from websites without writing any code
- - https://fminer.com/ - a visual web scraping tool that lets you extract data from websites without writing any code
- Scraping helpers
- - Proxy servers
- - Cloud deployment
- - Captcha cracking services

## Web scraping vs. web crawling
- The difference between scraping and crawling is somewhat vague (many will use both terms interchangeably)
- In general terms, the term crawler indicates a program's ability to navigate web pages on its own, perhaps event without a well-defined end-goal or purpose (also known as a spider)
- Web Crawlers are heavily used by search engines like Google to retrieve contents for a URL, examine that page for other links, retrieve the URLs for those links, and so on.
- Design chooses:
- - In many cases, crawling will be restricted to a well-defined set of pages, e.g. a product pages of an online shop
- - In other cases, you will restrict yorself to a single website, but do not have a clear target regarding information extraction in mind. Instead, you simple wanna to create 
    a copy of site.
- - You might want to keep your crawling very open ended. For example, you might wish to start from a series of keywords, Google each of them, crawl to the top ten results for 
    every query and crawl those pages for table, images, articles and so on.
- Think carefully about which data you actually want to gather.
- Use a database to store the data you gather.
- Separate crawling from scraping.
- Stop early and often ( provide a way to stop the process)
- Retry on aborts (provide a way to retry the process)
- Crawling the queue (provide a way to prioritize the pages you want to crawl)
- Parallel programming (provide a way to crawl multiple pages at once, and hence the need for a database to store the data)
- Keep in mind the legal aspects of web scraping and crawling
- Usage of pre-build APIs can be beneficial here

## Web scraping vs. AI and ML
- To bypass Captcha you can use next technique:
- - Text mining
- - OCR
- - Visual web scraping
- - Computer vision
- - Captcha cracking
- - Used in heavy crawling projects where the scraper is the main product

## Web scraping vs. RPA
- Robotic Process Automation (RPA) is a technology that uses software robots to automate repetitive tasks
- - To automate simple back-end processes
- - Second wave of automation
- Many commercial web scraping tools have rebranded themselves as RPA tools
- RPA: web scraping but also + PDF scraping + screen UI scrapping
- Also a workflow oriented design
- Maturity is typically better
- - Though can still be tricky with complex sites
- - Expensive licenses
- - Maintenance required for workflow changes
- - Also not 100% accurate (OCR, granular selection rules)
- - But - interesting to consider given the use case


## <...> scraping
- PDF scrapping?
- - PDF to text tools
- - PDF libraries (https://github.com/pmaupin/pdfrw)
- - Tabula: for table extraction (https://tabula.technology/)
- - Camelot (https://blog.socialcops.com/technology/engineering/camelot-python-library-pdf-data/)
- OCR
- - Tesseract (https://github.com/tesseract-ocr/tesseract)
- Unstructured text
- - Toolkits such as SpaCy, AllenNPL, nltk, gensim
- Computer Vision
- - OpenCV, TensorFlow, Keras (Using deep learning approaches to detect objects or scenes)
- Screen scraping and instrumentation
- - PyAutoGUI, Selenium, Puppeteer, AutoIt, Sikuli
- - UiAutomation (https://pypi.org.project/UiAutomation/)
- - Pywinauto (https://pywinauto.readthedocs.io/en/latest/)
- - Automagica (https://automagica.readthedocs.io/en/latest/)
- - WinAppDriver (Service to automate Windows application testing)
- Mobile App Scraping 
- - Using e.g. Android emulator combined with automation framework or screen scraping
- Other proprietary web technologies
- - Java applets - can be easily decompiled
- - Flash - can be decompiled
- - WebGL - still a relatively new technology, but can be scraped

## Legal concerns
- The legal aspects of web scraping are complex and vary from country to country
- hiq vs. LinkedIn scraping case (https://www.eff.org/cases/hiq-v-linkedin)
- Facebook vs. Power Ventures (https://www.eff.org/cases/facebook-v-power-ventures)
- Google vs. Microsoft (https://www.eff.org/cases/google-v-microsoft)
- RyanAir vs. Screen Scraping (https://www.eff.org/cases/ryanair-v-screen-scraping)
- Breach of Terms and Conditions
- Copyright or Trademark Infringement
- Computer Fraud and Abuse Act (CFAA)
- Trespass to Chattels
- Robots Exclusion Protocol (robots.txt)
- The Digital Millennium Copyright Act (DMCA), CAN-SPAM Act 
- The EU Database Directive of 1996
- The Computer Misuse Act 1990 (UK)
- General Data Protection Regulation (GDPR)
- Article 13 and 11 of the EU Copyright Directive
- Most of this can be summarized in:
- - Copyright law
- - Privacy law
- - Breach of contract
- Get written permission from the website owner
- - The best way to avoid legal issues is to get written permission from a website's owner covering which data you can scrape and which extent
- Check the website's terms of service
- - These will often include explicit provisions against web scraping
- Public information only
- - If a site exposes info publicly, it's generally fair game
- - Dont login into a site and scrape data
- No personal information (privacy concerns)
- Don't cause damage (hammer websites by request, overloading networks, etc.)
- Copyrighted material (check carefully whether your scraping case would fall under far use and do not use copyrighted material for commercial purposes)
- Check the robots.txt file
- Allowed but no private information, no personal information, ni copyrighted material

## Web scraping as part of data science
- A one-shot project where web scraping can offer valuable data?
- - E.g. a single report or descriptive analysis
- - Then not really a deployment or maintainability issue
- A predictive model trained on web scraped data?
- - Which features need to be refreshed?
- - Puts extra pressure on production
- - How long can we use the scraped data? What if the data source go down?
- - GOGO - Get data, Organize data, Get data, Organize data
- - When used in reporting context: same concerns in case reporting is continuous and repeated!
- Typical issue: Why can't we use/ don't we have Facebook's data? ( consider internal data sources)
- RPA has led in number of new job roles (robot supervisor, robot developer, idea champion)
- Similar roles apply in a web scraping context multidisciplinary team including: 
- - Database expert
- - Programmers
- - Data scientists
- - Web Developers
- - Compliance officer
- - Data engineer
- - Manager
- Scope, scale and size depends on view on web scraping projects
- - From strategic core of main business to tactical to simply operational or one-off projects
- Consider buy versus build decision
- - Build - initial training and setup cost, offers more flexibility in long run
- - Buy - faster, less maintenance, less flexibility
- - Both require stringent maintenance and monitoring
- Saying no to:
- - When there is no well-defined question ( e.g. we want to copy of Facebook)
- - When the legal risk is deemed too high
- - When technological capabilities are not available
- - When there is no clear business case

## The cat and mouse game
- Websites take increasingly more advanced measures to prevent scraping
- - Rate limiting
- - IP blocking
- - Browser checks
- - JS based checks
- - HTML and JS obfuscation
- - UI event fingerprinting
- Avoidance techniques:
- - Fake as much a possible: User-Agent, Referer, IP, headers, cookies
- - Use proxy's or the cloud: but not all providers welcome scrapers, and not all websites like all providers
- - Timing and retry strategies: slow down, randomize, use smart retries
- - Captcha avoidance: text mining, OCR, visual web scraping, computer vision, captcha cracking
- - Fake UI events (typing speed, mouse movements, scrolling)
- Captcha service (2captcha, anti-captcha, deathbycaptcha)
- Other Services (click farms, data entry services, Amazon Mechanical Turk)
- To read:
- - https://towardsdatascience.com/deep-learning-drops-breaking-captcha-20c8fc96e6a3
- - https://medium.com/@ageitgey/how-to-break-a-captcha-system-in-15-minutes-with-machine-learning-dbebb035a710
- - https://www.npr.org/sections/thetwo-way/2017/10/26/560082659/ai-model-fundamentally-cracks-captchas-scientists-say
- Check whether the captcha appears every time, or only after some amount of time or every so often.
- - https://www.f5.com/company/blog/detecting-phantomjs-based-visitors
- - https://github.com/intoli/intoli-article-materials/blob/master/articles/making-chrome-headless-undetectable/README.md
- - https://antoinevastel.com/bot%20detection/2017/08/05/detect-chrome-headless.html?utm_source=frontendfocus&utm_medium=email

## Closing best practices
- Go for an API first (always check first whether the site you wish to scrape has an API)
- Use the best tools (e.g. Scrapy, BeautifulSoup, Selenium)
- Play nice (don't hammer a website with hundreds of requests per second, consider contacting the webmaster of the site and work out a way to work together)
- Consider the user agent and refer (remember the User-Agent and Referer headers)
- Web servers are picky (Whether it's URL parameters, headers or form data, some sites come with very picky and strange requirements regarding their ordering, presence and values)
- Check your browser (start from a fresh session and use dev tools to follow along through the requests, if everythng goes well try to emulate the same behaviour as well, use 
  curl and other CLI to debug difficult cases)
- Before going for a full JS engine, consider internal APIs (often the data you want is already available in a more structured form)
- Assume it will crash (the web is dynamic place, and websites change all the time. make sure to write scrapers in such a way that they provide early and detailed warnings when 
  something goes wrong)
- Crawling is hard (when writing an advance crawler you quickly need to incorporate a database deal with restarting scripts, monitoring, queue management, timestamps and so on)
- Some tools are helpful, some aren't 
- - There are various many companies offering 'cloud scraping' solutions like Scrapy
- - The main benefit of using is that you can utilize their fleet of servers to quickly parallelize your scraping tasks
- - Don't put too much trust in expensive GUE scraping tools, however in most cases thyey'll only work with basic pages, cannot deal with JS or will lead to construction of a 
    scrapping pipeline that is hard to maintain.
- Scraping is a cat-and-mouse game (websites will try to prevent you from scraping, and you will need to find ways to avoid their countermeasures)
- Keep in mind the managerial and legal concerns, and where web scraping fits in your data science pipeline