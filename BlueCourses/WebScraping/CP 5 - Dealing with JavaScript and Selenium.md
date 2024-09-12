## JavaScript

- JavaScript is a high-level programming language that is used to make web pages interactive.
- - Contrary to many other programming languages, JavaScript is not executed on the server but on the client side.
- - Browsers come with a JS engine and execute it
- - <script>...</script> tags or reference to a file containing source code
- For complex sites, JS can end up doing a lot:
- - SPA (Single Page Applications) are built using JS (React, Angular, Vue)
- - Dynamically changed and added content
- - Setting cookies 
- - Performing browser actions/checks
- - AJAX requests
- Requests/Beatiful Soup doesn't come with a JavaScript engine.
- - For those JS-heavy web-sites we might have to emulate a browser to get the content we want.

## Selenium
- Selenium - a browser instrumentation/automation framework
- - https://www.selenium.dev/
- - For other languages as well
- - Depends on a WebDriver - a browser (Chrome, Firefox, etc.) specific driver
- - Most close to real browser, but comes with overhead
- - Also a bit harder - No longer perform HTTP request and parse HTML, but 'click buttons', 'find text', 'wait for this to load', 'inject JS'

```python
from selenium import webdriver # Import the webdriver class

    
driver = webdriver.Chrome() # Create a new instance of the Chrome driver
driver.implicitly_wait(10) # Wait for up to 10 seconds for elements to appear

url = 'https://www.bluecourses.com' # URL to scrape

driver.get(url) # Load the web page
driver.quit()  # Close the browser
```

### Chrome Headless example
```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument("--headless") # Run in headless mode
chrome_options.add_argument("--window-size=1920x1080") # Set window size

driver = webdriver.Chrome(options=chrome_options)
```

### Selenium Selectors
- find_element_by_id # Find an element by its id
- find_element_by_name # Find an element by its name
- find_element_by_xpath # Find an element by its XPath
- find_element_by_link_text # Find an element by its link text
- find_element_by_partial_link_text # Find an element by its partial link text
- find_element_by_tag_name # Find an element by its tag name
- find_element_by_class_name # Find an element by its class name
- find_element_by_css_selector # Find an element by its CSS selector


## Selenium Examples

```python
# In this notebook, we'll take a look at a couple of examples using Selenium. We start by importing the modules we need and starting the Selenium-driven web browser.

# We don't use headless mode here, as we would like to see what's going on as we execute our commands.

from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys
driver = webdriver.Chrome()
driver.implicitly_wait(10)
# Navigating BlueCourses
# For this first example, let's visit our home page and read out a list of courses, as we did before using Beautiful Soup.

driver.get('https://www.bluecourses.com')
# Note that Selenium provides many ways to find elements. E.g. by using CSS selectors (more feature proof than select() in Beautiful Soup). Note that attributes here should be retrieved using get_attribute().

courses = driver.find_elements_by_css_selector('article.course')
for course in courses:
    print(course.find_element_by_css_selector('.course-title').text)
    print(course.find_element_by_tag_name('a').get_attribute('href'))
    
    
# Filling out a simple form
# For a second example, we can show how to interact with various form elements. This example illustrates how Selenium requires a more UI-driven way of working rather than thinking from an HTTP interaction perspective.

driver.get('http://www.webscrapingfordatascience.com/postform2/')
# Textual elements can be filled in using clear and send_keys.

driver.find_element_by_name('name').clear()
driver.find_element_by_name('name').send_keys('Seppe')
# We can also retrieve elements through XPath selectors. XPath is a relatively complex but powerful XML query language. See https://www.w3schools.com/xml/xpath_syntax.asp for a good overview of the syntax.

driver.find_element_by_xpath('//input[@name="gender"][@value="N"]').click()
driver.find_element_by_name('fries').click()
driver.find_element_by_name('salad').click()
Select(driver.find_element_by_name('haircolor')).select_by_value('brown')
driver.find_element_by_name('comments').clear()
driver.find_element_by_name('comments').send_keys(['First line', Keys.ENTER, 'Second line'])
driver.find_element_by_xpath('//input[@type="submit"]').click()
driver.find_element_by_tag_name('body').text
# 'Thanks for submitting your information\nHere\'s a dump of the form data that was submitted:\narray(6) {\n  ["name"]=>\n  string(5) "Seppe"\n  ["gender"]=>\n  string(1) "N"\n  ["fries"]=>\n  string(4) "like"\n  ["salad"]=>\n  string(4) "like"\n  ["haircolor"]=>\n  string(5) "brown"\n  ["comments"]=>\n  string(23) "First line\nSecond line"\n}'
# Note two special properties, innerHTML and outerHTML (DOM attributes), which allow to get the full inner and outer HTML contents of tags. Note that you could still use a HTML parsing library like Beautiful Soup if you'd like to parse these further without using Selenium.

driver.find_element_by_tag_name('body').get_attribute('innerHTML')
# '\n\n\n<h2>Thanks for submitting your information</h2>\n\n<p>Here\'s a dump of the form data that was submitted:</p>\n\n<pre>array(6) {\n  ["name"]=&gt;\n  string(5) "Seppe"\n  ["gender"]=&gt;\n  string(1) "N"\n  ["fries"]=&gt;\n  string(4) "like"\n  ["salad"]=&gt;\n  string(4) "like"\n  ["haircolor"]=&gt;\n  string(5) "brown"\n  ["comments"]=&gt;\n  string(23) "First line\nSecond line"\n}\n</pre>\n\n\n\t\n\n'
driver.find_element_by_tag_name('body').get_attribute('outerHTML')
# '<body>\n\n\n<h2>Thanks for submitting your information</h2>\n\n<p>Here\'s a dump of the form data that was submitted:</p>\n\n<pre>array(6) {\n  ["name"]=&gt;\n  string(5) "Seppe"\n  ["gender"]=&gt;\n  string(1) "N"\n  ["fries"]=&gt;\n  string(4) "like"\n  ["salad"]=&gt;\n  string(4) "like"\n  ["haircolor"]=&gt;\n  string(5) "brown"\n  ["comments"]=&gt;\n  string(23) "First line\nSecond line"\n}\n</pre>\n\n\n\t\n\n</body>'
# Getting a list of McDonalds locations in New York
driver.get('https://www.mcdonalds.com/us/en-us/restaurant-locator.html')
driver.find_element_by_id('search').send_keys('New York')
driver.find_element_by_css_selector('button[aria-label="search"]').click()
driver.find_element_by_css_selector('.button-toggle button[aria-label="List View"]').click()

# ---------------------------------------------------------------------------
# ElementNotInteractableException           Traceback (most recent call last)
# <ipython-input-52-c61b1b816c91> in <module>
# ----> 1 driver.find_element_by_css_selector('.button-toggle button[aria-label="List View"]').click()
# ...
# ElementNotInteractableException: Message: element not interactable
#   (Session info: chrome=84.0.4147.105)

# Alternatively, we could also do the following, by executing JavaScript in the browser:

driver.execute_script(
    'arguments[0].click();', 
    driver.find_element_by_css_selector('.button-toggle button[aria-label="List View"]')
)
# Next, we'll continue to load in all results until the 'Load More' button disappears. Normally, you'd opt to use a more robust approach here using explicit waits (https://www.selenium.dev/documentation/en/webdriver/waits/). Since we have defined an implicit wait above, Selenium will try executing our commands until the implicit timeout is reached, after which it throws an exception.

while True:
    try:
        driver.find_element_by_css_selector('div.rl-listview__load-more button').click()
    except:
        break # All results loaded
for details in driver.find_elements_by_css_selector('.rl-details'):
    print(details.text)

# OPEN
# 160 Broadway
# New York, Ny 10038

# If you follow along with the network requests in the browser. You might also have noticed that the restaurant location retriever actually calls an internal JavaScript API. Hence, we could also try accessing this directly using Requests and see whether that works. The URL parameters obviously expose ways to play around with this:
```
```python
import requests
requests.get('https://www.mcdonalds.com/googleapps/GoogleRestaurantLocAction.do', params={
    'method': 'searchLocation',
    'latitude': 40.7127753,
    'longitude': -74.0059728,
    'radius': 30.045,
    'maxResults': 3,
    'country': 'us',
    'language': 'en-us'
}).json()
{'features': [{'geometry': {'coordinates': [-74.010086, 40.709438]},
   'properties': {'jobUrl': '',
    'longDescription': '',
    'todayHours': '04:00 - 04:00',
    'driveTodayHours': '04:00 - 04:00',
    'id': '195500284446-en-us',
    'filterType': ['WIFI',
     'GIFTCARDS',
     'MOBILEOFFERS',
     'MOBILEORDERS',
     'INDOORDININGAVAILABLE',
     'MCDELIVERY',
     'TWENTYFOURHOURS'],
    'addressLine1': '160 Broadway',
    'addressLine2': 'STAMFORD FIELD OFFICE',
    'addressLine3': 'New York',
    'addressLine4': 'USA',
    'subDivision': 'NY',
    'postcode': '10038',
    'customAddress': 'New York, NY 10038',
    'telephone': '(212) 385-2066',
    'restauranthours': {'hoursMonday': '04:00 - 04:00',
     'hoursTuesday': '04:00 - 04:00',
     'hoursWednesday': '04:00 - 04:00',
     'hoursThursday': '04:00 - 04:00',
     'hoursFriday': '04:00 - 04:00',
     'hoursSaturday': '04:00 - 04:00',
     'hoursSunday': '04:00 - 04:00'},
    'drivethruhours': {'driveHoursMonday': '04:00 - 04:00',
     'driveHoursTuesday': '04:00 - 04:00',
     'driveHoursWednesday': '04:00 - 04:00',
     'driveHoursThursday': '04:00 - 04:00',
     'driveHoursFriday': '04:00 - 04:00',
     'driveHoursSaturday': '04:00 - 04:00',
     'driveHoursSunday': '04:00 - 04:00'},
    'familyevent': [],
    'identifiers': {'storeIdentifier': [{'identifierType': 'SiteIdNumber',
       'identifierValue': '311090'},
      {'identifierType': 'NatlStrNumber', 'identifierValue': '10528'},
      {'identifierType': 'Region ID', 'identifierValue': '30'},
      {'identifierType': 'Co-Op', 'identifierValue': 'NEW YORK METRO'},
      {'identifierType': 'Co-Op ID', 'identifierValue': '246'},
      {'identifierType': 'TV-Market', 'identifierValue': 'NEW YORK CITY, NY'},
      {'identifierType': 'TV-Market ID', 'identifierValue': '16200'}],
     'gblnumber': '195500284446'},
    'birthDaysParties': '0',
    'driveThru': '0',
    'outDoorPlayGround': '0',
    'indoorPlayGround': '0',
    'wifi': '0',
    'breakFast': '0',
    'nightMenu': '0',
    'giftCards': '0',
    'mobileOffers': '0',
    'restaurantUrl': 'https://www.ubereats.com/new-york/food-delivery/mcdonalds-fidi-160-broadway/Cn7rm31RTym93cHeWFUSow/?utm_source=Nero&utm_medium=loyalty&utm_campaign=yoyowallet&ue=ue',
    'storeNotice': '',
    'openstatus': 'OPEN',
    'identifierValue': '10528',
    'noticeStartDate': '',
    'noticeEndDate': '',
    'webStatus': 'OPEN',
    'mcDeliveries': {'mcDelivery': [{'identifier': 'UberEats',
       'marketingName': 'Uber Eats',
       'deliveryURL': 'https://www.ubereats.com/new-york/food-delivery/mcdonalds-fidi-160-broadway/Cn7rm31RTym93cHeWFUSow/?utm_source=Nero&utm_medium=loyalty&utm_campaign=yoyowallet&ue=ue'},
      {'identifier': 'DoorDash',
       'marketingName': 'DoorDash',
       'deliveryURL': 'https://www.doordash.com/store/837189'},
      {'identifier': 'Grubhub',
       'marketingName': 'Grubhub',
       'deliveryURL': 'https://www.grubhub.com/restaurant/mcdonalds-160-broadway-new-york/1338459?utm_source=mcdonalds_website&utm_medium=enterprise-rest_partner&utm_campaign=rest-brand_5de59e10-3ba6-11e9-b0a9-3d145c71a10c&utm_content=mcdonalds'}]}}},
  {'geometry': {'coordinates': [-74.010736, 40.716366]},
   'properties': {'jobUrl': '',
    'longDescription': '',
    'todayHours': '05:00 - 04:00',
    'driveTodayHours': '04:00 - 04:00',
    'id': '195500284712-en-us',
    'filterType': ['WIFI',
     'GIFTCARDS',
     'MOBILEOFFERS',
     'MOBILEORDERS',
     'MCDELIVERY',
     'TWENTYFOURHOURS'],
    'addressLine1': '167 Chambers St (303 Greenwich St)',
    'addressLine2': 'STAMFORD FIELD OFFICE',
    'addressLine3': 'New York',
    'addressLine4': 'USA',
    'subDivision': 'NY',
    'postcode': '10013',
    'customAddress': 'New York, NY 10013',
    'telephone': '(212) 608-2405',
    'restauranthours': {'hoursMonday': '05:00 - 04:00',
     'hoursTuesday': '05:00 - 04:00',
     'hoursWednesday': '05:00 - 04:00',
     'hoursThursday': '05:00 - 04:00',
     'hoursFriday': '05:00 - 04:00',
     'hoursSaturday': '05:00 - 04:00',
     'hoursSunday': '05:00 - 04:00'},
    'drivethruhours': {'driveHoursMonday': '04:00 - 04:00',
     'driveHoursTuesday': '04:00 - 04:00',
     'driveHoursWednesday': '04:00 - 04:00',
     'driveHoursThursday': '04:00 - 04:00',
     'driveHoursFriday': '04:00 - 04:00',
     'driveHoursSaturday': '04:00 - 04:00',
     'driveHoursSunday': '04:00 - 04:00'},
    'familyevent': [],
    'identifiers': {'storeIdentifier': [{'identifierType': 'SiteIdNumber',
       'identifierValue': '311193'},
      {'identifierType': 'NatlStrNumber', 'identifierValue': '11163'},
      {'identifierType': 'Region ID', 'identifierValue': '30'},
      {'identifierType': 'Co-Op', 'identifierValue': 'NEW YORK METRO'},
      {'identifierType': 'Co-Op ID', 'identifierValue': '246'},
      {'identifierType': 'TV-Market', 'identifierValue': 'NEW YORK CITY, NY'},
      {'identifierType': 'TV-Market ID', 'identifierValue': '16200'}],
     'gblnumber': '195500284712'},
    'birthDaysParties': '0',
    'driveThru': '0',
    'outDoorPlayGround': '0',
    'indoorPlayGround': '0',
    'wifi': '0',
    'breakFast': '0',
    'nightMenu': '0',
    'giftCards': '0',
    'mobileOffers': '0',
    'restaurantUrl': 'https://www.ubereats.com/new-york/food-delivery/mcdonalds-tribeca-chambers-%26-greenwich/PbjRsUAQR1GHD1D0ZQqRCw/?utm_source=Nero&utm_medium=loyalty&utm_campaign=yoyowallet&ue=ue',
    'storeNotice': '',
    'openstatus': 'OPEN',
    'identifierValue': '11163',
    'noticeStartDate': '',
    'noticeEndDate': '',
    'webStatus': 'OPEN',
    'mcDeliveries': {'mcDelivery': [{'identifier': 'UberEats',
       'marketingName': 'Uber Eats',
       'deliveryURL': 'https://www.ubereats.com/new-york/food-delivery/mcdonalds-tribeca-chambers-%26-greenwich/PbjRsUAQR1GHD1D0ZQqRCw/?utm_source=Nero&utm_medium=loyalty&utm_campaign=yoyowallet&ue=ue'},
      {'identifier': 'DoorDash',
       'marketingName': 'DoorDash',
       'deliveryURL': 'https://www.doordash.com/store/837187'},
      {'identifier': 'Grubhub',
       'marketingName': 'Grubhub',
       'deliveryURL': 'https://www.grubhub.com/restaurant/mcdonalds-167-chambers-st-new-york/1339054?utm_source=mcdonalds_website&utm_medium=enterprise-rest_partner&utm_campaign=rest-brand_5de59e10-3ba6-11e9-b0a9-3d145c71a10c&utm_content=mcdonalds'}]}}},
  {'geometry': {'coordinates': [-74.001052, 40.718587]},
   'properties': {'jobUrl': '',
    'longDescription': '',
    'todayHours': '07:00 - 19:00',
    'driveTodayHours': '07:00 - 19:00',
    'id': '195500283562-en-us',
    'filterType': ['WIFI',
     'GIFTCARDS',
     'MOBILEOFFERS',
     'MOBILEORDERS',
     'INDOORDININGAVAILABLE',
     'MCDELIVERY'],
    'addressLine1': '262 Canal St',
    'addressLine2': 'STAMFORD FIELD OFFICE',
    'addressLine3': 'New York',
    'addressLine4': 'USA',
    'subDivision': 'NY',
    'postcode': '10013',
    'customAddress': 'New York, NY 10013',
    'telephone': '(212) 941-5823',
    'restauranthours': {'hoursMonday': '07:00 - 19:00',
     'hoursTuesday': '07:00 - 19:00',
     'hoursWednesday': '07:00 - 19:00',
     'hoursThursday': '07:00 - 19:00',
     'hoursFriday': '07:00 - 19:00',
     'hoursSaturday': '07:00 - 19:00',
     'hoursSunday': '07:00 - 19:00'},
    'drivethruhours': {'driveHoursMonday': '07:00 - 19:00',
     'driveHoursTuesday': '07:00 - 19:00',
     'driveHoursWednesday': '07:00 - 19:00',
     'driveHoursThursday': '07:00 - 19:00',
     'driveHoursFriday': '07:00 - 19:00',
     'driveHoursSaturday': '07:00 - 19:00',
     'driveHoursSunday': '07:00 - 19:00'},
    'familyevent': [],
    'identifiers': {'storeIdentifier': [{'identifierType': 'SiteIdNumber',
       'identifierValue': '310630'},
      {'identifierType': 'NatlStrNumber', 'identifierValue': '4682'},
      {'identifierType': 'Region ID', 'identifierValue': '30'},
      {'identifierType': 'Co-Op', 'identifierValue': 'NEW YORK METRO'},
      {'identifierType': 'Co-Op ID', 'identifierValue': '246'},
      {'identifierType': 'TV-Market', 'identifierValue': 'NEW YORK CITY, NY'},
      {'identifierType': 'TV-Market ID', 'identifierValue': '16200'}],
     'gblnumber': '195500283562'},
    'birthDaysParties': '0',
    'driveThru': '0',
    'outDoorPlayGround': '0',
    'indoorPlayGround': '0',
    'wifi': '0',
    'breakFast': '0',
    'nightMenu': '0',
    'giftCards': '0',
    'mobileOffers': '0',
    'restaurantUrl': 'https://www.ubereats.com/new-york/food-delivery/mcdonalds-chinatown-canal-st/J1GVi-cQTvuueL7b3qSk9w/?utm_source=Nero&utm_medium=loyalty&utm_campaign=yoyowallet&ue=ue',
    'storeNotice': '',
    'openstatus': 'OPEN',
    'identifierValue': '4682',
    'noticeStartDate': '',
    'noticeEndDate': '',
    'webStatus': 'OPEN',
    'mcDeliveries': {'mcDelivery': [{'identifier': 'UberEats',
       'marketingName': 'Uber Eats',
       'deliveryURL': 'https://www.ubereats.com/new-york/food-delivery/mcdonalds-chinatown-canal-st/J1GVi-cQTvuueL7b3qSk9w/?utm_source=Nero&utm_medium=loyalty&utm_campaign=yoyowallet&ue=ue'},
      {'identifier': 'DoorDash',
       'marketingName': 'DoorDash',
       'deliveryURL': 'https://www.doordash.com/store/837238'},
      {'identifier': 'Grubhub',
       'marketingName': 'Grubhub',
       'deliveryURL': 'https://www.grubhub.com/restaurant/mcdonalds-262-canal-st-new-york/1339415?utm_source=mcdonalds_website&utm_medium=enterprise-rest_partner&utm_campaign=rest-brand_5de59e10-3ba6-11e9-b0a9-3d145c71a10c&utm_content=mcdonalds'}]}}}]}
# Even if the website you wish to scrape does not provide an API, it's always recommended to keep an eye on your browser's developer tools networking information to see if you can spot JavaScript-driven requests to URL endpoints which return nicely structured JSON data, as is the case here.

# Even although an API might not be documented, fetching the information directly from such an "internal APIs" is always a clever idea, as this will avoid having to deal with the HTML soup. In fact, we get here nicely structured JSON data directly!
```

- Recall: 
- - Even if the website you wish to scrape doesn't provide an API, it's always reccomended to keep eye on your browser's de tools networking info to see if you can spot 
    JS-driven requests to URL endpoins which return nicely structured JSON data, as is the case here.
- - Even althrough an API might not be documented, fetching the ifo directly from such 'internal APIs' is always a clever idea, as this will avoid having to deal with the HTML 
    soup.
- Implicit waits and explicit wait conditions
- Action chains for more complex actions like hovering and dragging and dropping.
- JS injection (execute_script method)
