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

## Requests hands##on

## Using Requests

## Getting the weather with Requests
