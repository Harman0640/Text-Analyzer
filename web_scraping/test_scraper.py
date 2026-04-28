from scraping import scrape_website

url = "https://www.bbc.com/news"
result = scrape_website(url)

print(result[:1000])  # Print first 1000 characters
