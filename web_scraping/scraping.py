# import requests
# from bs4 import BeautifulSoup


# def scrape_website(url: str) -> str:
#     """
#     Fetches a website and extracts paragraph text.
#     Returns clean text or a user-friendly fallback message.
#     """

#     headers = {
#         "User-Agent": (
#             "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#             "AppleWebKit/537.36 (KHTML, like Gecko) "
#             "Chrome/120.0.0.0 Safari/537.36"
#         )
#     }

#     try:
#         response = requests.get(url, headers=headers, timeout=10)

#         # ---- Handle blocked / restricted sites ----
#         if response.status_code in [403, 429, 500]:
#             return (
#                 "⚠️ This website restricts automated access.\n\n"
#                 "Please try another public website such as:\n"
#                 "• https://quotes.toscrape.com\n"
#                 "• https://books.toscrape.com\n"
#                 "• https://www.geeksforgeeks.org"
#             )

#         response.raise_for_status()

#         soup = BeautifulSoup(response.text, "lxml")
#         paragraphs = soup.find_all("p")

#         if not paragraphs:
#             return "⚠️ No readable text found on this page."

#         text = " ".join(p.get_text(strip=True) for p in paragraphs)

#         return text

#     except requests.exceptions.MissingSchema:
#         return "⚠️ Invalid URL. Please include http:// or https://"

#     except requests.exceptions.Timeout:
#         return "⚠️ The website took too long to respond. Try another URL."

#     except requests.exceptions.ConnectionError:
#         return "⚠️ Failed to connect to the website."

#     except Exception:
#         return (
#             "⚠️ Unable to scrape this website due to access restrictions.\n"
#             "Please try a different public website."
#         )

# ------------------------------- New Scraping Code -----------------------------------

import requests
from bs4 import BeautifulSoup


def scrape_website(url: str) -> dict:
    """
    Scrapes a public website and extracts readable text.

    Returns:
        dict:
            {
                "success": bool,
                "text": str,
                "word_count": int,
                "source": str,
                "error": str (if any)
            }
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)

        # ---- Handle restricted access ----
        if response.status_code in [403, 429, 500]:
            return {
                "success": False,
                "text": "",
                "word_count": 0,
                "source": url,
                "error": "⚠️ This website restricts automated access."
            }

        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")

        text = ""

        # ---- Strategy 1: Paragraphs ----
        paragraphs = soup.find_all("p")
        if paragraphs:
            text = " ".join(p.get_text(strip=True) for p in paragraphs)

        # ---- Strategy 2: Quotes (quotes.toscrape.com) ----
        if len(text) < 50:
            quotes = soup.find_all("span", class_="text")
            if quotes:
                text = " ".join(q.get_text(strip=True) for q in quotes)

        # ---- Strategy 3: Article content ----
        if len(text) < 50:
            article = soup.find("article")
            if article:
                text = article.get_text(separator=" ", strip=True)

        # ---- Strategy 4: Fallback (entire page text) ----
        if len(text) < 50:
            text = soup.get_text(separator=" ", strip=True)

        # ---- Clean text ----
        text = " ".join(text.split())  # remove extra spaces

        # ---- Final validation ----
        if len(text.strip()) < 20:
            return {
                "success": False,
                "text": "",
                "word_count": 0,
                "source": url,
                "error": "⚠️ Could not extract meaningful content from this page."
            }

        return {
            "success": True,
            "text": text,
            "word_count": len(text.split()),
            "source": url,
            "error": ""
        }

    except requests.exceptions.MissingSchema:
        return {
            "success": False,
            "text": "",
            "word_count": 0,
            "source": url,
            "error": "⚠️ Invalid URL. Please include http:// or https://"
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "text": "",
            "word_count": 0,
            "source": url,
            "error": "⚠️ The website took too long to respond."
        }

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "text": "",
            "word_count": 0,
            "source": url,
            "error": "⚠️ Failed to connect to the website."
        }

    except Exception:
        return {
            "success": False,
            "text": "",
            "word_count": 0,
            "source": url,
            "error": "⚠️ Unable to scrape this website."
        }