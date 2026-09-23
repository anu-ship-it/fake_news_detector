"""
news_feed.py - downloads today's headlines from public RSS feeds (no API key needed).
Edit the FEEDS dictionary to add or remove news sources.
"""

import html
import re
import time
import xml.etree.ElementTree as ET

import requests

FEEDS = {
    "Google News (India)": "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en",
    "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "The Guardian": "https://www.theguardian.com/international/rss",
    "Times of India": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    "The Hindu": "https://www.thehindu.com/news/national/feeder/default.rss",
}

CACHE_SECONDS = 600  # reuse downloaded headlines for 10 minutes
_cache = {}


def _strip_html(text):
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def fetch_feed(name, limit=15, force=False):
    """Return a list of dicts: title, summary, link, source, published, text."""
    if not force and name in _cache and time.time() - _cache[name][0] < CACHE_SECONDS:
        return _cache[name][1]

    resp = requests.get(
        FEEDS[name], timeout=8, headers={"User-Agent": "Mozilla/5.0 (TruthLens college project)"}
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    is_google = "news.google.com" in FEEDS[name]
    items = []
    for item in root.iter("item"):
        title = _strip_html(item.findtext("title"))
        if not title:
            continue
        source = name
        if is_google:
            # Google titles look like "Headline - Publisher"; its description is only a link list
            source = _strip_html(item.findtext("source")) or name
            title = re.sub(r"\s+-\s+[^-]+$", "", title)
            summary = ""
        else:
            summary = _strip_html(item.findtext("description"))[:300]
        items.append({
            "title": title,
            "summary": summary,
            "link": (item.findtext("link") or "").strip(),
            "source": source,
            "published": (item.findtext("pubDate") or "")[:22],
            "text": f"{title}. {summary}".strip(),
        })
        if len(items) >= limit:
            break

    _cache[name] = (time.time(), items)
    return items