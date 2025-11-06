#!/usr/bin/env python3
"""
mangapark_scraper.py - Fixed, production-ready single-file MangaPark scraper.

Notable fixes/additions:
- Removes "Most Likes" and "Newly Added" entries that confuse indexing.
- Allows selecting chapters by chapter number (e.g. `138`), by index (1..N),
  by range (1-5), by list (1,3,7), or by title substring ("Black Wolf").
- Handles images with alpha channel by flattening them with Pillow before img2pdf.
- Playwright fallback for dynamic pages (optional).
- Metadata saving is disabled by default.
- Compatible with Python 3.11.9.

Usage:
    python mangapark_scraper.py "https://mangapark.io/title/127529-en-the-lazy-lord-masters-the-sword"
    python mangapark_scraper.py "naruto" -c "138"        # choose by chapter number
    python mangapark_scraper.py "naruto" -c "1-3,5"      # choose by index ranges
    python mangapark_scraper.py "naruto"                 # interactive

Dependencies:
    pip install requests beautifulsoup4 playwright img2pdf aiohttp tqdm pyyaml pillow
    playwright install   # once if using Playwright
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import io
import json
import logging
import os
import pathlib
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple, Union

import requests
from bs4 import BeautifulSoup

# Optional libs
try:
    import aiohttp
except Exception:
    aiohttp = None

try:
    import img2pdf
except Exception:
    img2pdf = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import yaml
except Exception:
    yaml = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Playwright (optional fallback)
try:
    from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = None

from urllib.parse import urljoin

# ---------------------------
# Config / constants
# ---------------------------
DEFAULT_THROTTLE = 1.0
DEFAULT_MAX_RETRIES = 4
DEFAULT_WORKERS = 4
USER_AGENT = "mangapark-scraper/1.0 (+https://github.com/your/repo) Requests/urllib3"

_ILLEGAL_PATH_RE = re.compile(r'[:<>"/\\|?*\x00-\x1F]')
_MAX_FILENAME_LEN = 255

logger = logging.getLogger("mangapark_scraper")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(_handler)

MANGAPARK_BASES = (
    "https://mangapark.com",
    "https://mangapark.org",
    "https://mangapark.me",
    "https://mangapark.net",
    "https://mangapark.io",
)


# ---------------------------
# Data classes
# ---------------------------
@dataclass
class Config:
    throttle: float = DEFAULT_THROTTLE
    max_retries: int = DEFAULT_MAX_RETRIES
    workers: int = DEFAULT_WORKERS
    timeout: int = 30
    user_agent: str = USER_AGENT
    session_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class ChapterRef:
    title: str
    url: str
    index: int  # 1-based index in list presented to user
    chapter_number: Optional[int] = None  # guessed numeric chapter id if any


@dataclass
class ScrapeMetadata:
    title: str
    source_url: str
    scraped_chapters: List[Dict]
    scraped_at: str


# ---------------------------
# Utils
# ---------------------------
def sanitize_filename(name: str, max_len: int = _MAX_FILENAME_LEN) -> str:
    name = (name or "").strip()
    name = re.sub(r"\s+", " ", name)
    name = _ILLEGAL_PATH_RE.sub("", name)
    name = name.rstrip(" .")
    if len(name) > max_len:
        name = name[: max_len - 3] + "..."
    return name or "untitled"


def ensure_dir(path: Union[str, pathlib.Path]) -> pathlib.Path:
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def parse_range_selection(sel: str) -> List[str]:
    """
    Parse a selection string into tokens. We return tokens as strings because
    tokens could be:
      - 'all'
      - '1-5'
      - '138' (chapter number)
      - '3' (index)
      - 'Black Wolf' (title substring)  <-- handled later
    We'll split by comma and return trimmed tokens.
    """
    if sel is None:
        return []
    sel = sel.strip()
    if not sel:
        return []
    parts = [p.strip() for p in sel.split(",") if p.strip()]
    return parts


# ---------------------------
# Retry decorator
# ---------------------------
def retry_backoff(max_retries: int = 4, base_delay: float = 1.0, allowed_exceptions=(Exception,)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except allowed_exceptions as e:
                    attempt += 1
                    if attempt > max_retries:
                        logger.error("Max retries reached for %s(): %s", func.__name__, e)
                        raise
                    delay = base_delay * (2 ** (attempt - 1))
                    jitter = delay * 0.1 * (0.5 - (time.time() % 1))
                    wait = max(0.1, delay + jitter)
                    logger.warning(
                        "Transient error in %s(): %s — retrying in %.1f seconds (attempt %d/%d)",
                        func.__name__,
                        e,
                        wait,
                        attempt,
                        max_retries,
                    )
                    time.sleep(wait)

        return wrapper

    return decorator


# ---------------------------
# HTTP client
# ---------------------------
class HttpClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.session = requests.Session()
        headers = {"User-Agent": self.cfg.user_agent, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
        headers.update(self.cfg.session_headers)
        self.session.headers.update(headers)
        self.timeout = cfg.timeout

    @retry_backoff()
    def get(self, url: str, **kwargs) -> requests.Response:
        logger.debug("GET %s", url)
        resp = self.session.get(url, timeout=self.timeout, **kwargs)
        resp.raise_for_status()
        return resp


# ---------------------------
# Playwright fallback
# ---------------------------
def render_with_playwright(url: str, wait_until: str = "networkidle", timeout: int = 30) -> str:
    if sync_playwright is None:
        raise RuntimeError("Playwright is not installed. pip install playwright ; playwright install")
    logger.info("Falling back to Playwright to render dynamic content for %s", url)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_default_navigation_timeout(timeout * 1000)
        page.goto(url, wait_until=wait_until)
        time.sleep(1.0)
        content = page.content()
        browser.close()
        return content


# ---------------------------
# Heuristics
# ---------------------------
def _guess_chapter_number_from_text(text: str) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"(?:ch(?:apter)?\.?|ch-|chapter[-\s]?)(\d+(\.\d+)?)", text, re.I)
    if m:
        try:
            num_str = m.group(1)
            if "." in num_str:
                num = int(num_str.split(".")[0])
            else:
                num = int(num_str)
            return num
        except Exception:
            pass
    m2 = re.findall(r"(\d+)", text)
    if m2:
        try:
            return int(m2[-1])
        except Exception:
            return None
    return None


def _is_mplists_link(href: str, text: str) -> bool:
    """
    Detect links that are 'mplists?sortby=' or 'Most Likes' / 'Newly Added' tabs.
    Return True to skip.
    """
    if not href:
        return False
    if "mplists?sortby=" in href:
        return True
    if text and ("most likes" in text.lower() or "newly added" in text.lower()):
        return True
    return False


# ---------------------------
# HTML parsers
# ---------------------------
def extract_title_and_chapters_from_title_page(html: str, base_url: str) -> Tuple[str, List[ChapterRef]]:
    soup = BeautifulSoup(html, "html.parser")

    # title
    title_tag = soup.find("h1")
    title = None
    if title_tag and title_tag.text.strip():
        title = title_tag.text.strip()
    else:
        og = soup.find("meta", attrs={"property": "og:title"}) or soup.find("meta", attrs={"name": "og:title"})
        if og and og.get("content"):
            title = og.get("content").strip()
    if not title:
        title = (soup.title.string or "untitled").strip()

    # find chapter anchors but skip mplists tabs and other noise
    candidates = []
    # prefer containers with 'chapter' in class/id
    for container in soup.find_all(True, attrs={"class": re.compile(r"chap|chapter", re.I)}):
        candidates.extend(container.find_all("a", href=True))
    if not candidates:
        for container in soup.find_all(True, attrs={"id": re.compile(r"chap|chapter", re.I)}):
            candidates.extend(container.find_all("a", href=True))
    # fallback: all anchors that look like chapter links
    if not candidates:
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/title/" in href or "/chapter/" in href or re.search(r"/\d+-ch", href):
                candidates.append(a)

    seen = set()
    chapters: List[ChapterRef] = []
    idx = 0
    for a in candidates:
        href = a.get("href")
        text = (a.text or "").strip()
        if _is_mplists_link(href or "", text):
            continue
        url = requests.compat.urljoin(base_url, href)
        if url in seen:
            continue
        seen.add(url)
        if not text:
            text = url.split("/")[-1]
        idx += 1
        chap_num = _guess_chapter_number_from_text(text) or _guess_chapter_number_from_text(url)
        chapters.append(ChapterRef(title=text, url=url, index=idx, chapter_number=chap_num))

    # If numeric chapter numbers are detected and appear decreasing, reverse to make ascending
    if len(chapters) >= 2:
        nums = [c.chapter_number for c in chapters[:6] if c.chapter_number is not None]
        if len(nums) >= 2 and nums[0] > nums[1]:
            chapters.reverse()
            for i, ch in enumerate(chapters, start=1):
                ch.index = i

    return title, chapters


def extract_image_urls_from_chapter_html(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    img_urls: List[str] = []

    lazy_attrs = ("data-src", "data-lazy", "data-url", "data-original", "data-cfsrc", "data-image", "src")
    srcset_attrs = ("srcset", "data-srcset")
    for img in soup.find_all("img"):
        found = False
        for attr in lazy_attrs:
            src = img.get(attr)
            if src:
                if isinstance(src, str) and src.startswith("data:"):
                    continue
                if re.search(r"\.(jpe?g|png|webp|gif)(\?|$)", src, re.I):
                    full = requests.compat.urljoin(base_url, src)
                    img_urls.append(full)
                    found = True
                    break
        if found:
            continue
        for attr in srcset_attrs:
            ss = img.get(attr)
            if ss:
                parts = [p.strip() for p in ss.split(",") if p.strip()]
                if parts:
                    candidate = parts[-1].split(" ")[0].strip()
                    if re.search(r"\.(jpe?g|png|webp|gif)(\?|$)", candidate, re.I):
                        img_urls.append(requests.compat.urljoin(base_url, candidate))
                        found = True
                        break
        if found:
            continue

    # parse scripts for arrays or quoted urls
    for script in soup.find_all("script"):
        text = script.string or script.get_text() or ""
        m_arr = re.search(r"(\[\"https?://[^\]]+?\.(?:jpg|jpeg|png|webp|gif)\"[^\]]*?\])", text, re.I | re.S)
        if m_arr:
            try:
                arr = json.loads(m_arr.group(1))
                for url in arr:
                    if isinstance(url, str) and re.search(r"\.(jpe?g|png|webp|gif)(\?|$)", url, re.I):
                        img_urls.append(url)
            except Exception:
                pass
        for m in re.finditer(r'"(https?://[^"]+\.(?:jpg|jpeg|png|webp|gif)(?:\?[^"]*)?)"', text, re.I):
            img_urls.append(m.group(1))

    if not img_urls:
        for m in re.finditer(r'"(https?://[^"]+\.(?:jpg|jpeg|png|webp|gif)(?:\?[^"]*)?)"', html, re.I):
            img_urls.append(m.group(1))

    final: List[str] = []
    seen = set()
    for u in img_urls:
        u2 = u.split("?")[0]
        if u2 not in seen:
            seen.add(u2)
            final.append(u)
    return final


# ---------------------------
# Image download + PDF
# ---------------------------
@retry_backoff()
def sync_download_image(session: requests.Session, url: str, timeout: int = 30) -> bytes:
    logger.debug("Downloading image (sync): %s", url)
    resp = session.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    return resp.content


async def aio_download_image(client: aiohttp.ClientSession, url: str, timeout: int = 30) -> bytes:
    logger.debug("Downloading image (async): %s", url)
    async with client.get(url, timeout=timeout) as resp:
        resp.raise_for_status()
        return await resp.read()


def save_images_to_pdf_bytes(img_bytes_list: Sequence[bytes]) -> bytes:
    """
    Flatten images with alpha channels and produce a PDF bytes object.
    Uses Pillow to convert to RGB JPEGs and img2pdf to combine.
    """
    if Image is None:
        raise RuntimeError("Pillow is required: pip install pillow")
    if img2pdf is None:
        raise RuntimeError("img2pdf is required: pip install img2pdf")

    processed = []
    for b in img_bytes_list:
        try:
            img = Image.open(io.BytesIO(b))
        except Exception:
            continue
        # Flatten alpha if present
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            try:
                rgba = img.convert("RGBA")
                bg.paste(rgba, mask=rgba.split()[3])
            except Exception:
                bg.paste(img.convert("RGB"))
            out = io.BytesIO()
            bg.save(out, format="JPEG", quality=95)
            processed.append(out.getvalue())
        else:
            out = io.BytesIO()
            img.convert("RGB").save(out, format="JPEG", quality=95)
            processed.append(out.getvalue())

    if not processed:
        raise RuntimeError("No valid images to convert to PDF")

    pdf_bytes = img2pdf.convert(processed)
    return pdf_bytes


# ---------------------------
# Main scraper class
# ---------------------------
class MangaParkScraper:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.http = HttpClient(self.cfg)

    def fetch_html(self, url: str) -> str:
        resp = self.http.get(url)
        return resp.text

    def fetch_all_chapters(self, title_url: str) -> Tuple[str, List[ChapterRef]]:
        """
        Fetch the title page and return (title, chapters) while filtering
        out mplists tabs and other noise so index/chapter mapping is clean.
        """
        logger.info("Fetching title page (clean chapters): %s", title_url)
        try:
            html = self.fetch_html(title_url)
        except Exception as e:
            logger.warning("Requests fetching failed: %s; trying Playwright if available", e)
            if sync_playwright is None:
                raise
            html = render_with_playwright(title_url)

        title, chapters = extract_title_and_chapters_from_title_page(html, base_url=title_url)

        # If chapters list seems empty, try Playwright render
        if not chapters and sync_playwright is not None:
            logger.info("No chapters via requests; trying Playwright render...")
            rendered = render_with_playwright(title_url)
            title, chapters = extract_title_and_chapters_from_title_page(rendered, base_url=title_url)

        if not chapters:
            logger.error("No chapters found on the title page.")
            raise RuntimeError("No chapters found on title page")

        logger.info("Title: %s — Chapters found (clean): %d", title, len(chapters))
        return title, chapters

    def fetch_title_page(self, url: str) -> Tuple[str, List[ChapterRef]]:
        # kept for backward compatibility — use fetch_all_chapters for cleaner behavior
        return self.fetch_all_chapters(url)

    def fetch_chapter_image_urls(self, chapter_url: str) -> List[str]:
        logger.info("Inspecting chapter for images: %s", chapter_url)

        # Skip obviously bad chapter history pages
        if "chapter-history" in chapter_url.lower():
            logger.warning("Skipped invalid chapter-history URL: %s", chapter_url)
            return []

        try:
            resp = self.http.get(chapter_url)
            html = resp.text
            img_urls = extract_image_urls_from_chapter_html(html, base_url=chapter_url)
            if not img_urls and sync_playwright is not None:
                rendered = render_with_playwright(chapter_url)
                img_urls = extract_image_urls_from_chapter_html(rendered, base_url=chapter_url)
        except Exception as e:
            logger.warning("Failed to extract via requests: %s — trying Playwright if available", e)
            if sync_playwright is None:
                raise
            rendered = render_with_playwright(chapter_url)
            img_urls = extract_image_urls_from_chapter_html(rendered, base_url=chapter_url)

        if not img_urls:
            logger.error("No image URLs found for chapter: %s", chapter_url)
            raise RuntimeError("No image URLs found for chapter")
        logger.info("Found %d images in chapter", len(img_urls))
        return img_urls

    def download_images_sync(self, urls: List[str], output_dir: pathlib.Path) -> List[bytes]:
        logger.info("Downloading %d images (sync mode)...", len(urls))
        imgs: List[bytes] = []
        s = self.http.session
        for i, url in enumerate(urls, start=1):
            time.sleep(self.cfg.throttle)
            logger.info("Downloading image %d/%d", i, len(urls))
            b = sync_download_image(s, url, timeout=self.cfg.timeout)
            imgs.append(b)
        return imgs

    async def download_images_async(self, urls: List[str], output_dir: pathlib.Path) -> List[bytes]:
        if aiohttp is None:
            raise RuntimeError("aiohttp is required for async mode")
        connector = aiohttp.TCPConnector(limit=self.cfg.workers)
        headers = {"User-Agent": self.cfg.user_agent}
        imgs: List[bytes] = [b""] * len(urls)
        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
            sem = asyncio.Semaphore(self.cfg.workers)

            async def fetch_one(i: int, url: str):
                async with sem:
                    for attempt in range(1, self.cfg.max_retries + 1):
                        try:
                            async with session.get(url, timeout=self.cfg.timeout) as resp:
                                resp.raise_for_status()
                                data = await resp.read()
                                imgs[i] = data
                                return
                        except Exception as e:
                            if attempt >= self.cfg.max_retries:
                                raise
                            await asyncio.sleep((2 ** (attempt - 1)) * 0.5)

            tasks = [fetch_one(i, u) for i, u in enumerate(urls)]
            if tqdm:
                for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Downloading images"):
                    await f
            else:
                await asyncio.gather(*tasks)
        return imgs

    def download_images(self, urls: List[str], output_dir: pathlib.Path, use_async: bool = True) -> List[bytes]:
        if use_async and aiohttp is not None:
            try:
                return asyncio.run(self.download_images_async(urls, output_dir))
            except Exception as e:
                logger.warning("Async download failed, falling back to sync: %s", e)
        return self.download_images_sync(urls, output_dir)

    def chapter_to_pdf(self, chapter_ref: ChapterRef, output_dir: pathlib.Path, zero_pad: int = 3) -> Tuple[pathlib.Path, int]:
        output_dir = ensure_dir(output_dir)
        img_urls = self.fetch_chapter_image_urls(chapter_ref.url)
        logger.info("Downloading %d images for %s", len(img_urls), chapter_ref.title)
        imgs = self.download_images(img_urls, output_dir)
        pdf_bytes = save_images_to_pdf_bytes(imgs)
        chap_num = chapter_ref.chapter_number or chapter_ref.index
        pdf_name = f"Chapter_{chap_num:0{zero_pad}d}.pdf"
        pdf_path = (output_dir / pdf_name).resolve()
        with pdf_path.open("wb") as fh:
            fh.write(pdf_bytes)
            fh.flush()
            os.fsync(fh.fileno())
        logger.info("Saved PDF: %s (pages=%d)", str(pdf_path), len(img_urls))
        return pdf_path, len(img_urls)

    def search_titles(self, query: str, limit: int = 10) -> List[Tuple[str, str]]:
        url = f"https://mangapark.com/search?word={requests.utils.requote_uri(query)}"
        logger.info("Searching MangaPark for: %s", query)
        resp = self.http.get(url)
        html = resp.text
        soup = BeautifulSoup(html, "html.parser")
        results: List[Tuple[str, str]] = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/title/" in href:
                title = a.get_text(strip=True)
                if not title:
                    title = a.get("title") or href.split("/")[-1]
                full = requests.compat.urljoin(url, href)
                if (title, full) not in results:
                    results.append((title, full))
            if len(results) >= limit:
                break
        logger.info("Found %d results", len(results))
        return results

    def search_and_select_title(self, query: str) -> Tuple[Optional[str], List[ChapterRef]]:
        results = self.search_titles(query)
        if not results:
            print(f"[ERROR] No titles found for query: {query}")
            return None, []
        for idx, (title, url) in enumerate(results, start=1):
            print(f"  {idx}: {title}\n     {url}")
        while True:
            try:
                choice = input("\nEnter the number of the manga you want (or press Enter for 1): ").strip()
                if not choice:
                    choice = "1"
                choice = int(choice)
                if 1 <= choice <= len(results):
                    selected_title, selected_url = results[choice - 1]
                    break
                else:
                    print("Invalid selection. Try again.")
            except ValueError:
                print("Invalid input. Enter a number.")
        title, chapters = self.fetch_all_chapters(selected_url)
        return title, chapters

    def scrape(self, url_or_term: str, destination: pathlib.Path, select_tokens: Optional[List[str]] = None, no_async: bool = False):
        resolved = url_or_term
        # If it's not a URL, run search and selection flow
        if not re.match(r"^https?://", url_or_term):
            title, chapters = self.search_and_select_title(url_or_term)
            if not title:
                return
            title_url = None  # we don't need the url here
        else:
            title_url = url_or_term
            title, chapters = self.fetch_all_chapters(title_url)

        safe_title = sanitize_filename(title)
        base_dir = ensure_dir(destination / safe_title)
        logger.info("Using output folder: %s", base_dir)

        # display chapters: show index and chapter_number if available
        print(f"Title: {title}")
        if title_url:
            print(f"Source: {title_url}")
        max_index = len(chapters)
        for ch in chapters:
            num_display = f" (#{ch.chapter_number})" if ch.chapter_number is not None else ""
            print(f"{ch.index:3d}: {ch.title}{num_display} — {ch.url}")

        # determine selection tokens
        if not select_tokens:
            sel = input("Select chapters (index, range, chapter-number, title substring, 'all'): ").strip()
            if not sel:
                print("No selection provided — exiting.")
                return
            tokens = parse_range_selection(sel)
        else:
            tokens = select_tokens

        # expand tokens into a list of chapter indices (based on index, chapter_number, or title match)
        selected_indices: List[int] = []
        for t in tokens:
            if t.lower() in ("all", "*"):
                selected_indices = [c.index for c in chapters]
                break
            # range like 1-5 (could be index or chapter-number)
            m_range = re.match(r"^(\d+)\s*-\s*(\d+)$", t)
            if m_range:
                a = int(m_range.group(1))
                b = int(m_range.group(2))
                if a > b:
                    a, b = b, a
                # we try to map range as index-range first; if out of bounds we'll attempt chapter-number mapping
                if 1 <= a <= max_index and 1 <= b <= max_index:
                    selected_indices.extend(range(a, b + 1))
                    continue
                # otherwise treat as chapter-number range: find chapters with chapter_number inside range
                for c in chapters:
                    if c.chapter_number is not None and a <= c.chapter_number <= b:
                        selected_indices.append(c.index)
                continue
            # single integer: index or chapter-number
            if re.fullmatch(r"\d+", t):
                n = int(t)
                # prefer direct index if valid
                if 1 <= n <= max_index:
                    selected_indices.append(n)
                    continue
                # else try matching as chapter_number
                mapped = [c.index for c in chapters if c.chapter_number == n]
                if mapped:
                    selected_indices.extend(mapped)
                    continue
                # else try approximate: match chapter_number that equals integer part
                mapped2 = [c.index for c in chapters if c.chapter_number is not None and int(c.chapter_number) == n]
                if mapped2:
                    selected_indices.extend(mapped2)
                    continue
                logger.warning("Couldn't map numeric selection '%s' to an index or chapter number; ignored", t)
                continue
            # non-numeric token: match title substring (case-insensitive)
            tlower = t.lower()
            matches = [c.index for c in chapters if tlower in c.title.lower()]
            if matches:
                selected_indices.extend(matches)
                continue
            # last-resort: allow user to pass something like 'Ch.138' - try to extract number
            m_num_in_text = re.search(r"(\d+)", t)
            if m_num_in_text:
                num = int(m_num_in_text.group(1))
                mapped = [c.index for c in chapters if c.chapter_number == num]
                if mapped:
                    selected_indices.extend(mapped)
                    continue
            logger.warning("Token '%s' didn't match index, chapter-number, or title substring.", t)

        # unique & sorted
        selected_indices = sorted(set(selected_indices))

        if not selected_indices:
            print("No chapters matched your selection. Exiting.")
            return

        zero_pad = max(3, len(str(len(chapters))))
        scraped_list = []

        for i_count, idx in enumerate(selected_indices, start=1):
            chapter = next((c for c in chapters if c.index == idx), None)
            if not chapter:
                logger.warning("Chapter index %d not found; skipping", idx)
                continue
            logger.info("Downloading Chapter %d/%d — %s", i_count, len(selected_indices), chapter.title)
            try:
                pdf_path, pages = self.chapter_to_pdf(chapter, base_dir, zero_pad=zero_pad)
                scraped_list.append(
                    {
                        "chapter_index": chapter.index,
                        "chapter_title": chapter.title,
                        "pdf_file": str(pdf_path.name),
                        "pages": pages,
                        "downloaded_at": now_iso(),
                        "source_url": chapter.url,
                    }
                )
            except Exception as e:
                logger.exception("Failed to download chapter %s: %s", chapter.title, e)

        logger.info("Skipping metadata save (disabled by user request)")
        logger.info("Scrape finished for %s", title)


# ---------------------------
# CLI
# ---------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mangapark_scraper", description="MangaPark scraper — download chapters to per-title folders and compile into PDFs.")
    p.add_argument("input", nargs=1, help="Mangapark URL or a search term (title).")
    p.add_argument("-o", "--output", default=".", help="Output directory (default: current folder).")
    p.add_argument("-d", "--delay", type=float, default=DEFAULT_THROTTLE, help=f"Throttle delay in seconds between requests (default {DEFAULT_THROTTLE}).")
    p.add_argument("-r", "--retries", type=int, default=DEFAULT_MAX_RETRIES, help=f"Maximum retries on transient failures (default {DEFAULT_MAX_RETRIES}).")
    p.add_argument("-w", "--workers", type=int, default=DEFAULT_WORKERS, help=f"Concurrent image downloads (aiohttp) workers (default {DEFAULT_WORKERS}).")
    p.add_argument("-c", "--chapters", default=None, help="Optional pre-selection: e.g. '1-3,5' or '138' or 'Black Wolf' or 'all'.")
    p.add_argument("--no-async", action="store_true", help="Disable async aiohttp downloads and use synchronous requests instead.")
    p.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    cfg = Config(throttle=args.delay, max_retries=args.retries, workers=args.workers)
    scraper = MangaParkScraper(cfg)
    input_val = args.input[0]
    out_dir = pathlib.Path(args.output).expanduser().resolve()
    select_tokens = None
    if args.chapters:
        select_tokens = parse_range_selection(args.chapters)
    try:
        scraper.scrape(input_val, out_dir, select_tokens, no_async=args.no_async)
    except Exception as e:
        logger.exception("Fatal error while scraping: %s", e)
        sys.exit(2)


if __name__ == "__main__":
    main()
