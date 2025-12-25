import argparse
import concurrent.futures
import io
import json
import logging
import os
import re
import sys
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from PIL import Image


if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

from dashscope_vision import DashscopeVisionClient
from progress_utils import ProgressTracker

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore


SEC_ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"
DEFAULT_USER_AGENT = os.environ.get(
    "SEC_PHOTO_USER_AGENT",
    "Academic Research (scrapephoto pipeline)",
)
DEFAULT_TIMEOUT = 40
MIN_IMAGE_EDGE = 80
MAX_DOWNLOAD_PER_FILING = 25
KEYWORD_WINDOW_TEXT_LENGTH = 1200
FALLBACK_IMAGE_LIMIT = 12
SEC_RATE_LIMIT = 10
KEYWORD_LINE_LOOKBACK = 40
CEO_NAME_CONTEXT_LINES = 40
GIF_MIN_PIXEL_AREA = 10_000
DASHSCOPE_BATCH_SIZE = 1
MIN_OPENCV_FACE_CONFIDENCE = 0.7

NAME_TITLE_PREFIXES = [
    "mr",
    "ms",
    "mrs",
    "miss",
    "dr",
    "sir",
    "madam",
    "madame",
    "chairman",
    "chairwoman",
    "chairperson",
]

PHOTO_COLUMN_NAME = "photo"


@dataclass
class CandidateImage:
    url: str
    score: float
    stage: str
    context: str = ""
    bytes_data: Optional[bytes] = field(default=None, repr=False)
    reason: str = ""
    photo_prominence: Optional[int] = None
    position_index: int = 0  # Position in HTML (0 = first image)
    estimated_size: int = 0  # Estimated size in pixels (width * height from HTML attributes)


@dataclass
class PhotoResult:
    cik10: str
    fyear: int
    accession: Optional[str]
    ceo: str
    status: str
    detail: str = ""
    sec_local_path: Optional[Path] = None
    shared_local_path: Optional[Path] = None
    source_url: Optional[str] = None
    stage: Optional[str] = None
    photo_prominence: int = 1
    position_index: int = 0
    estimated_size: int = 0


def sanitize_company_name(name: str) -> str:
    clean = re.sub(r"[^\w]+", " ", name or "")
    return re.sub(r"\s+", " ", clean).strip().lower()


def build_cik_directory_lookup(sec_root: Path) -> Dict[str, List[Path]]:
    """
    Pre-scan the SEC data directory so we can map cik10 to possible company folders.
    """
    mapping: Dict[str, List[Path]] = {}
    for entry in sec_root.iterdir():
        if not entry.is_dir():
            continue
        parts = entry.name.rsplit("_", maxsplit=1)
        if len(parts) != 2:
            continue
        cik = parts[1]
        if len(cik) != 10 or not cik.isdigit():
            continue
        mapping.setdefault(cik, []).append(entry)
    return mapping


def name_variants(full_name: str) -> List[str]:
    """
    Generate a set of name variants for locating CEO references in HTML content.
    """
    tokens = [token for token in re.split(r"[\s,.;/]+", full_name.strip()) if token]
    if not tokens:
        return []
    variants: List[str] = [full_name]

    suffixes = {
        "JR",
        "SR",
        "II",
        "III",
        "IV",
        "V",
        "VI",
        "VII",
        "VIII",
        "IX",
        "X",
    }

    core_tokens = [token for token in tokens if token.upper() not in suffixes]
    if len(core_tokens) >= 2:
        variants.append(f"{core_tokens[0]} {core_tokens[-1]}")
        middle_parts = core_tokens[1:-1]
        if middle_parts:
            middle_initial = middle_parts[0][0]
            variants.append(f"{core_tokens[0]} {middle_initial}. {core_tokens[-1]}")
    else:
        variants.append(core_tokens[0])

    unique_variants = []
    seen = set()
    for item in variants:
        normalized = item.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_variants.append(item)

    return unique_variants


class RateLimiter:
    def __init__(self, max_requests: int, time_window: float = 1.0):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = Lock()

    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            while self.requests and self.requests[0] < now - self.time_window:
                self.requests.popleft()
            if len(self.requests) >= self.max_requests:
                sleep_time = self.time_window - (now - self.requests[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()
                    while self.requests and self.requests[0] < now - self.time_window:
                        self.requests.popleft()
            self.requests.append(time.time())


_sec_rate_limiter = RateLimiter(max_requests=SEC_RATE_LIMIT, time_window=1.0)


def ensure_request_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
    return session


def build_image_url(src: str, cik10: str, accession: str) -> str:
    if src.startswith("http://") or src.startswith("https://"):
        return src
    if src.startswith("//"):
        return f"https:{src}"

    cik_clean = cik10.lstrip("0") or "0"
    accession_clean = accession.replace("-", "")
    if src.startswith("/"):
        return f"{SEC_ARCHIVE_BASE}/{cik_clean}/{accession_clean}{src}"

    base = f"{SEC_ARCHIVE_BASE}/{cik_clean}/{accession_clean}/"
    return f"{base}{src}"


def read_html_file(html_path: Path) -> Optional[str]:
    if not html_path.exists():
        return None
    try:
        data = html_path.read_bytes()
    except OSError:
        return None

    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding, errors="ignore")
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def extract_context_text(img_tag: Tag, max_length: int = KEYWORD_WINDOW_TEXT_LENGTH) -> str:
    """
    Collect nearby text around an image node to evaluate relevance.
    """
    parts: List[str] = []

    # Include alt and title attributes of the image itself
    for attr in ("alt", "title"):
        value = img_tag.get(attr)
        if value:
            parts.append(value)

    # Traverse parent and siblings, also check parent attributes
    parent = img_tag.parent
    steps = 0
    while parent and steps < 5:
        # Check parent's id, title, class attributes
        for attr in ("id", "title", "class"):
            value = parent.get(attr)
            if value:
                if isinstance(value, list):
                    parts.append(" ".join(str(v) for v in value))
                else:
                    parts.append(str(value))
        
        # Extract text content
        texts = list(_iter_text(parent))
        if texts:
            parts.append(" ".join(texts))
        parent = parent.parent
        steps += 1

    combined = " ".join(parts)
    return combined[:max_length]


def verify_ceo_context(
    img_tag: Tag, 
    ceo_name: str, 
    strict_mode: bool = False,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Check whether nearby text includes the CEO name or title.

    Args:
        img_tag: Image tag.
        ceo_name: CEO full name.
        strict_mode: If True, use stricter title matching.
        logger: Optional logger for debug output.

    Returns:
        True if CEO-related context is found, else False.
    """

    context = extract_context_text(img_tag, max_length=2000)
    context_lower = context.lower()
    

    ceo_titles_loose = [

        "chief executive officer",
        "ceo",
        "president and ceo",
        "president & ceo",
        "chief executive",
        "chairman and ceo",
        "chairman & ceo",

        "chairman of the board",
        "chairman",
        "chairman of board",
        "board chairman",

        "executive chair",
        "chair executive",

        "president",
        "president and",

        "executive officer",
        "executive",
    ]
    

    ceo_titles_strict = [
        "chief executive officer",
        "ceo",
        "president and ceo",
        "president & ceo",
        "chief executive",
        "chairman and ceo",
        "chairman & ceo",
        "chairman of the board",
        "chairman",
    ]
    

    ceo_titles = ceo_titles_strict if strict_mode else ceo_titles_loose
    

    has_title = any(title in context_lower for title in ceo_titles)
    

    has_name = False
    if ceo_name:

        ceo_variants = name_variants(ceo_name)
        

        for variant in ceo_variants:
            if not variant:
                continue
            variant_lower = variant.lower().strip()
            

            if variant_lower in context_lower:
                has_name = True
                break
            

            name_parts = variant_lower.split()
            if len(name_parts) > 0:
                last_name = name_parts[-1]

                if len(last_name) >= 3 and last_name in context_lower:
                    has_name = True
                    if logger:
                        logger.debug("Matched last name: %s", last_name)
                    break
            

            if len(name_parts) > 1:
                first_name = name_parts[0]

                if len(first_name) >= 3 and first_name in context_lower:
                    has_name = True
                    if logger:
                        logger.debug("Matched first name: %s", first_name)
                    break
        

        if not has_name and len(ceo_variants) > 0:
            full_name = ceo_variants[0].lower()
            name_parts = full_name.split()
            if len(name_parts) >= 2:

                first = name_parts[0]
                last = name_parts[-1]

                if len(first) >= 2 and len(last) >= 3:
                    if first in context_lower and last in context_lower:
                        has_name = True
                        if logger:
                            logger.debug("Matched first+last separately: %s %s", first, last)
    
    if logger:
        logger.debug(
            "CEO context check (strict=%s): has_title=%s, has_name=%s (ceo='%s')",
            strict_mode,
            has_title,
            has_name,
            ceo_name,
        )
    


    return has_title or has_name


def _iter_text(node: Tag) -> Iterable[str]:
    for element in node.descendants:
        if isinstance(element, NavigableString):
            text = str(element).strip()
            if text:
                yield text





def estimate_image_size(img_tag: Tag) -> int:
    """
    Estimate image size from HTML attributes (width * height).
    Returns 0 if size cannot be estimated.
    """
    width = 0
    height = 0
    
    # Try to get width and height from attributes
    width_attr = img_tag.get("width")
    height_attr = img_tag.get("height")
    
    if width_attr:
        try:
            width = int(float(str(width_attr).replace("px", "").strip()))
        except (ValueError, TypeError):
            pass
    
    if height_attr:
        try:
            height = int(float(str(height_attr).replace("px", "").strip()))
        except (ValueError, TypeError):
            pass
    
    # Try to get from style attribute
    if width == 0 or height == 0:
        style = img_tag.get("style", "")
        if style:
            width_match = re.search(r"width\s*:\s*(\d+)", style, re.IGNORECASE)
            height_match = re.search(r"height\s*:\s*(\d+)", style, re.IGNORECASE)
            if width_match and width == 0:
                width = int(width_match.group(1))
            if height_match and height == 0:
                height = int(height_match.group(1))
    
    return width * height if width > 0 and height > 0 else 0


def estimate_image_dimensions(img_tag: Tag) -> Tuple[int, int]:
    """
    Estimate image width/height from HTML attributes or style.
    Returns (width, height); falls back to (0, 0) if unknown.
    """
    width = 0
    height = 0
    
    width_attr = img_tag.get("width")
    height_attr = img_tag.get("height")
    
    if width_attr:
        try:
            width = int(float(str(width_attr).replace("px", "").strip()))
        except (ValueError, TypeError):
            pass
    
    if height_attr:
        try:
            height = int(float(str(height_attr).replace("px", "").strip()))
        except (ValueError, TypeError):
            pass
    
    if width == 0 or height == 0:
        style = img_tag.get("style", "")
        if style:
            width_match = re.search(r"width\s*:\s*(\d+)", str(style), re.IGNORECASE)
            height_match = re.search(r"height\s*:\s*(\d+)", str(style), re.IGNORECASE)
            if width_match and width == 0:
                width = int(width_match.group(1))
            if height_match and height == 0:
                height = int(height_match.group(1))
    
    return (width, height)


def is_extreme_aspect_ratio(img_tag: Tag, max_ratio: float = 4.0) -> bool:
    """
    Detect extreme aspect ratios (e.g., thin separators) to filter non-portraits.

    Args:
        img_tag: Image tag.
        max_ratio: Maximum allowed aspect ratio (width/height or height/width).

    Returns:
        True if ratio exceeds the threshold; otherwise False.
    """
    width, height = estimate_image_dimensions(img_tag)
    

    if width <= 0 or height <= 0:
        return False
    

    ratio = max(width / height, height / width)
    
    return ratio > max_ratio


def is_small_gif_image(url: str, pixel_area: int) -> bool:
    """Heuristic: filter tiny GIFs that are likely decorative icons."""
    return url.lower().endswith(".gif") and 0 < pixel_area < GIF_MIN_PIXEL_AREA





def find_ceo_letter_blocks(
    soup: BeautifulSoup,
    logger: Optional[logging.Logger] = None,
) -> List[Tag]:
    """
    Identify layout blocks that likely belong to the CEO letter.

    Strategy:
    1. Locate strong opening phrases (e.g., "to our shareholders").
    2. Find the full parent block containing that phrase.
    3. Search upward to include headers, since photos may sit above the phrase.
    """

    opening_phrases = [

        "to our stockholders",
        "to our shareholders",
        "to the stockholders",
        "to the shareholders",
        "dear fellow stockholders",
        "dear fellow shareholders",
        "dear stockholders",
        "dear shareholders",
        "letter to stockholders",
        "letter to shareholders",
        "letter to the stockholders",
        "letter to the shareholders",

        "to our valued stockholders",
        "to our valued shareholders",
        "to our shareholders and employees",
        "to our shareholders and customers",
        "shareholders letter",
        "stockholders letter",
        "letter from the chairman",
        "letter from the ceo",
        "letter from the president",

        "dear shareholder",
        "dear stockholder",
        "to shareholders",
        "to stockholders",
        "fellow shareholders",
        "fellow stockholders",
            "message from the ceo",
            "ceo message",
            "ceo's message",
            "message to shareholders",
            "message from our ceo",
    ]
    

    opening_nodes = []
    for phrase in opening_phrases:
        phrase_lower = phrase.lower()
        for text_node in soup.find_all(string=lambda t: t and phrase_lower in str(t).strip().lower()):
            opening_nodes.append((text_node, phrase))
    
    if not opening_nodes:
        if logger:
            logger.debug("No CEO letter opening phrases found")
        return []
    
    if logger:
        logger.info("Found %d CEO letter opening phrases", len(opening_nodes))
    

    container_blocks: List[Tag] = []
    seen_containers: set[int] = set()
    
    for opening_node, phrase in opening_nodes:
        parent = opening_node.parent
        steps = 0
        best_container = None
        best_container_score = 0
        

        while parent and steps < 20:
            if hasattr(parent, "name") and parent.name:

                if parent.name in ("div", "section", "article", "body", "td", "table"):
                    try:
                        img_tags = parent.find_all("img", recursive=True)
                        
                        if len(img_tags) > 0:

                            text_content = " ".join(_iter_text(parent))
                            text_length = len(text_content)
                            




                            score = len(img_tags) * 10
                            
                            if 200 <= text_length <= 8000:
                                score += 20
                            elif text_length < 100 or text_length > 15000:
                                score -= 5
                            

                            if score > best_container_score:
                                best_container = parent
                                best_container_score = score
                        
                    except Exception:
                        pass
            
            parent = getattr(parent, "parent", None)
            steps += 1
    

        if best_container is not None and id(best_container) not in seen_containers:
            seen_containers.add(id(best_container))
            container_blocks.append(best_container)
            if logger:
                img_count = len(best_container.find_all("img", recursive=True))
                text_len = len(" ".join(_iter_text(best_container)))
                logger.info(
                    "✓ Found CEO letter block: <%s> with %d images, %d chars (phrase: '%s', score: %d)",
                    best_container.name,
                    img_count,
                    text_len,
                    phrase,
                    best_container_score,
                )
    
    if logger:
        logger.info("Found %d CEO letter blocks total", len(container_blocks))
    
    return container_blocks


def extract_images_from_blocks(
    blocks: List[Tag],
    soup: BeautifulSoup,
    cik10: str,
    accession: str,
    logger: Optional[logging.Logger] = None,
) -> List[CandidateImage]:
    """
    Extract images from DOM blocks and sort by position then size.
    """
    if not blocks:
        return []
    
    candidates: List[CandidateImage] = []
    seen_urls: set[str] = set()
    

    all_imgs = soup.find_all("img")
    img_to_index = {id(img): idx for idx, img in enumerate(all_imgs)}
    
    MIN_SIZE = 8_000
    
    for block in blocks:
        block_imgs = block.find_all("img", recursive=True)
        
        for img in block_imgs:
            src = img.get("src")
            if not src:
                continue
            
            url = build_image_url(src, cik10, accession)
            if url in seen_urls:
                continue
            seen_urls.add(url)
            
            position_index = img_to_index.get(id(img), len(all_imgs))
            estimated_size = estimate_image_size(img)
            

            if estimated_size > 0 and estimated_size < MIN_SIZE:
                if logger:
                    logger.debug("Skipping small image: %s (size=%d)", src, estimated_size)
                continue
            

            if is_extreme_aspect_ratio(img, max_ratio=4.0):
                if logger:
                    logger.debug("Skipping extreme aspect ratio image: %s", src)
                continue
            
            candidates.append(CandidateImage(
                url=url,
                score=0.0,
                stage="ceo_letter_block",
                context="",
                position_index=position_index,
                estimated_size=estimated_size,
            ))
            
            if logger:
                logger.debug(
                    "Extracted image: %s (position=%d, size=%d)",
                    src,
                    position_index,
                    estimated_size,
                )
    




    candidates.sort(key=lambda x: (x.position_index, -x.estimated_size))
    
    if logger:
        logger.info(
            "Extracted %d images from CEO letter blocks (sorted by position, then size)",
            len(candidates),
        )
    
    return candidates


def get_early_document_images(
    soup: BeautifulSoup,
    cik10: str,
    accession: str,
    limit: int = 15,
    logger: Optional[logging.Logger] = None,
) -> List[CandidateImage]:
    """
    Collect early-document images (CEO photos often appear near the start).
    """
    all_imgs = soup.find_all("img")
    
    if not all_imgs:
        return []
    

    top_imgs = all_imgs[:limit]
    
    candidates: List[CandidateImage] = []
    seen_urls: set[str] = set()
    
    MIN_SIZE = 10_000
    
    for idx, img in enumerate(top_imgs):

        if idx == 0:
            continue
        
        src = img.get("src")
        if not src:
            continue
        
        url = build_image_url(src, cik10, accession)
        if url in seen_urls:
            continue
        seen_urls.add(url)
        
        estimated_size = estimate_image_size(img)
        

        if estimated_size > 0 and estimated_size < MIN_SIZE:
            continue
        

        if is_extreme_aspect_ratio(img, max_ratio=4.0):
            continue
        
        candidates.append(CandidateImage(
            url=url,
            score=0.0,
            stage="early_images",
            context="",
            position_index=idx,
            estimated_size=estimated_size,
        ))
        

    candidates.sort(key=lambda x: (-x.estimated_size, x.position_index))
    
    if logger:
        logger.info("Found %d early images (positions 1-%d)", len(candidates), limit)
    
    return candidates





def load_image_bytes(session: requests.Session, url: str) -> Optional[bytes]:
    if "sec.gov" in url or "www.sec.gov" in url:
        _sec_rate_limiter.wait_if_needed()
    try:
        resp = session.get(url, timeout=DEFAULT_TIMEOUT)
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    return resp.content


def load_images_batch(
    session: requests.Session, 
    urls: List[str], 
    max_workers: int = 5
) -> Dict[str, Optional[bytes]]:
    """
    Download multiple images in parallel (I/O bound).

    Args:
        session: requests session
        urls: List of image URLs.
        max_workers: Max worker threads (default 5).

    Returns:
        Mapping of URL -> image bytes (None on failure).
    """
    results = {}
    
    def download_one(url: str) -> Tuple[str, Optional[bytes]]:
        return (url, load_image_bytes(session, url))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_one, url): url for url in urls}
        
        for future in concurrent.futures.as_completed(futures):
            url, image_bytes = future.result()
            results[url] = image_bytes
    
    return results


def basic_image_checks(image_bytes: bytes) -> Tuple[bool, float, str, Tuple[int, int]]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
            if min(width, height) < MIN_IMAGE_EDGE:
                return False, 0.1, "too_small", (width, height)
            ratio = width / height if height else 0
            if ratio < 0.4 or ratio > 1.8:
                return False, 0.2, "aspect_ratio_out_of_range", (width, height)
            img.load()
    except Exception as exc:
        return False, 0.0, f"pillow_error:{exc}", (0, 0)
    return True, 0.5, "basic_checks_pass", (width, height)


def basic_image_checks_batch(
    images_data: List[Tuple[str, bytes]], 
    max_workers: int = 4
) -> Dict[str, Tuple[bool, float, str, Tuple[int, int]]]:
    """
    Run basic image checks in parallel (CPU bound).

    Args:
        images_data: List of (url, image_bytes).
        max_workers: Max worker processes (default 4).

    Returns:
        Mapping of URL -> basic check result.
    """
    results = {}
    
    def check_one(url_and_bytes: Tuple[str, bytes]) -> Tuple[str, Tuple[bool, float, str, Tuple[int, int]]]:
        url, image_bytes = url_and_bytes
        return (url, basic_image_checks(image_bytes))
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_one, item): item[0] for item in images_data}
        
        for future in concurrent.futures.as_completed(futures):
            url, check_result = future.result()
            results[url] = check_result
    
    return results


def detect_face_with_cv(
    image_bytes: bytes,
) -> Tuple[bool, float, str, Optional[List[Tuple[int, int, int, int]]]]:
    if cv2 is None:
        return False, 0.0, "cv2_not_available", None

    # Check if cv2 has required attributes
    if not hasattr(cv2, "imdecode"):
        return False, 0.0, "cv2_imdecode_not_available", None
    if not hasattr(cv2, "IMREAD_COLOR"):
        return False, 0.0, "cv2_IMREAD_COLOR_not_available", None

    try:
        ensure_numpy_import()
    except ImportError:
        return False, 0.0, "numpy_not_available", None

    try:
        array = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8),  # type: ignore[name-defined]
            cv2.IMREAD_COLOR,
        )
    except AttributeError as exc:
        return False, 0.0, f"cv2_attribute_error:{exc}", None
    except Exception as exc:  # pragma: no cover
        return False, 0.0, f"cv2_decode_error:{exc}", None

    if array is None:
        return False, 0.0, "cv2_decode_none", None

    gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    classifier_path = getattr(cv2.data, "haarcascades", "")
    cascade_file = os.path.join(classifier_path, "haarcascade_frontalface_default.xml")
    if not os.path.exists(cascade_file):
        return False, 0.0, "cv2_cascade_missing", None

    classifier = cv2.CascadeClassifier(cascade_file)

    faces = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    if len(faces) == 0:
        return False, 0.3, "no_face_detected", []

    height, width = gray.shape
    areas = [(w * h) / (width * height) for (_, _, w, h) in faces]
    best_area = max(areas)
    return True, min(0.9, 0.6 + best_area), "face_detected", [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def detect_faces_batch(
    images_data: List[Tuple[str, bytes]], 
    max_workers: int = 4
) -> Dict[str, Tuple[bool, float, str, Optional[List[Tuple[int, int, int, int]]]]]:
    """
    Detect faces across many images in parallel (CPU bound).

    Args:
        images_data: List of (url, image_bytes).
        max_workers: Max worker threads (default 4; cv2 releases the GIL).

    Returns:
        Mapping of URL -> face detection result.
    """
    results = {}
    
    def detect_one(url_and_bytes: Tuple[str, bytes]) -> Tuple[str, Tuple[bool, float, str, Optional[List[Tuple[int, int, int, int]]]]]:
        url, image_bytes = url_and_bytes
        return (url, detect_face_with_cv(image_bytes))
    

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(detect_one, item): item[0] for item in images_data}
        
        for future in concurrent.futures.as_completed(futures):
            url, face_result = future.result()
            results[url] = face_result
    
    return results


def ensure_numpy_import() -> None:
    """
    Lazy import numpy for OpenCV face detection. This keeps numpy optional in case
    the user does not install cv2.
    """
    global np  # type: ignore
    if "np" in globals():
        return
    import numpy as np  # type: ignore  # noqa: F401

def calculate_photo_prominence(
    image_size: Tuple[int, int],
    face_boxes: Optional[List[Tuple[int, int, int, int]]],
) -> int:
    width, height = image_size
    if width <= 0 or height <= 0:
        return 1

    total_area = width * height
    if face_boxes is None:
        if min(width, height) >= 450 and total_area >= 300_000:
            return 3
        return 2

    if len(face_boxes) == 0:
        return 2

    if len(face_boxes) > 1:
        return 2

    _, _, face_w, face_h = face_boxes[0]
    face_ratio = (face_w * face_h) / total_area
    if face_ratio >= 0.4:
        return 4
    return 3


class CEOPhotoPipeline:
    STRATEGY_KEYWORD_NEAR_PHRASE = 1
    STRATEGY_CEO_NAME_CONTEXT = 2
    STRATEGY_CEO_LETTER_BLOCK = 3
    STRATEGY_EARLY_DOCUMENT = 4
    STRATEGY_HISTORICAL_FALLBACK = 5
    STRATEGY_SEQUENTIAL_FALLBACK = 6

    def __init__(
        self,
        csv_path: Path,
        sec_data_root: Path,
        shared_output_root: Path,
        results_output_path: Path,
        dashscope_client: Optional[DashscopeVisionClient] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        verbose: bool = False,
        use_memory: bool = True,
        strategy_limit: Optional[int] = None,
        skip_existing_companies: bool = False,
        only_2021_year: bool = False,
        strategy_0_5_only: bool = False,
        brute_force_strategy: bool = False,
    ) -> None:
        self.csv_path = csv_path
        self.sec_data_root = sec_data_root
        self.shared_output_root = shared_output_root
        self.results_output_path = results_output_path
        self.dashscope_client = dashscope_client or DashscopeVisionClient()
        self.limit = limit
        self.offset = offset
        self.year_start = year_start
        self.year_end = year_end
        self.verbose = verbose
        self.session = ensure_request_session()
        self.logger = logging.getLogger("ceo_photo_pipeline")
        self.filter_stats: Counter[str] = Counter()
        self.dashscope_batch_size = DASHSCOPE_BATCH_SIZE
        self.strategy_limit = strategy_limit
        self.skip_existing_companies = skip_existing_companies
        self.only_2021_year = only_2021_year
        self.strategy_0_5_only = strategy_0_5_only
        self.brute_force_strategy = brute_force_strategy
        logging.getLogger("PIL").setLevel(logging.INFO)

        self.shared_output_root.mkdir(parents=True, exist_ok=True)

    def _record_filter_event(self, key: str) -> None:
        """Track hit/reject counts for each filter stage."""
        self.filter_stats[key] += 1

    def _is_strategy_enabled(self, strategy_order: int) -> bool:
        """Return whether a strategy (by 1-based order) is allowed under the limit."""
        return self.strategy_limit is None or strategy_order <= self.strategy_limit

    def _maybe_flush_dashscope_queue(
        self,
        queue: List[Tuple[CandidateImage, bytes, str, str]],
        stage_hint: str,
        force: bool = False,
    ) -> Optional[CandidateImage]:
        """Trigger DashScope batch call when queue is full or forced."""
        if not queue:
            return None
        if not force and len(queue) < self.dashscope_batch_size:
            return None
        return self._flush_dashscope_queue(queue, stage_hint)

    def _flush_dashscope_queue(
        self,
        queue: List[Tuple[CandidateImage, bytes, str, str]],
        stage_hint: str,
    ) -> Optional[CandidateImage]:
        items = list(queue)
        queue.clear()
        if not items:
            return None

        payload = [(img_bytes, ceo_name) for (_, img_bytes, ceo_name, _) in items]
        results = list(self.dashscope_client.analyze_portraits_batch(payload)) if self.dashscope_client.is_ready() else []
        if len(results) < len(items):
            missing = len(items) - len(results)
            results.extend(
                {
                    'is_person': False,
                    'confidence': 0.0,
                    'reason': 'dashscope_no_response',
                }
                for _ in range(missing)
            )

        for (candidate, _, _ceo_name, base_reason), result in zip(items, results):
            is_person = bool(result.get('is_person'))
            confidence = float(result.get('confidence', 0.0))
            reason = result.get('reason', '')

            if is_person:
                candidate.reason = f"{base_reason}|dashscope_confirmed|prominence:{candidate.photo_prominence}"
                if self.verbose:
                    self.logger.info(
                        "✓✓✓ DashScope confirmed (%s): conf=%.2f, reason=%s",
                        stage_hint,
                        confidence,
                        reason,
                    )
                self._record_filter_event("dashscope_pass")
                self._record_filter_event("candidate_selected")
                return candidate

            if self.verbose:
                self.logger.info(
                    "❌ DashScope rejected (%s): %s",
                    stage_hint,
                    reason,
                )
            candidate.reason = f"{base_reason}|dashscope_reject:{reason}"
            self._record_filter_event("dashscope_reject")

        return None

    def _should_skip_small_gif(self, url: str, pixel_area: int) -> bool:
        """Skip tiny GIFs to avoid wasted detection work."""
        if not is_small_gif_image(url, pixel_area):
            return False
        if self.verbose:
            self.logger.debug(
                "Skipping small GIF (<%d px): %s (area=%d)",
                GIF_MIN_PIXEL_AREA,
                url,
                pixel_area,
            )
        self._record_filter_event("gif_small_filtered")
        return True

    def run(self) -> None:
        df = pd.read_csv(self.csv_path)
        df["cik"] = pd.to_numeric(df["cik"], errors="coerce")
        df = df.dropna(subset=["cik", "fyear"])
        df["cik10"] = df["cik"].astype(int).astype(str).str.zfill(10)
        df["fyear"] = df["fyear"].astype(int)


        if self.year_start is not None:
            df = df[df["fyear"] >= self.year_start]
            if self.verbose:
                self.logger.info("Filtered by year_start >= %d, remaining: %d", self.year_start, len(df))
        
        if self.year_end is not None:
            df = df[df["fyear"] <= self.year_end]
            if self.verbose:
                self.logger.info("Filtered by year_end <= %d, remaining: %d", self.year_end, len(df))

        mapping = build_cik_directory_lookup(self.sec_data_root)

        try:
            latest_years: Dict[str, int] = df.groupby('cik10')['fyear'].max().to_dict()
        except Exception:
            latest_years = {}
        results: List[PhotoResult] = []

        df_pairs = df.drop_duplicates(subset=["cik10", "fyear"])
        total_pairs = len(df_pairs)
        if total_pairs == 0:
            self.logger.warning("No records found for processing.")
            self._write_results_json(results)
            return

        # Apply offset: skip first N entries
        if self.offset > 0:
            if self.offset >= total_pairs:
                self.logger.warning(
                    "Offset %d is greater than or equal to total pairs %d. Nothing to process.",
                    self.offset,
                    total_pairs,
                )
                self._write_results_json(results)
                return
            df_pairs = df_pairs.iloc[self.offset:].reset_index(drop=True)
            if self.verbose:
                self.logger.info("Skipped first %d entries, starting from position %d", self.offset, self.offset)

        # Calculate effective total considering offset and limit
        remaining_after_offset = len(df_pairs)
        effective_total = (
            min(remaining_after_offset, int(self.limit))
            if self.limit is not None
            else remaining_after_offset
        )
        progress = ProgressTracker(effective_total, prefix="Extracting photos")

        processed = 0

        for _, row in df_pairs.iterrows():
            if self.limit is not None and processed >= self.limit:
                break

            cik10 = str(row["cik10"])
            fyear = int(row["fyear"])
            ceo_name = str(row.get("ceo_full_name", "")).strip()
            company_variants = [
                sanitize_company_name(str(row.get("primary_company_name", ""))),
                sanitize_company_name(str(row.get("compustat_name", ""))),
                sanitize_company_name(str(row.get("crsp_company_name", ""))),
            ]

            result = self.process_single_entry(cik10, fyear, ceo_name, company_variants, mapping, latest_years)
            results.append(result)
            processed += 1
            

            is_success = (result.status == "success")
            status_suffix = "✓" if is_success else "✗"
            progress.update(processed, suffix=status_suffix)

        progress.finish()

        success_total = sum(1 for item in results if item.status == "success")
        failure_total = len(results) - success_total
        self.logger.info(
            "Extraction summary: %d entries processed, %d succeeded, %d failed",
            len(results),
            success_total,
            failure_total,
        )

        if self.filter_stats:
            self.logger.info("Filter stage statistics (pass/fail counts):")
            for key, count in sorted(self.filter_stats.items()):
                self.logger.info("  %s: %d", key, count)

        self._write_results_json(results)

    def process_single_entry(
        self,
        cik10: str,
        fyear: int,
        ceo_name: str,
        company_variants: List[str],
        mapping: Dict[str, List[Path]],
        latest_years: Dict[str, int],
    ) -> PhotoResult:

        if self.only_2021_year:
            if fyear != 2021:
                if self.verbose:
                    self.logger.info(
                        "Skipping year %d for CIK %s (only_2021_year mode enabled)",
                        fyear,
                        cik10
                    )
                return PhotoResult(
                    cik10=cik10,
                    fyear=fyear,
                    accession=None,
                    ceo=ceo_name,
                    status="skipped",
                    detail="only_2021_year_mode",
                )
        

        if self.skip_existing_companies:


            has_any_photo = False
            try:



                if any(self.shared_output_root.glob(f"{cik10}_*.png")):
                    has_any_photo = True
            except Exception as e:
                if self.verbose:
                    self.logger.warning("Error checking for existing company photos: %s", e)
            
            if has_any_photo:
                if self.verbose:
                    self.logger.info(
                        "Skipping company %s (CIK %s) because photos already exist in pool",
                        company_variants[0] if company_variants else "Unknown",
                        cik10
                    )
                return PhotoResult(
                    cik10=cik10,
                    fyear=fyear,
                    accession=None,
                    ceo=ceo_name,
                    status="skipped",
                    detail="company_already_has_photos",
                )




        if self.brute_force_strategy:
            try:
                glob_pattern = f"{cik10}_*.png"
                existing_files = list(self.shared_output_root.glob(glob_pattern))
                if existing_files:

                    self.logger.info(
                        "BRUTE FORCE SKIP: Found existing photo(s) for CIK %s in pool: %s, skipping all years",
                        cik10,
                        [f.name for f in existing_files],
                    )
                    return PhotoResult(
                        cik10=cik10,
                        fyear=fyear,
                        accession=None,
                        ceo=ceo_name,
                        status="skipped",
                        detail="brute_force_pool_has_company_photo",
                    )
                else:
                    if self.verbose:
                        self.logger.info(
                            "BRUTE FORCE CHECK: No existing photo for CIK %s in pool (glob: %s)",
                            cik10,
                            glob_pattern,
                        )
            except Exception as e:

                self.logger.warning(
                    "BRUTE FORCE: Error checking existing photos for CIK %s: %s",
                    cik10,
                    e,
                )

            latest = latest_years.get(cik10)
            if latest is not None and fyear != int(latest):
                if self.verbose:
                    self.logger.info(
                        "BRUTE FORCE: Skipping CIK %s year %d (only run on latest year %d)",
                        cik10,
                        fyear,
                        latest,
                    )
                return PhotoResult(
                    cik10=cik10,
                    fyear=fyear,
                    accession=None,
                    ceo=ceo_name,
                    status="skipped",
                    detail="brute_force_not_latest_year",
                )


        existing_photo_path = self.shared_output_root / f"{cik10}_{fyear}.png"
        existing_photo_fill_path = self.shared_output_root / f"{cik10}_{fyear}_fill.png"
        
        if existing_photo_path.exists() and existing_photo_path.is_file():
            if self.verbose:
                self.logger.info(
                    "Photo already exists in pool: %s, skipping extraction",
                    existing_photo_path.name
                )
            return PhotoResult(
                cik10=cik10,
                fyear=fyear,
                accession=None,
                ceo=ceo_name,
                status="success",
                detail="already_exists_in_pool",
                shared_local_path=existing_photo_path,
            )
        
        if existing_photo_fill_path.exists() and existing_photo_fill_path.is_file():
            if self.verbose:
                self.logger.info(
                    "Filled photo already exists in pool: %s, skipping extraction",
                    existing_photo_fill_path.name
                )
            return PhotoResult(
                cik10=cik10,
                fyear=fyear,
                accession=None,
                ceo=ceo_name,
                status="success",
                detail="already_exists_in_pool_filled",
                shared_local_path=existing_photo_fill_path,
            )
        
        if not ceo_name:
            return PhotoResult(
                cik10=cik10,
                fyear=fyear,
                accession=None,
                ceo=ceo_name,
                status="skipped",
                detail="missing_ceo_name",
            )

        year_suffix = f"{fyear % 100:02d}"
        candidates_for_cik = mapping.get(cik10, [])
        if not candidates_for_cik:
            return PhotoResult(
                cik10=cik10,
                fyear=fyear,
                accession=None,
                ceo=ceo_name,
                status="missing_company_directory",
            )

        company_dir = self._choose_company_directory(candidates_for_cik, company_variants)
        if company_dir is None:
            return PhotoResult(
                cik10=cik10,
                fyear=fyear,
                accession=None,
                ceo=ceo_name,
                status="company_directory_not_matched",
            )

        filing_dir = company_dir / "DEF_14A"
        if not filing_dir.exists():
            return PhotoResult(
                cik10=cik10,
                fyear=fyear,
                accession=None,
                ceo=ceo_name,
                status="missing_def14a_directory",
            )

        target_accession = self._select_accession(filing_dir, year_suffix)
        if target_accession is None:
            return PhotoResult(
                cik10=cik10,
                fyear=fyear,
                accession=None,
                ceo=ceo_name,
                status="missing_accession_for_year",
            )

        html_path = self._find_primary_document(target_accession)
        if html_path is None:
            return PhotoResult(
                cik10=cik10,
                fyear=fyear,
                accession=target_accession.name,
                ceo=ceo_name,
                status="missing_primary_document",
            )

        html_content = read_html_file(html_path)
        if not html_content:
            return PhotoResult(
                cik10=cik10,
                fyear=fyear,
                accession=target_accession.name,
                ceo=ceo_name,
                status="empty_html_content",
            )

        try:
            soup = BeautifulSoup(html_content, "lxml")
        except Exception:
            soup = BeautifulSoup(html_content, "html.parser")
        
        all_imgs = soup.find_all("img")
        if self.verbose:
            self.logger.info(
                "Processing cik=%s year=%s, ceo='%s', accession=%s, found %d img tags",
                cik10,
                fyear,
                ceo_name,
                target_accession.name,
                len(all_imgs),
            )
        

        position_range = None
        


        candidates: List[CandidateImage] = []
        chosen = None
        keyword_found = False
        

        if self.brute_force_strategy:
            if self.verbose:
                self.logger.info("BRUTE FORCE MODE: Scanning all images (excluding cover) for faces...")
            
            chosen = self._brute_force_face_search(
                soup,
                cik10,
                target_accession.name,
                ceo_name=ceo_name,
            )
            
            if chosen is not None:

                final_check_path = self.shared_output_root / f"{cik10}_{fyear}.png"
                if final_check_path.exists():
                    self.logger.warning(
                        "BRUTE FORCE DUPLICATE PREVENTED: Photo already exists at %s, skipping save!",
                        final_check_path,
                    )
                    return PhotoResult(
                        cik10=cik10,
                        fyear=fyear,
                        accession=target_accession.name,
                        ceo=ceo_name,
                        status="success",
                        detail="already_exists_final_check",
                        shared_local_path=final_check_path,
                    )
                

                self.logger.info(
                    "BRUTE FORCE SAVING: CIK %s year %d -> %s",
                    cik10,
                    fyear,
                    final_check_path.name,
                )
                sec_path, shared_path = self._persist_photo(
                    image_bytes=chosen.bytes_data,  # type: ignore[arg-type]
                    company_dir=company_dir,
                    accession_dir=target_accession,
                    year=fyear,
                    cik10=cik10,
                )
                
                detail = chosen.reason or "brute_force"
                return PhotoResult(
                    cik10=cik10,
                    fyear=fyear,
                    accession=target_accession.name,
                    ceo=ceo_name,
                    status="success",
                    detail=detail,
                    sec_local_path=sec_path,
                    shared_local_path=shared_path,
                    source_url=chosen.url,
                    stage="brute_force",
                    photo_prominence=chosen.photo_prominence or 3,
                    position_index=chosen.position_index,
                    estimated_size=chosen.estimated_size,
                )
            else:

                return PhotoResult(
                    cik10=cik10,
                    fyear=fyear,
                    accession=target_accession.name,
                    ceo=ceo_name,
                    status="no_candidate_passed_filters",
                    detail="brute_force_no_face_found",
                )
        
        if self.strategy_0_5_only:

            if self.verbose:
                self.logger.info("STRATEGY 0.5 ONLY MODE: Skipping strategies 0, 1, 2 and going directly to CEO name-based search (nearby 20 images)...")
            
            chosen = self._find_and_check_images_near_ceo_name(
                soup,
                cik10,
                target_accession.name,
                ceo_name=ceo_name,
            )
            
            if chosen is not None:

                sec_path, shared_path = self._persist_photo(
                    image_bytes=chosen.bytes_data,  # type: ignore[arg-type]
                    company_dir=company_dir,
                    accession_dir=target_accession,
                    year=fyear,
                    cik10=cik10,
                )
                
                detail = chosen.reason or "ceo_name_based_check"
                return PhotoResult(
                    cik10=cik10,
                    fyear=fyear,
                    accession=target_accession.name,
                    ceo=ceo_name,
                    status="success",
                    detail=detail,
                    sec_local_path=sec_path,
                    shared_local_path=shared_path,
                    source_url=chosen.url,
                    stage="ceo_name_based_check",
                    photo_prominence=chosen.photo_prominence or 3,
                    position_index=chosen.position_index,
                    estimated_size=chosen.estimated_size,
                )
            else:

                return PhotoResult(
                    cik10=cik10,
                    fyear=fyear,
                    accession=target_accession.name,
                    ceo=ceo_name,
                    status="no_candidate_passed_filters",
                )
        else:


            keyword_found = self._check_keywords_exist(soup)
        
        if not self.strategy_0_5_only and keyword_found and self._is_strategy_enabled(self.STRATEGY_KEYWORD_NEAR_PHRASE):
            if self.verbose:
                self.logger.info("STEP 0: Found keywords, using keyword-based sequential face check...")
            
            chosen = self._find_and_check_images_near_keywords(
                soup,
                cik10,
                target_accession.name,
                ceo_name=ceo_name,
                max_images=10,
            )
            
            if chosen is not None:

                sec_path, shared_path = self._persist_photo(
                    image_bytes=chosen.bytes_data,  # type: ignore[arg-type]
                    company_dir=company_dir,
                    accession_dir=target_accession,
                    year=fyear,
                    cik10=cik10,
                )
                
                detail = chosen.reason or "keyword_sequential_check"
                return PhotoResult(
                    cik10=cik10,
                    fyear=fyear,
                    accession=target_accession.name,
                    ceo=ceo_name,
                    status="success",
                    detail=detail,
                    sec_local_path=sec_path,
                    shared_local_path=shared_path,
                    source_url=chosen.url,
                    stage="keyword_sequential_check",
                    photo_prominence=chosen.photo_prominence or 3,
                    position_index=chosen.position_index,
                    estimated_size=chosen.estimated_size,
                )
            else:
                if self.verbose:
                    self.logger.info("No face found in 10 images near keywords, falling back to CEO name-based search...")
        elif keyword_found and not self._is_strategy_enabled(self.STRATEGY_KEYWORD_NEAR_PHRASE):
            if self.verbose:
                self.logger.info(
                    "Skipping STEP 0 (keyword sequential check) because strategy_limit=%s",
                    self.strategy_limit,
                )
        

        if not self.strategy_0_5_only and chosen is None and ceo_name and self._is_strategy_enabled(self.STRATEGY_CEO_NAME_CONTEXT):
            if self.verbose:
                self.logger.info("STEP 0.5: Searching images near CEO name/surname/firstname mentions...")
            
            chosen = self._find_and_check_images_near_ceo_name(
                soup,
                cik10,
                target_accession.name,
                ceo_name=ceo_name,
            )
            
            if chosen is not None:

                sec_path, shared_path = self._persist_photo(
                    image_bytes=chosen.bytes_data,  # type: ignore[arg-type]
                    company_dir=company_dir,
                    accession_dir=target_accession,
                    year=fyear,
                    cik10=cik10,
                )
                
                
                detail = chosen.reason or "ceo_name_based_check"
                return PhotoResult(
                    cik10=cik10,
                    fyear=fyear,
                    accession=target_accession.name,
                    ceo=ceo_name,
                    status="success",
                    detail=detail,
                    sec_local_path=sec_path,
                    shared_local_path=shared_path,
                    source_url=chosen.url,
                    stage="ceo_name_based_check",
                    photo_prominence=chosen.photo_prominence or 3,
                    position_index=chosen.position_index,
                    estimated_size=chosen.estimated_size,
                )
            else:
                if self.verbose:
                    self.logger.info("No face found near CEO name mentions, continuing to CEO letter block strategy...")
        elif chosen is None and ceo_name:
            if self.verbose:
                self.logger.info(
                    "Skipping STEP 0.5 (CEO name search) because strategy_limit=%s",
                    self.strategy_limit,
                )
        

        if not self.strategy_0_5_only and self._is_strategy_enabled(self.STRATEGY_CEO_LETTER_BLOCK):
            if self.verbose:
                self.logger.info("STEP 1: Finding CEO letter blocks...")
            
            blocks = find_ceo_letter_blocks(soup, logger=self.logger if self.verbose else None)
            candidates = extract_images_from_blocks(
                blocks,
                soup,
                cik10,
                target_accession.name,
                logger=self.logger if self.verbose else None,
            )
            
            if self.verbose:
                if candidates:
                    self.logger.info("Found %d images in CEO letter blocks", len(candidates))
                else:
                    self.logger.info("No images in CEO letter blocks, trying early document images")
        else:
            if self.verbose:
                self.logger.info(
                    "Skipping STEP 1 (CEO letter block strategy) because strategy_limit=%s",
                    self.strategy_limit,
                )


        if not candidates:
            if self._is_strategy_enabled(self.STRATEGY_EARLY_DOCUMENT):
                if self.verbose:
                    self.logger.info("STEP 2: Getting early document images...")
                
                candidates = get_early_document_images(
                    soup,
                    cik10,
                    target_accession.name,
                    limit=25,
                    logger=self.logger if self.verbose else None,
                )
                
                if self.verbose:
                    self.logger.info("Found %d early images", len(candidates))
            elif self.verbose:
                self.logger.info(
                    "Skipping STEP 2 (early document images) because strategy_limit=%s",
                    self.strategy_limit,
                )
        

        if candidates and position_range:
            pos_min, pos_max = position_range
            if self.verbose:
                self.logger.info(
                    "📊 Re-ranking candidates based on position range %d-%d",
                    pos_min, pos_max
                )
            

            in_range = []
            out_range = []
            
            for candidate in candidates:
                if pos_min <= candidate.position_index <= pos_max:
                    in_range.append(candidate)
                else:
                    out_range.append(candidate)
            

            in_range.sort(key=lambda x: (x.position_index, -x.estimated_size))

            out_range.sort(key=lambda x: (x.position_index, -x.estimated_size))
            

            candidates = in_range + out_range
            
            if self.verbose:
                self.logger.info(
                    "✓ Re-ranked: %d in range [%d-%d], %d out of range",
                    len(in_range), pos_min, pos_max, len(out_range)
                )


        if not candidates:
            if self._is_strategy_enabled(self.STRATEGY_HISTORICAL_FALLBACK):

                has_historical_photo = self._check_historical_photo_exists(cik10, fyear)
                
                if has_historical_photo:
                    if self.verbose:
                        self.logger.info(
                            "⚠ No candidates found, but historical photo exists. Using keyword-distance fallback strategy..."
                        )
                    

                    candidates = self._find_images_by_keyword_distance(
                        soup,
                        cik10,
                        target_accession.name,
                        logger=self.logger if self.verbose else None,
                    )
                    
                    if candidates and self.verbose:
                        self.logger.info(
                            "Found %d images using keyword-distance fallback strategy",
                            len(candidates)
                        )
            elif self.verbose:
                self.logger.info(
                    "Skipping fallback strategies because strategy_limit=%s",
                    self.strategy_limit,
                )

        if not candidates:
            if self.verbose:
                self.logger.warning(
                    "No candidates found after all strategies for cik=%s acc=%s (total img tags: %d)",
                    cik10,
                    target_accession.name,
                    len(all_imgs),
                )
            return PhotoResult(
                cik10=cik10,
                fyear=fyear,
                accession=target_accession.name,
                ceo=ceo_name,
                status="no_images_detected",
            )

        chosen = self._evaluate_candidates(
            candidates,
            ceo_name=ceo_name,
            cik10=cik10,
            accession=target_accession.name,
            soup=soup,
        )


        if (
            chosen is None
            and candidates
            and len(candidates) > 0
            and candidates[0].stage == "keyword_distance_fallback"
        ):
            if self._is_strategy_enabled(self.STRATEGY_SEQUENTIAL_FALLBACK):
                if self.verbose:
                    self.logger.info(
                        "No candidate passed filters, but using sequential face detection for fallback candidates..."
                    )
                
                chosen = self._evaluate_candidates_sequential_face_check(
                    candidates,
                    ceo_name=ceo_name,
                    cik10=cik10,
                    accession=target_accession.name,
                    soup=soup,
                )
            elif self.verbose:
                self.logger.info(
                    "Skipping sequential face fallback because strategy_limit=%s",
                    self.strategy_limit,
                )

        if chosen is None:
            return PhotoResult(
                cik10=cik10,
                fyear=fyear,
                accession=target_accession.name,
                ceo=ceo_name,
                status="no_candidate_passed_filters",
            )

        sec_path, shared_path = self._persist_photo(
            image_bytes=chosen.bytes_data,  # type: ignore[arg-type]
            company_dir=company_dir,
            accession_dir=target_accession,
            year=fyear,
            cik10=cik10,
        )

        detail = chosen.reason or "selected_candidate"
        return PhotoResult(
            cik10=cik10,
            fyear=fyear,
            accession=target_accession.name,
            ceo=ceo_name,
            status="success",
            detail=detail,
            sec_local_path=sec_path,
            shared_local_path=shared_path,
            source_url=chosen.url,
            stage=chosen.stage,
            photo_prominence=chosen.photo_prominence or 3,
            position_index=chosen.position_index,
            estimated_size=chosen.estimated_size,
        )

    def _choose_company_directory(self, candidates: List[Path], company_variants: List[str]) -> Optional[Path]:
        if len(candidates) == 1:
            return candidates[0]

        scored: List[Tuple[int, Path]] = []
        for entry in candidates:
            name_part = sanitize_company_name(entry.name.rsplit("_", maxsplit=1)[0])
            match_score = 0
            for variant in company_variants:
                if not variant:
                    continue
                if variant == name_part:
                    match_score += 3
                elif variant in name_part:
                    match_score += 2
            scored.append((match_score, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored and scored[0][0] > 0:
            return scored[0][1]
        return candidates[0] if candidates else None

    def _select_accession(self, filing_dir: Path, year_suffix: str) -> Optional[Path]:
        matches = [
            path
            for path in filing_dir.iterdir()
            if path.is_dir() and f"-{year_suffix}-" in path.name
        ]
        if not matches:
            return None
        matches.sort(key=lambda p: p.name, reverse=True)
        return matches[0]

    def _find_primary_document(self, accession_dir: Path) -> Optional[Path]:
        priority_patterns = [
            "primary-document.html",
            "primary_document.html",
            "primary-document.htm",
            "primary_document.htm",
        ]
        for pattern in priority_patterns:
            target = accession_dir / pattern
            if target.exists():
                return target

        for pattern in ("primary-document*", "index.htm*", "*.html", "*.htm"):
            paths = list(accession_dir.glob(pattern))
            for path in paths:
                if path.is_file():
                    return path
        return None

    def _evaluate_candidates(
        self,
        candidates: List[CandidateImage],
        ceo_name: str,
        cik10: str,
        accession: str,
        soup: BeautifulSoup,
    ) -> Optional[CandidateImage]:
        """
        Candidate evaluation pipeline:
        1) Iterate candidates in order.
        2) Basic checks (size, aspect ratio).
        3) OpenCV face detection.
        4) If a face is found, verify CEO context (name or title).
        5) If context passes, use DashScope for final validation (face visible, no mask).
        6) Return the first candidate passing all checks.

        Notes:
        - Skip DashScope if OpenCV finds no face (token savings).
        - DashScope is only for final confirmation (clear face, no mask).
        """
        if self.verbose:
            self.logger.info(
                "Evaluating %d candidates for cik=%s acc=%s",
                len(candidates),
                cik10,
                accession,
            )


        all_imgs = soup.find_all("img")
        url_to_img = {}
        for img in all_imgs:
            src = img.get("src")
            if src:
                url = build_image_url(src, cik10, accession)
                url_to_img[url] = img

        processed = 0
        pending_dashscope: List[Tuple[CandidateImage, bytes, str, str]] = []
        for candidate in candidates:
            if processed >= MAX_DOWNLOAD_PER_FILING:
                break
            
            processed += 1
            
            if self.verbose:
                self.logger.info(
                    "Checking candidate %d/%d: position=%d size=%d url=%s",
                    processed,
                    min(len(candidates), MAX_DOWNLOAD_PER_FILING),
                    candidate.position_index,
                    candidate.estimated_size,
                    candidate.url,
                )
            

            if candidate.position_index == 0:
                if self.verbose:
                    self.logger.info("Skipping position 0 (cover page)")
                continue
            if self._should_skip_small_gif(candidate.url, candidate.estimated_size):
                candidate.reason = "gif_small_estimated"
                continue
            

            image_bytes = load_image_bytes(self.session, candidate.url)
            if not image_bytes:
                if self.verbose:
                    self.logger.debug("Download failed, skipping")
                self._record_filter_event("download_failed")
                continue
            
            candidate.bytes_data = image_bytes


            basic_ok, basic_conf, basic_reason, image_size = basic_image_checks(image_bytes)
            if not basic_ok:
                if self.verbose:
                    self.logger.info("❌ Basic check failed: %s", basic_reason)
                candidate.reason = basic_reason
                self._record_filter_event("basic_fail")
                continue
            actual_area = image_size[0] * image_size[1]
            if self._should_skip_small_gif(candidate.url, actual_area):
                candidate.reason = f"{basic_reason}|gif_small_actual"
                continue
            self._record_filter_event("basic_pass")


            img_tag = url_to_img.get(candidate.url)
            if img_tag:
                should_include, is_retired = self._check_retired_ceo_filter(
                    soup,
                    img_tag,
                    ceo_name,
                    context_lines=15,
                )
                
                if not should_include:
                    if self.verbose:
                        self.logger.info(
                            "⛔ Candidate at position %d filtered: retired CEO detected, skipping...",
                            candidate.position_index
                        )
                    candidate.reason = f"{basic_reason}|retired_ceo_filtered"
                    self._record_filter_event("retired_filter_blocked")
                    continue
                
                if is_retired and self.verbose:
                    self.logger.info(
                        "✓ Candidate at position %d: retired CEO context, but CEO name matched - exempted",
                        candidate.position_index
                    )



            face_ok = False
            face_conf = 0.0
            face_boxes = None
            face_reason = ""

            if cv2 is not None:
                face_ok, face_conf, face_reason, face_boxes = detect_face_with_cv(image_bytes)
                if face_ok and face_conf < MIN_OPENCV_FACE_CONFIDENCE:
                    if self.verbose:
                        self.logger.info(
                            "❌ Face confidence %.2f below threshold %.2f, skipping",
                            face_conf,
                            MIN_OPENCV_FACE_CONFIDENCE,
                        )
                    face_ok = False
                    face_reason = "face_conf_too_low"
                if self.verbose:
                    if face_ok:
                        self.logger.info("✓ Face detected by OpenCV (confidence=%.2f)", face_conf)
                    else:
                        self.logger.info("❌ No face detected by OpenCV: %s", face_reason)
            else:
                if self.verbose:
                    self.logger.info("OpenCV not available, skipping face detection")
            

            if not face_ok:
                fail_reason = face_reason or "no_face_detected"
                candidate.reason = f"{basic_reason}|{fail_reason}"
                if face_reason == "face_conf_too_low":
                    self._record_filter_event("face_conf_too_low")
                else:
                    self._record_filter_event("face_fail")
                continue
            self._record_filter_event("face_pass")

            candidate.photo_prominence = calculate_photo_prominence(image_size, face_boxes)

            base_reason = f"{basic_reason}|face_detected"





            is_first_layer = (candidate.stage == "ceo_letter_block")
            is_fallback_stage = candidate.stage in ("keyword_distance_fallback", "sequential_face_check")
            strict_mode = not (is_first_layer or is_fallback_stage)
            
            has_ceo_context = False
            img_tag = url_to_img.get(candidate.url)
            if img_tag:
                has_ceo_context = verify_ceo_context(
                    img_tag, 
                    ceo_name, 
                    strict_mode=strict_mode,
                    logger=self.logger if self.verbose else None
                )
            

            if not is_first_layer and not is_fallback_stage and not has_ceo_context:
                if self.verbose:
                    self.logger.info(
                        "❌ No CEO name/title found near image (strict mode), skipping"
                    )
                candidate.reason = f"{basic_reason}|face_detected|no_ceo_context"
                self._record_filter_event("context_fail")
                continue
            

            if is_first_layer or is_fallback_stage:
                if has_ceo_context:
                    if self.verbose:
                        self.logger.info(
                            "✓ CEO context verified (name or title found, loose mode)"
                        )
                    self._record_filter_event("context_pass")
                    base_reason += "|ceo_context_ok_loose"
                else:
                    if self.verbose:
                        self.logger.info(
                            "⚠ No CEO context found, but allowing (loose mode)"
                        )
                    self._record_filter_event("context_skipped_loose")
                    base_reason += "|ceo_context_skipped_loose"
            elif has_ceo_context:
                self._record_filter_event("context_pass")
                base_reason += "|ceo_context_ok"
            

            if self.dashscope_client.is_ready():
                assert candidate.bytes_data is not None
                pending_dashscope.append((candidate, candidate.bytes_data, ceo_name, base_reason))
                selected = self._maybe_flush_dashscope_queue(
                    pending_dashscope,
                    stage_hint="candidate_eval",
                )
                if selected:
                    if self.verbose:
                        self.logger.info("✓ SELECTED! This is the CEO photo.")
                    return selected
            else:

                if self.verbose:
                    self.logger.warning("DashScope not available, rejecting image to ensure all photos in pool are validated")
                candidate.reason = f"{base_reason}|dashscope_unavailable|rejected"
                self._record_filter_event("dashscope_unavailable")
                continue
        

        selected_final = self._maybe_flush_dashscope_queue(
            pending_dashscope,
            stage_hint="candidate_eval",
            force=True,
        )
        if selected_final:
            if self.verbose:
                self.logger.info("✓ SELECTED via deferred DashScope batch")
            return selected_final

        if self.verbose:
            self.logger.info("No suitable candidate found after checking %d images", processed)
        
        return None

    def _check_keywords_exist(self, soup: BeautifulSoup) -> bool:
        """
        Check whether core opening phrases (e.g., "to our shareholders") exist in HTML.

        Returns:
            True if a phrase is found, else False.
        """
        opening_phrases = [
            "to our stockholders",
            "to our shareholders",
            "to the stockholders",
            "to the shareholders",
            "dear fellow stockholders",
            "dear fellow shareholders",
            "dear stockholders",
            "dear shareholders",
            "letter to stockholders",
            "letter to shareholders",
            "letter to the stockholders",
            "letter to the shareholders",
            "to our valued stockholders",
            "to our valued shareholders",
            "shareholders letter",
            "stockholders letter",
            "letter from the chairman",
            "letter from the ceo",
            "letter from the president",
            "dear shareholder",
            "dear stockholder",
            "to shareholders",
            "to stockholders",
            "fellow shareholders",
            "fellow stockholders",
            "message from the ceo",
            "ceo message",
            "ceo's message",
            "message to shareholders",
            "message from our ceo",
        ]
        
        for phrase in opening_phrases:
            phrase_lower = phrase.lower()
            for text_node in soup.find_all(string=lambda t: t and phrase_lower in str(t).strip().lower()):
                return True
        
        return False
    
    def _check_retired_ceo_filter(
        self,
        soup: BeautifulSoup,
        img: Tag,
        ceo_name: str,
        context_lines: int = 15,
    ) -> Tuple[bool, bool]:
        """
        Detect "retired CEO" wording near an image to avoid obsolete portraits.

        Rules:
        1) If HTML lines within ±context_lines contain retired CEO terms, exclude the image.
        2) Exception: if the current CEO name also appears nearby, keep it.

        Args:
            soup: Parsed HTML.
            img: Image tag.
            ceo_name: Current CEO name.
            context_lines: Line window to inspect (default 15).

        Returns:
            (should_include, is_retired_ceo_detected)
        """
        try:

            html_str = str(soup)
            html_lines = html_str.split('\n')
            

            img_str = str(img)
            img_line = None
            
            for idx, line in enumerate(html_lines):
                if img_str in line or img.get("src", "") in line:
                    img_line = idx
                    break
            

            if img_line is None:

                img_pos = html_str.find(img_str)
                if img_pos == -1:
                    src = img.get("src", "")
                    img_pos = html_str.find(src) if src else -1
                
                if img_pos == -1:

                    return True, False
                

                img_line = html_str[:img_pos].count('\n')
            

            context_start = max(0, img_line - context_lines)
            context_end = min(len(html_lines), img_line + context_lines + 1)
            context_lines_text = '\n'.join(html_lines[context_start:context_end]).lower()
            

            retired_keywords = [
                "retired ceo",
                "former ceo",
                "former chief executive officer",
                "retired chief executive officer",
                "retired chief executive",
                "retired chairman",
                "former chairman",
                "emeritus",
                "retired president",
                "former president",
                "retired executive",
                "former executive",
            ]
            

            is_retired = False
            for keyword in retired_keywords:
                if keyword in context_lines_text:
                    is_retired = True
                    break
            

            if is_retired:

                if ceo_name:
                    ceo_variants = name_variants(ceo_name)
                    
                    for variant in ceo_variants:
                        if not variant:
                            continue
                        variant_lower = variant.lower().strip()
                        

                        if variant_lower in context_lines_text:
                            if self.verbose:
                                self.logger.debug(
                                    "Found retired CEO context, but CEO name matched - exempting: %s",
                                    variant
                                )
                            return True, True
                        

                        name_parts = variant_lower.split()
                        if len(name_parts) > 0:
                            last_name = name_parts[-1]
                            if len(last_name) >= 3 and last_name in context_lines_text:
                                if self.verbose:
                                    self.logger.debug(
                                        "Found retired CEO context, but CEO last name matched - exempting: %s",
                                        last_name
                                    )
                                return True, True
                

                if self.verbose:
                    self.logger.info(
                        "Image filtered: retired CEO context detected, no current CEO name match"
                    )
                return False, True
            

            return True, False
        
        except Exception as exc:
            if self.verbose:
                self.logger.debug("Error checking retired CEO filter: %s", exc)

            return True, False
    
    def _check_ceo_context_near_image(
        self,
        soup: BeautifulSoup,
        img: Tag,
        ceo_name: str,
        context_chars: int = 500,
    ) -> bool:
        """
        Inspect surrounding HTML (±context_chars) for CEO names or titles.

        Checks:
        1) CEO name variants (full, last, first).
        2) CEO titles (CEO, Chairman, President, etc.).

        Args:
            soup: Parsed HTML.
            img: Image tag.
            ceo_name: CEO full name.
            context_chars: Character window size (default 500).

        Returns:
            True if CEO-related context is found, else False.
        """
        try:

            html_str = str(soup)
            

            img_str = str(img)
            img_pos = html_str.find(img_str)
            
            if img_pos == -1:

                src = img.get("src", "")
                if src:
                    img_pos = html_str.find(src)
                    if img_pos == -1:
                        return False
                else:
                    return False
            

            context_start = max(0, img_pos - context_chars)
            context_end = min(len(html_str), img_pos + context_chars)
            context = html_str[context_start:context_end].lower()
            

            ceo_titles = [
                "chief executive officer",
                "ceo",
                "president and ceo",
                "president & ceo",
                "chief executive",
                "chairman and ceo",
                "chairman & ceo",
                "chairman of the board",
                "chairman",
                "chairman of board",
                "board chairman",
                "executive chair",
                "chair executive",
                "president",
                "executive",
            ]
            

            for title in ceo_titles:
                if title in context:
                    if self.verbose:
                        self.logger.debug("Found CEO title in context: %s", title)
                    return True
            

            if ceo_name:
                ceo_variants = name_variants(ceo_name)
                
                for variant in ceo_variants:
                    if not variant:
                        continue
                    variant_lower = variant.lower().strip()
                    

                    if variant_lower in context:
                        if self.verbose:
                            self.logger.debug("Found CEO full name in context: %s", variant)
                        return True
                    

                    name_parts = variant_lower.split()
                    if len(name_parts) > 0:
                        last_name = name_parts[-1]

                        if len(last_name) >= 3 and last_name in context:
                            if self.verbose:
                                self.logger.debug("Found CEO last name in context: %s", last_name)
                            return True
                    

                    if len(name_parts) > 1:
                        first_name = name_parts[0]

                        if len(first_name) >= 3 and first_name in context:
                            if self.verbose:
                                self.logger.debug("Found CEO first name in context: %s", first_name)
                            return True
            
            return False
        
        except Exception as exc:
            if self.verbose:
                self.logger.debug("Error checking CEO context near image: %s", exc)
            return False
    
    def _find_and_check_images_near_keywords(
        self,
        soup: BeautifulSoup,
        cik10: str,
        accession: str,
        ceo_name: str,
        max_images: int = 10,
    ) -> Optional[CandidateImage]:
        """
        Locate images near core keywords and check faces sequentially.

        Strategy:
        1) Find keyword positions.
        2) Start from KEYWORD_LINE_LOOKBACK lines above and scan downward.
        3) Stop at the first face found.
        4) Inspect up to max_images.

        Returns:
            First candidate with a detected face, or None.
        """

        opening_phrases = [
            "to our stockholders",
            "to our shareholders",
            "to the stockholders",
            "to the shareholders",
            "dear fellow stockholders",
            "dear fellow shareholders",
            "dear stockholders",
            "dear shareholders",
            "letter to stockholders",
            "letter to shareholders",
            "letter to the stockholders",
            "letter to the shareholders",
            "to our valued stockholders",
            "to our valued shareholders",
            "shareholders letter",
            "stockholders letter",
            "letter from the chairman",
            "letter from the ceo",
            "letter from the president",
            "dear shareholder",
            "dear stockholder",
            "to shareholders",
            "to stockholders",
            "fellow shareholders",
            "fellow stockholders",
            "message from the ceo",
            "ceo message",
            "ceo's message",
            "message to shareholders",
            "message from our ceo",
        ]
        

        keyword_nodes = []
        for phrase in opening_phrases:
            phrase_lower = phrase.lower()
            for text_node in soup.find_all(string=lambda t: t and phrase_lower in str(t).strip().lower()):
                keyword_nodes.append(text_node)
        
        if not keyword_nodes:
            return None
        

        keyword_positions = []
        for keyword_node in keyword_nodes:
            keyword_pos = getattr(keyword_node, 'sourceline', None)
            if keyword_pos is None:
                parent = keyword_node.parent
                keyword_pos = getattr(parent, 'sourceline', None) if parent else None
            
            if keyword_pos is not None:
                keyword_positions.append(keyword_pos)
        

        if not keyword_positions:
            all_elements = list(soup.descendants)
            for keyword_node in keyword_nodes:
                try:
                    keyword_idx = all_elements.index(keyword_node)
                    keyword_positions.append(keyword_idx)
                except ValueError:
                    parent = keyword_node.parent
                    if parent:
                        try:
                            keyword_idx = all_elements.index(parent)
                            keyword_positions.append(keyword_idx)
                        except ValueError:
                            pass
        
        if not keyword_positions:
            return None
        

        earliest_keyword_pos = min(keyword_positions)
        

        start_pos = max(0, earliest_keyword_pos - KEYWORD_LINE_LOOKBACK)
        
        if self.verbose:
            self.logger.info(
                "Keyword found at position %d, starting image search from position %d (%d lines before)",
                earliest_keyword_pos,
                start_pos,
                KEYWORD_LINE_LOOKBACK
            )
        

        all_imgs = soup.find_all("img")
        if not all_imgs:
            return None
        

        img_to_index = {id(img): idx for idx, img in enumerate(all_imgs)}
        

        images_with_pos: List[Tuple[Tag, int, int]] = []  # (img_tag, html_pos, position_index)
        
        all_elements_for_img = list(soup.descendants)
        
        for img in all_imgs:
            img_html_pos = getattr(img, 'sourceline', None)
            
            if img_html_pos is None:
                try:
                    img_html_pos = all_elements_for_img.index(img)
                except ValueError:
                    img_position = img_to_index.get(id(img), len(all_imgs))
                    img_html_pos = img_position * 50
            

            if img_html_pos >= start_pos:
                img_position = img_to_index.get(id(img), len(all_imgs))
                images_with_pos.append((img, img_html_pos, img_position))
        

        images_with_pos.sort(key=lambda x: x[1])
        

        images_with_pos = images_with_pos[:max_images]
        
        if self.verbose:
            self.logger.info(
                "Found %d images to check (from position %d onwards)",
                len(images_with_pos),
                start_pos
            )
        

        processed = 0
        pending_dashscope: List[Tuple[CandidateImage, bytes, str, str]] = []
        for img, img_html_pos, img_position in images_with_pos:
            if processed >= max_images:
                break
            
            processed += 1
            
            src = img.get("src")
            if not src:
                continue
            
            url = build_image_url(src, cik10, accession)
            

            if img_position == 0:
                continue
            

            estimated_size = estimate_image_size(img)
            MIN_SIZE = 8_000
            if estimated_size > 0 and estimated_size < MIN_SIZE:
                if self.verbose:
                    self.logger.info(
                        "Skipping small image at position %d (estimated_size=%d)",
                        img_position,
                        estimated_size
                    )
                self._record_filter_event("size_filtered")
                continue

            if is_extreme_aspect_ratio(img, max_ratio=4.0):
                if self.verbose:
                    self.logger.info(
                        "Skipping extreme aspect ratio image at position %d",
                        img_position,
                    )
                self._record_filter_event("aspect_ratio_filtered")
                continue
            if self._should_skip_small_gif(url, estimated_size):
                continue
            
            if self.verbose:
                self.logger.info(
                    "Checking image %d/%d: position=%d, html_pos=%d, url=%s",
                    processed,
                    len(images_with_pos),
                    img_position,
                    img_html_pos,
                    url,
                )
            

            image_bytes = load_image_bytes(self.session, url)
            if not image_bytes:
                self._record_filter_event("download_failed")
                continue
            

            basic_ok, basic_conf, basic_reason, image_size = basic_image_checks(image_bytes)
            if not basic_ok:
                if self.verbose:
                    self.logger.info("❌ Basic check failed: %s", basic_reason)
                self._record_filter_event("basic_fail")
                continue
            actual_area = image_size[0] * image_size[1]
            if self._should_skip_small_gif(url, actual_area):
                continue
            self._record_filter_event("basic_pass")
            

            # should_include, is_retired = self._check_retired_ceo_filter(
            #     soup,
            #     img,
            #     ceo_name,
            #     context_lines=15,
            # )
            # 
            # if not should_include:
            #     if self.verbose:
            #         self.logger.info(

            #             img_position
            #         )
            #     self._record_filter_event("retired_filter_blocked")
            #     continue
            # 
            # if is_retired and self.verbose:
            #     self.logger.info(

            #         img_position
            #     )

            

            face_ok = False
            face_conf = 0.0
            face_boxes = None
            
            if cv2 is not None:
                face_ok, face_conf, face_reason, face_boxes = detect_face_with_cv(image_bytes)
                if face_ok and face_conf < MIN_OPENCV_FACE_CONFIDENCE:
                    if self.verbose:
                        self.logger.info(
                            "⚠ Face confidence %.2f below threshold %.2f, skipping",
                            face_conf,
                            MIN_OPENCV_FACE_CONFIDENCE,
                        )
                    face_ok = False
                    face_reason = "face_conf_too_low"
                if face_ok:
                    if self.verbose:
                        self.logger.info(
                            "✓ Face detected by OpenCV (position=%d, confidence=%.2f)",
                            img_position,
                            face_conf
                        )
                    self._record_filter_event("face_pass")
                    


                    has_ceo_context = self._check_ceo_context_near_image(
                        soup,
                        img,
                        ceo_name,
                        context_chars=2000,
                    )
                    
                    if not has_ceo_context:
                        if self.verbose:
                            self.logger.info(
                                "⚠ Face detected, but no CEO context found in ±500 chars, skipping..."
                            )
                        self._record_filter_event("context_fail")
                        continue
                    
                    if self.verbose:
                        self.logger.info(
                            "✓ CEO context verified (name or title found in image vicinity)"
                        )
                    self._record_filter_event("context_pass")

                    
                    photo_prominence = calculate_photo_prominence(image_size, face_boxes)

                    candidate = CandidateImage(
                        url=url,
                        score=0.0,
                        stage="keyword_sequential_check",
                        position_index=img_position,
                        estimated_size=estimated_size,
                        bytes_data=image_bytes,
                        photo_prominence=photo_prominence,
                    )
                    base_reason = f"keyword_sequential_check|face_detected|ceo_context_ok"

                    if self.dashscope_client.is_ready():
                        pending_dashscope.append((candidate, image_bytes, ceo_name, base_reason))
                        selected = self._maybe_flush_dashscope_queue(
                            pending_dashscope,
                            stage_hint="keyword_sequential",
                        )
                        if selected:
                            return selected
                    else:
                        if self.verbose:
                            self.logger.warning("DashScope not available, rejecting image to ensure all photos in pool are validated")
                        candidate.reason = f"{base_reason}|dashscope_unavailable|rejected"
                        self._record_filter_event("dashscope_unavailable")
                        continue
                else:
                    if self.verbose:
                        self.logger.info(
                            "No face detected at position %d, continuing...",
                            img_position
                        )
                    if face_reason == "face_conf_too_low":
                        self._record_filter_event("face_conf_too_low")
                    else:
                        self._record_filter_event("face_fail")
            else:
                if self.verbose:
                    self.logger.info("OpenCV not available, skipping face check")
        
        selected_final = self._maybe_flush_dashscope_queue(
            pending_dashscope,
            stage_hint="keyword_sequential",
            force=True,
        )
        if selected_final:
            return selected_final

        if self.verbose:
            self.logger.info("No face detected in %d images near keywords", processed)
        
        return None
    
    def _find_and_check_images_near_ceo_name(
        self,
        soup: BeautifulSoup,
        cik10: str,
        accession: str,
        ceo_name: str,
    ) -> Optional[CandidateImage]:
        """
        Find images based on CEO name occurrences (strategy v0.5).

        Strategy:
        1) Build name variants (full, last, first).
        2) Locate each variant in HTML.
        3) Around each location, gather nearby images (roughly ±20).
        4) Run batched face detection and LLM validation.
        5) Add face-positive photos to the pool and return in order.

        Returns:
            First candidate with a detected face, or None.
        """
        try:

            ceo_variants = name_variants(ceo_name)
            
            if self.verbose:
                self.logger.info(
                    "CEO name variants to search: %s",
                    [v for v in ceo_variants if v]
                )
            

            all_imgs = soup.find_all("img")
            if not all_imgs:
                return None
            
            img_to_index = {id(img): idx for idx, img in enumerate(all_imgs)}
            

            collected_candidates: List[Tuple[CandidateImage, int]] = []  # (candidate, position_in_html)
            seen_urls: set[str] = set()
            
            html_str = str(soup)
            
            for variant in ceo_variants:
                if not variant or len(variant) < 2:
                    continue
                
                variant_lower = variant.lower()
                

                search_pos = 0
                while True:
                    pos = html_str.lower().find(variant_lower, search_pos)
                    if pos == -1:
                        break
                    
                    if self.verbose:
                        self.logger.info(
                            "Found CEO name variant '%s' at HTML position %d, collecting nearby images...",
                            variant,
                            pos
                        )
                    


                    img_positions = []
                    for img in all_imgs:
                        img_str = str(img)
                        img_pos = html_str.find(img_str)
                        if img_pos != -1:
                            img_positions.append((img, img_pos, img_to_index.get(id(img), len(all_imgs))))
                    

                    img_positions.sort(key=lambda x: x[1])
                    

                    nearby_imgs = []
                    for img, img_pos, img_position in img_positions:
                        abs_distance = abs(img_pos - pos)
                        nearby_imgs.append((img, img_pos, img_position, abs_distance))
                    

                    nearby_imgs.sort(key=lambda x: x[3])
                    nearby_imgs = nearby_imgs[:20]
                    

                    nearby_imgs.sort(key=lambda x: x[1])
                    
                    if self.verbose:
                        self.logger.info(
                            "Collected %d images near CEO name mention",
                            len(nearby_imgs)
                        )
                    

                    for img, img_pos, img_position, distance in nearby_imgs:
                        src = img.get("src", "")
                        

                        estimated_size = estimate_image_size(img)
                        MIN_SIZE = 8_000
                        if estimated_size > 0 and estimated_size < MIN_SIZE:
                            continue
                        

                        if is_extreme_aspect_ratio(img, max_ratio=4.0):
                            continue
                        
                        if img_position == 0:
                            continue
                        
                        url = build_image_url(src, cik10, accession)
                        if self._should_skip_small_gif(url, estimated_size):
                            continue
                        
                        if url not in seen_urls:
                            candidate = CandidateImage(
                                url=url,
                                score=0.0,
                                stage="ceo_name_based_check",
                                position_index=img_position,
                                estimated_size=estimated_size,
                            )
                            collected_candidates.append((candidate, img_pos))
                            seen_urls.add(url)
                    
                    search_pos = pos + 1
            
            if not collected_candidates:
                if self.verbose:
                    self.logger.info("No images found near CEO name mentions")
                return None
            
            if self.verbose:
                self.logger.info(
                    "Total collected %d unique images near CEO name mentions, checking each...",
                    len(collected_candidates)
                )
            

            pending_dashscope: List[Tuple[CandidateImage, bytes, str, str]] = []
            for candidate, _ in collected_candidates:
                if self.verbose:
                    self.logger.info(
                        "Checking image at position %d (size=%d, url=%s)",
                        candidate.position_index,
                        candidate.estimated_size,
                        candidate.url,
                    )
                

                image_bytes = load_image_bytes(self.session, candidate.url)
                if not image_bytes:
                    if self.verbose:
                        self.logger.debug("Failed to download image")
                    self._record_filter_event("download_failed")
                    continue
                
                candidate.bytes_data = image_bytes
                

                basic_ok, basic_conf, basic_reason, image_size = basic_image_checks(image_bytes)
                if not basic_ok:
                    if self.verbose:
                        self.logger.debug("Basic check failed: %s", basic_reason)
                    self._record_filter_event("basic_fail")
                    continue
                actual_area = image_size[0] * image_size[1]
                if self._should_skip_small_gif(candidate.url, actual_area):
                    candidate.reason = f"{basic_reason}|gif_small_actual"
                    continue
                self._record_filter_event("basic_pass")
                

                face_ok, face_conf, face_reason, face_boxes = detect_face_with_cv(image_bytes) if cv2 is not None else (False, 0.0, "", None)
                
                if not face_ok:
                    if self.verbose:
                        self.logger.debug("No face detected")
                    self._record_filter_event("face_fail")
                    continue
                if face_conf < MIN_OPENCV_FACE_CONFIDENCE:
                    if self.verbose:
                        self.logger.debug(
                            "Face confidence %.2f below threshold %.2f",
                            face_conf,
                            MIN_OPENCV_FACE_CONFIDENCE,
                        )
                    candidate.reason = "face_conf_too_low"
                    self._record_filter_event("face_conf_too_low")
                    continue
                
                if self.verbose:
                    self.logger.info(
                        "✓ Face detected at position %d (confidence=%.2f)",
                        candidate.position_index,
                        face_conf
                    )
                self._record_filter_event("face_pass")
                
                candidate.photo_prominence = calculate_photo_prominence(image_size, face_boxes)
                base_reason = "ceo_name_based_check|face_detected"
                

                if self.dashscope_client.is_ready():
                    pending_dashscope.append((candidate, image_bytes, ceo_name, base_reason))
                    selected = self._maybe_flush_dashscope_queue(
                        pending_dashscope,
                        stage_hint="ceo_name_check",
                    )
                    if selected:
                        return selected
                else:

                    if self.verbose:
                        self.logger.warning("DashScope not available, rejecting image to ensure all photos in pool are validated")
                    candidate.reason = f"{base_reason}|dashscope_unavailable|rejected"
                    self._record_filter_event("dashscope_unavailable")
                    continue
            
            if self.verbose:
                self.logger.info("No face detected in any images near CEO name mentions")

            selected_final = self._maybe_flush_dashscope_queue(
                pending_dashscope,
                stage_hint="ceo_name_check",
                force=True,
            )
            if selected_final:
                return selected_final
            
            return None
        
        except Exception as exc:
            if self.verbose:
                self.logger.error("Error in CEO name-based search: %s", exc)
            return None

    def _brute_force_face_search(
        self,
        soup: BeautifulSoup,
        cik10: str,
        accession: str,
        ceo_name: str,
    ) -> Optional[CandidateImage]:
        """
        Brute-force strategy: inspect all size-eligible images.

        Steps:
        1) Collect all images in the document.
        2) Skip the first image (cover).
        3) Run OpenCV face detection on each size-eligible image.
        4) Send face-positive images to DashScope for final confirmation.
        5) Return the first confirmed face candidate.

        Returns:
            First candidate with a detected face, or None.
        """
        try:
            all_imgs = soup.find_all("img")
            if not all_imgs:
                if self.verbose:
                    self.logger.info("BRUTE FORCE: No images found in document")
                return None
            
            if self.verbose:
                self.logger.info("BRUTE FORCE: Scanning all %d images (excluding first/cover)...", len(all_imgs))
            
            MIN_SIZE = 8_000
            checked_count = 0
            passed_face_count = 0
            pending_dashscope: List[Tuple[CandidateImage, bytes, str, str]] = []
            
            for idx, img in enumerate(all_imgs):

                if idx == 0:
                    if self.verbose:
                        self.logger.debug("BRUTE FORCE: Skipping first image (cover)")
                    continue
                
                src = img.get("src")
                if not src:
                    continue
                

                estimated_size = estimate_image_size(img)
                if estimated_size > 0 and estimated_size < MIN_SIZE:
                    continue
                

                if is_extreme_aspect_ratio(img, max_ratio=4.0):
                    continue
                
                url = build_image_url(src, cik10, accession)
                

                if self._should_skip_small_gif(url, estimated_size):
                    continue
                
                checked_count += 1
                
                if self.verbose:
                    self.logger.info(
                        "BRUTE FORCE: Checking image #%d at position %d (size=%d)",
                        checked_count,
                        idx,
                        estimated_size,
                    )
                

                image_bytes = load_image_bytes(self.session, url)
                if not image_bytes:
                    if self.verbose:
                        self.logger.debug("BRUTE FORCE: Failed to download image")
                    self._record_filter_event("brute_force_download_failed")
                    continue
                

                basic_ok, basic_conf, basic_reason, image_size = basic_image_checks(image_bytes)
                if not basic_ok:
                    if self.verbose:
                        self.logger.debug("BRUTE FORCE: Basic check failed: %s", basic_reason)
                    self._record_filter_event("brute_force_basic_fail")
                    continue
                
                actual_area = image_size[0] * image_size[1]
                if self._should_skip_small_gif(url, actual_area):
                    self._record_filter_event("brute_force_gif_small")
                    continue
                
                self._record_filter_event("brute_force_basic_pass")
                

                face_ok, face_conf, face_reason, face_boxes = detect_face_with_cv(image_bytes) if cv2 is not None else (False, 0.0, "", None)
                
                if not face_ok:
                    if self.verbose:
                        self.logger.debug("BRUTE FORCE: No face detected at position %d", idx)
                    self._record_filter_event("brute_force_face_fail")
                    continue
                
                if face_conf < MIN_OPENCV_FACE_CONFIDENCE:
                    if self.verbose:
                        self.logger.debug(
                            "BRUTE FORCE: Face confidence %.2f below threshold %.2f at position %d",
                            face_conf,
                            MIN_OPENCV_FACE_CONFIDENCE,
                            idx,
                        )
                    self._record_filter_event("brute_force_face_conf_low")
                    continue
                
                passed_face_count += 1
                if self.verbose:
                    self.logger.info(
                        "BRUTE FORCE: ✓ Face detected at position %d (confidence=%.2f), sending to DashScope...",
                        idx,
                        face_conf,
                    )
                self._record_filter_event("brute_force_face_pass")
                

                candidate = CandidateImage(
                    url=url,
                    score=face_conf,
                    stage="brute_force",
                    position_index=idx,
                    estimated_size=estimated_size,
                    bytes_data=image_bytes,
                    photo_prominence=calculate_photo_prominence(image_size, face_boxes),
                )
                
                base_reason = "brute_force|face_detected"
                

                if self.dashscope_client.is_ready():
                    pending_dashscope.append((candidate, image_bytes, ceo_name, base_reason))
                    selected = self._maybe_flush_dashscope_queue(
                        pending_dashscope,
                        stage_hint="brute_force",
                    )
                    if selected:
                        if self.verbose:
                            self.logger.info(
                                "BRUTE FORCE: ✓✓ Found valid face photo at position %d, stopping search",
                                idx,
                            )
                        return selected
                else:

                    if self.verbose:
                        self.logger.warning("BRUTE FORCE: DashScope not available, rejecting image")
                    candidate.reason = f"{base_reason}|dashscope_unavailable|rejected"
                    self._record_filter_event("brute_force_dashscope_unavailable")
                    continue
            

            selected_final = self._maybe_flush_dashscope_queue(
                pending_dashscope,
                stage_hint="brute_force",
                force=True,
            )
            if selected_final:
                if self.verbose:
                    self.logger.info("BRUTE FORCE: ✓✓ Found valid face photo in final batch")
                return selected_final
            
            if self.verbose:
                self.logger.info(
                    "BRUTE FORCE: No valid face photo found. Checked %d images, %d passed face detection",
                    checked_count,
                    passed_face_count,
                )
            return None
        
        except Exception as exc:
            if self.verbose:
                self.logger.error("BRUTE FORCE: Error during search: %s", exc)
            return None

    def _check_historical_photo_exists(self, cik10: str, current_year: int) -> bool:
        """
        Check whether earlier-year CEO photos already exist in the pool.

        Args:
            cik10: Company CIK.
            current_year: Year being processed.

        Returns:
            True if an earlier photo is found, otherwise False.
        """

        for year_offset in range(1, 6):
            check_year = current_year - year_offset
            if check_year < 2000:
                break
            

            photo_path = self.shared_output_root / f"{cik10}_{check_year}.png"
            photo_fill_path = self.shared_output_root / f"{cik10}_{check_year}_fill.png"
            
            if photo_path.exists() and photo_path.is_file():
                if self.verbose:
                    self.logger.info(
                        "Found historical photo: %s (year %d)",
                        photo_path.name,
                        check_year
                    )
                return True
            
            if photo_fill_path.exists() and photo_fill_path.is_file():
                if self.verbose:
                    self.logger.info(
                        "Found historical filled photo: %s (year %d)",
                        photo_fill_path.name,
                        check_year
                    )
                return True
        
        return False
    
    def _find_images_by_keyword_distance(
        self,
        soup: BeautifulSoup,
        cik10: str,
        accession: str,
        logger: Optional[logging.Logger] = None,
    ) -> List[CandidateImage]:
        """
        Fallback strategy: rank images by distance to core keywords.

        Steps:
        1) Find core keywords ("to our shareholders", etc.).
        2) Compute HTML-distance from each image to those keywords.
        3) Sort images by distance (nearest first).

        Returns:
            Candidate list ordered by proximity.
        """

        opening_phrases = [
            "to our stockholders",
            "to our shareholders",
            "to the stockholders",
            "to the shareholders",
            "dear fellow stockholders",
            "dear fellow shareholders",
            "dear stockholders",
            "dear shareholders",
            "letter to stockholders",
            "letter to shareholders",
            "letter to the stockholders",
            "letter to the shareholders",
            "to our valued stockholders",
            "to our valued shareholders",
            "shareholders letter",
            "stockholders letter",
            "letter from the chairman",
            "letter from the ceo",
            "letter from the president",
            "dear shareholder",
            "dear stockholder",
            "to shareholders",
            "to stockholders",
            "fellow shareholders",
            "fellow stockholders",
            "message from the ceo",
            "ceo message",
            "ceo's message",
            "message to shareholders",
            "message from our ceo",
        ]
        

        keyword_nodes = []
        for phrase in opening_phrases:
            phrase_lower = phrase.lower()
            for text_node in soup.find_all(string=lambda t: t and phrase_lower in str(t).strip().lower()):
                keyword_nodes.append(text_node)
        
        if not keyword_nodes:
            if logger:
                logger.debug("No keyword nodes found for distance-based search")
            return []
        

        all_imgs = soup.find_all("img")
        if not all_imgs:
            return []
        

        img_to_index = {id(img): idx for idx, img in enumerate(all_imgs)}
        

        keyword_positions = []
        for keyword_node in keyword_nodes:

            keyword_pos = getattr(keyword_node, 'sourceline', None)
            
            if keyword_pos is None:

                parent = keyword_node.parent
                keyword_pos = getattr(parent, 'sourceline', None) if parent else None
            
            if keyword_pos is not None:
                keyword_positions.append(keyword_pos)
        

        use_dom_order = len(keyword_positions) == 0
        
        if use_dom_order:

            all_elements = list(soup.descendants)
            for keyword_node in keyword_nodes:
                try:
                    keyword_idx = all_elements.index(keyword_node)
                    keyword_positions.append(keyword_idx)
                except ValueError:

                    parent = keyword_node.parent
                    if parent:
                        try:
                            keyword_idx = all_elements.index(parent)
                            keyword_positions.append(keyword_idx)
                        except ValueError:
                            pass
        
        if not keyword_positions:
            if logger:
                logger.debug("Could not determine keyword positions")
            return []
        

        candidates_with_distance: List[Tuple[CandidateImage, int]] = []
        seen_urls: set[str] = set()
        

        all_elements_for_img = list(soup.descendants) if use_dom_order else None
        
        for img in all_imgs:
            src = img.get("src")
            if not src:
                    continue
            
            url = build_image_url(src, cik10, accession)
            if url in seen_urls:
                continue
            seen_urls.add(url)
            
            img_position = img_to_index.get(id(img), len(all_imgs))
            

            img_html_pos = getattr(img, 'sourceline', None)
            
            if img_html_pos is None and use_dom_order and all_elements_for_img:
                try:
                    img_html_pos = all_elements_for_img.index(img)
                except ValueError:
                    img_html_pos = img_position * 50
            elif img_html_pos is None:
                img_html_pos = img_position * 50
            

            min_distance = min(abs(img_html_pos - kp) for kp in keyword_positions)
            


            estimated_size = estimate_image_size(img)
            MIN_SIZE = 8_000
            if estimated_size > 0 and estimated_size < MIN_SIZE:
                continue

            if is_extreme_aspect_ratio(img, max_ratio=4.0):
                continue
            if self._should_skip_small_gif(url, estimated_size):
                continue
            

            if img_position == 0:
                continue
            
            candidate = CandidateImage(
                url=url,
                score=0.0,
                stage="keyword_distance_fallback",
                position_index=img_position,
                estimated_size=estimated_size,
            )
            
            candidates_with_distance.append((candidate, int(min_distance)))
        

        candidates_with_distance.sort(key=lambda x: x[1])
        candidates = [cand for cand, _ in candidates_with_distance]
        
        if logger:
            logger.info(
                "Found %d images using keyword-distance fallback (sorted by distance)",
                len(candidates)
            )
        
        return candidates
    
    def _evaluate_candidates_sequential_face_check(
        self,
        candidates: List[CandidateImage],
        ceo_name: str,
        cik10: str,
        accession: str,
        soup: BeautifulSoup,
    ) -> Optional[CandidateImage]:
        """
        Sequential face check fallback.

        Steps:
        1) Inspect candidates in order.
        2) Download each image and check for faces.
        3) Stop and return at the first detected face.
        4) Continue otherwise.

        Returns:
            First candidate with a detected face, or None.
        """
        if self.verbose:
            self.logger.info(
                "Sequential face check: evaluating %d candidates one by one",
                len(candidates)
            )
        
        processed = 0
        pending_dashscope: List[Tuple[CandidateImage, bytes, str, str]] = []
        for candidate in candidates:
            if processed >= MAX_DOWNLOAD_PER_FILING:
                break
            
            processed += 1
            
            if self.verbose:
                self.logger.info(
                    "Checking candidate %d/%d: position=%d url=%s",
                    processed,
                    min(len(candidates), MAX_DOWNLOAD_PER_FILING),
                    candidate.position_index,
                    candidate.url,
                )
            

            if candidate.position_index == 0:
                continue
            if self._should_skip_small_gif(candidate.url, candidate.estimated_size):
                candidate.reason = "gif_small_estimated"
                continue
            

            image_bytes = load_image_bytes(self.session, candidate.url)
            if not image_bytes:
                self._record_filter_event("download_failed")
                continue
            
            candidate.bytes_data = image_bytes
            





            basic_ok, basic_conf, basic_reason, image_size = basic_image_checks(image_bytes)
            if not basic_ok:
                if self.verbose:
                    self.logger.info("❌ Basic check failed: %s", basic_reason)
                self._record_filter_event("basic_fail")
                continue
            actual_area = image_size[0] * image_size[1]
            if self._should_skip_small_gif(candidate.url, actual_area):
                candidate.reason = f"{basic_reason}|gif_small_actual"
                continue
            self._record_filter_event("basic_pass")
            

            all_imgs = soup.find_all("img")
            url_to_img = {}
            for img in all_imgs:
                src = img.get("src")
                if src:
                    url = build_image_url(src, cik10, accession)
                    url_to_img[url] = img
            
            img_tag = url_to_img.get(candidate.url)
            if img_tag:
                should_include, is_retired = self._check_retired_ceo_filter(
                    soup,
                    img_tag,
                    ceo_name,
                    context_lines=15,
                )
                
                if not should_include:
                    if self.verbose:
                        self.logger.info(
                            "⛔ Candidate at position %d filtered: retired CEO detected, skipping...",
                            candidate.position_index
                        )
                    self._record_filter_event("retired_filter_blocked")
                    continue
                
                if is_retired and self.verbose:
                    self.logger.info(
                        "✓ Candidate at position %d: retired CEO context, but CEO name matched - exempted",
                        candidate.position_index
                    )

            

            face_ok = False
            face_conf = 0.0
            face_boxes = None
            
            if cv2 is not None:
                face_ok, face_conf, face_reason, face_boxes = detect_face_with_cv(image_bytes)
                if face_ok and face_conf < MIN_OPENCV_FACE_CONFIDENCE:
                    if self.verbose:
                        self.logger.info(
                            "Face confidence %.2f below threshold %.2f, skipping candidate",
                            face_conf,
                            MIN_OPENCV_FACE_CONFIDENCE,
                        )
                    face_ok = False
                    face_reason = "face_conf_too_low"
                if face_ok:
                    if self.verbose:
                        self.logger.info(
                            "✓ Face detected by OpenCV (position=%d, confidence=%.2f)",
                            candidate.position_index,
                            face_conf
                        )
                    self._record_filter_event("face_pass")
                    
                    candidate.photo_prominence = calculate_photo_prominence(image_size, face_boxes)
                    

                    base_reason = "sequential_face_check|face_detected|ceo_context_ok"
                    if self.dashscope_client.is_ready():
                        pending_dashscope.append((candidate, image_bytes, ceo_name, base_reason))
                        selected = self._maybe_flush_dashscope_queue(
                            pending_dashscope,
                            stage_hint="sequential_face_check",
                        )
                        if selected:
                            return selected
                    else:
                        if self.verbose:
                            self.logger.warning("DashScope not available, rejecting image to ensure all photos in pool are validated")
                        candidate.reason = f"{base_reason}|dashscope_unavailable|rejected"
                        self._record_filter_event("dashscope_unavailable")
                        continue
                else:
                    if self.verbose:
                        self.logger.info(
                            "No face detected, continuing to next candidate..."
                        )
                    if face_reason == "face_conf_too_low":
                        candidate.reason = "face_conf_too_low"
                        self._record_filter_event("face_conf_too_low")
                    else:
                        self._record_filter_event("face_fail")
            else:
                if self.verbose:
                    self.logger.info("OpenCV not available, skipping face check")
        
        selected_final = self._maybe_flush_dashscope_queue(
            pending_dashscope,
            stage_hint="sequential_face_check",
            force=True,
        )
        if selected_final:
            return selected_final

        if self.verbose:
            self.logger.warning("No face detected in any candidate")
        
        return None

    def _persist_photo(
        self,
        image_bytes: bytes,
        company_dir: Path,
        accession_dir: Path,
        year: int,
        cik10: str,
    ) -> Tuple[Path, Path]:
        year_path = accession_dir / "ceo_photo"
        year_path.mkdir(parents=True, exist_ok=True)
        sec_path = year_path / f"{year}.png"
        sec_path.write_bytes(image_bytes)

        shared_path = self.shared_output_root / f"{cik10}_{year}.png"
        shared_path.write_bytes(image_bytes)
        return sec_path, shared_path

    def _write_results_json(self, results: List[PhotoResult]) -> None:
        serializable: List[Dict[str, Any]] = []
        for item in results:
            record = asdict(item)
            if item.sec_local_path is not None:
                record["sec_local_path"] = str(item.sec_local_path)
            if item.shared_local_path is not None:
                record["shared_local_path"] = str(item.shared_local_path)
            serializable.append(record)

        self.results_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.results_output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        success_count = sum(1 for item in results if item.status == "success")
        self.logger.info(
            "Extraction finished: %d total, %d success. Results saved to %s",
            len(results),
            success_count,
            self.results_output_path,
        )




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CEO photo extraction pipeline")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("scrapephoto") / "execucomp_company_identifiers_annual_2000_2025_with_ceo_FINAL.csv",
        help="Path to the CSV source file.",
    )
    parser.add_argument(
        "--sec-data",
        type=Path,
        default=Path("D:/sec_data"),
        help="Root directory containing DEF 14A filings.",
    )
    parser.add_argument(
        "--shared-output",
        type=Path,
        default=Path("D:/ceo_photo_pool"),
        help="Directory where consolidated CEO photos will be stored.",
    )
    parser.add_argument(
        "--results-output",
        type=Path,
        default=Path("scrapephoto") / "ceo_photo_results.json",
        help="Path for the JSON output summarizing extraction results.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of entries to process for debugging.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of entries to skip before starting processing (default: 0).",
    )
    parser.add_argument(
        "--year-start",
        type=int,
        default=None,
        help="Start year (inclusive) for filtering, e.g., 2015.",
    )
    parser.add_argument(
        "--year-end",
        type=int,
        default=None,
        help="End year (inclusive) for filtering, e.g., 2023.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("scrapephoto") / "ceo_photo_pipeline.log",
        help="Optional log file path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging.",
    )
    parser.add_argument(
        "--start-delay-minutes",
        type=int,
        default=0,
        help="Optional delay (minutes) before pipeline run begins, e.g., 10 to wait ten minutes.",
    )
    parser.add_argument(
        "--strategy-limit",
        type=int,
        default=None,
        help="Optional cap on the number of strategies to attempt (set to 4 to use only the first four).",
    )
    parser.add_argument(
        "--dashscope-concise",
        action="store_true",
        help="Use a compact DashScope prompt/response that only reports pass/fail to reduce latency.",
    )
    parser.add_argument(
        "--dashscope-async",
        action="store_true",
        help="Enable DashScope async task submission with polling + 15 QPS throttle.",
    )
    parser.add_argument(
        "--dashscope-async-qps",
        type=int,
        default=15,
        help="Max QPS allowed while polling async tasks (default: 15).",
    )
    parser.add_argument(
        "--dashscope-async-poll-interval",
        type=float,
        default=2.0,
        help="Seconds to wait between async task polling attempts (default: 2.0).",
    )
    parser.add_argument(
        "--dashscope-async-timeout",
        type=float,
        default=30.0,
        help="Maximum seconds to wait for async completion before marking as timeout (default: 60).",
    )
    parser.add_argument(
        "--skip-existing-companies",
        action="store_true",
        help="Skip processing for a company if ANY photo for that company (any year) already exists in the output pool.",
    )
    parser.add_argument(
        "--only-2021-year",
        action="store_true",
        help="Only process photos for the year 2021, skip all other years.",
    )
    parser.add_argument(
        "--strategy-0-5-only",
        action="store_true",
        help="Only use Strategy 0.5 (CEO name-based search with nearby 20 images), skip all other strategies.",
    )
    parser.add_argument(
        "--brute-force-strategy",
        action="store_true",
        help="Brute force mode: exclude first page, check ALL images that meet size requirements through OpenCV face detection and DashScope. First face found goes to pool, then move to next company.",
    )
    return parser.parse_args()


def configure_logging(log_file: Path, verbose: bool = False) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers: List[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode="a", encoding="utf-8"),
    ]
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
    )
    
    # Suppress verbose DEBUG logs from third-party libraries
    logging.getLogger("dashscope").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_file, verbose=args.verbose)
    dashscope_client = DashscopeVisionClient(
        concise_mode=args.dashscope_concise,
        async_mode=args.dashscope_async,
        max_qps=args.dashscope_async_qps,
        poll_interval=args.dashscope_async_poll_interval,
        async_timeout=args.dashscope_async_timeout,
    )

    if args.start_delay_minutes and args.start_delay_minutes > 0:
        delay_seconds = args.start_delay_minutes * 60
        logging.getLogger("ceo_photo_pipeline").info(
            "Start delay requested: waiting %d minute(s) (%.0f seconds) before execution",
            args.start_delay_minutes,
            delay_seconds,
        )
        time.sleep(delay_seconds)

    pipeline = CEOPhotoPipeline(
        csv_path=args.csv,
        sec_data_root=args.sec_data,
        shared_output_root=args.shared_output,
        results_output_path=args.results_output,
        dashscope_client=dashscope_client,
        limit=args.limit,
        offset=args.offset,
        year_start=args.year_start,
        year_end=args.year_end,
        verbose=args.verbose,
        strategy_limit=args.strategy_limit,
        skip_existing_companies=args.skip_existing_companies,
        only_2021_year=args.only_2021_year,
        strategy_0_5_only=args.strategy_0_5_only,
        brute_force_strategy=args.brute_force_strategy,
    )
    pipeline.run()


if __name__ == "__main__":
    main()


# python scrapephoto/ceo_photo_pipeline_test.py --offset 500 --year-start 2019 --verbose
# python scrapephoto/ceo_photo_pipeline_test.py --year-start 2019 --verbose
# python scrapephoto/ceo_photo_pipeline_test.py --offset 500 --year-start 2019 --start-delay-minutes 8
# python scrapephoto/ceo_photo_pipeline_test.py --offset 505 --year-start 2019 --strategy-limit 4
# python scrapephoto/ceo_photo_pipeline_test.py --dashscope-concise --strategy-limit 3 --offset 5000
# python scrapephoto/ceo_photo_pipeline_test.py --skip-existing-companies --year-start 1500
# python scrapephoto/ceo_photo_pipeline_test.py --only-2021-year --skip-existing-companies --offset 15000

# python scrapephoto/ceo_photo_pipeline_test.py --skip-existing-companies --strategy-0-5-only --only-2021-year 

# python scrapephoto/ceo_photo_pipeline_test.py --brute-force-strategy --skip-existing-companies --only-2021-year 
