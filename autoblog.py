#!/usr/bin/env python3
"""
autoblog.py - one researched, AEO/SEO-optimised post per site per run.

Flow per site:
  1. Read existing WordPress posts (dedupe + internal-link candidates)
  2. Ask Gemini (free tier, Google Search grounding) for ONE fresh real question in today's niche
  3. Ask Gemini (free tier, Google Search grounding) to write the article as JSON
  4. Validate (answer-first, 3-4 dofollow sources that load, internal links, length)
  5. Publish through the WordPress REST API (status from sites.json) or save locally

Usage:
  python autoblog.py --dry-run            # writes to out/, touches nothing
  python autoblog.py --site techybio      # one site only
  python autoblog.py                      # all sites, uses "status" from sites.json
"""
import argparse
import datetime as dt
import difflib
import html
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from urllib.parse import urlparse

from google import genai
from google.genai import types
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).parent
STATE_DIR = ROOT / "state"
OUT_DIR = ROOT / "out"
MODELS = [m.strip() for m in os.getenv("GEMINI_MODELS", "gemini-3.5-flash,gemini-3.5-flash-lite").split(",")]
SITE_DELAY = int(os.getenv("SITE_DELAY", "45"))  # seconds between sites (free-tier RPM)
UA = {"User-Agent": "Mozilla/5.0 (compatible; AutoBlogBot/1.0)"}
BANNED = ["in today's digital landscape", "it's important to note", "dive into", "game-changer",
          "delve", "revolutionize", "seamlessly", "cutting-edge", "harness the power", "tapestry"]

OVERLOADED = set()  # models returning 503 'high demand' this run
BAD_MODELS = set()  # models that returned MALFORMED_FUNCTION_CALL this run
NO_GROUND = False  # set True when Google Search grounding is refused on this key
BAD_SRC = {"reddit.com", "quora.com", "medium.com", "facebook.com", "linkedin.com", "pinterest.com",
           "youtube.com", "x.com", "twitter.com", "tiktok.com", "instagram.com"}

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))


# --------------------------------------------------------------------------- helpers
def today():
    return dt.date.today()


def host_of(url):
    return urlparse(url).netloc.lower().replace("www.", "")


def root_domain(h):
    p = h.split(".")
    if len(p) >= 3 and ".".join(p[-2:]) in ("co.uk", "org.uk", "gov.uk", "ac.uk", "com.au", "co.in", "gov.in", "com.br"):
        return ".".join(p[-3:])
    return ".".join(p[-2:])


def norm(url):
    return url.split("#")[0].rstrip("/").lower()


def _resolve(uri):
    try:
        return requests.get(uri, headers=UA, timeout=15, allow_redirects=True, stream=True).url
    except requests.RequestException:
        return uri


def ask(system, user, max_tokens=12000, searches=0, need_json=False, search=True, check=None):
    """Call Gemini (free tier). Tries Google Search grounding first, then falls back to no-search.
    Returns (text, grounded, domains_seen_in_search)."""
    global NO_GROUND
    last = None
    for model in MODELS:
        for grounded in ([True, False] if (search and not NO_GROUND) else [False]):
            for attempt in range(4):
                if model in BAD_MODELS or model in OVERLOADED:
                    break
                try:
                    cfg = types.GenerateContentConfig(
                        system_instruction=system, max_output_tokens=max_tokens, temperature=0.6,
                        tools=[types.Tool(google_search=types.GoogleSearch())] if grounded else None)
                    r = client.models.generate_content(model=model, contents=user, config=cfg)
                    text = r.text or ""
                    fr = r.candidates[0].finish_reason if r.candidates else None
                    print(f"  [gemini {model} grounded={grounded}] finish={fr} chars={len(text)}")
                    if "MALFORMED" in str(fr):
                        BAD_MODELS.add(model)
                        last = RuntimeError(f"{model} returned MALFORMED_FUNCTION_CALL; skipping it for this run")
                        print("  ->", last)
                        break
                    checker = check or (extract_json if need_json else None)
                    if checker:
                        try:
                            checker(text)
                        except Exception as ce:
                            last = RuntimeError(f"{model} grounded={grounded}: reply unusable "
                                                f"(finish={fr}, chars={len(text)}): {ce}")
                            print("  ->", last)
                            break  # try next option (no-search / next model)
                    domains = set()
                    if grounded:
                        gm = getattr(r.candidates[0], "grounding_metadata", None)
                        for ch in (getattr(gm, "grounding_chunks", None) or []):
                            if getattr(ch, "web", None):
                                if ch.web.title:
                                    domains.add(ch.web.title.lower().replace("www.", ""))
                                if ch.web.uri:
                                    domains.add(host_of(_resolve(ch.web.uri)))
                    return text, bool(grounded and domains), domains
                except Exception as e:  # rate limit, overload, quota, unsupported tool...
                    last = e
                    msg = str(e)
                    print(f"  [gemini {model} grounded={grounded}] error: {msg[:240]}")
                    quota = "429" in msg or "RESOURCE_EXHAUSTED" in msg
                    if grounded and quota:
                        NO_GROUND = True  # Search grounding quota is 0 / used up on this key: don't wait for it
                        print("  -> Google Search grounding quota not available; using own research instead")
                        break
                    if "503" in msg or "UNAVAILABLE" in msg:
                        if attempt >= 1:  # still busy after 2 tries: use the other model for the rest of this run
                            OVERLOADED.add(model)
                            print(f"  -> {model} is overloaded; switching models for the rest of this run")
                            break
                        time.sleep(15)
                        continue
                    if any(k in msg for k in ("429", "500", "RESOURCE_EXHAUSTED", "DEADLINE")):
                        time.sleep(20 * (attempt + 1))  # temporary: wait and retry
                        continue
                    if grounded:
                        NO_GROUND = True
                        print("  -> Google Search grounding unavailable on this key; using own research instead")
                    break  # permanent for this option -> try the next one
    raise RuntimeError(f"Gemini unavailable (free quota may be used up): {last}")


def extract_json(text):
    t = (text or "").strip()
    m = re.search(r"<json>(.*?)(</json>|$)", t, re.S)
    if m:
        t = m.group(1).strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    i = t.find("{")
    if i < 0:
        raise ValueError("no JSON object found in model reply")
    obj, _ = json.JSONDecoder().raw_decode(t[i:])
    return obj


# --------------------------------------------------------------------------- WordPress
def wp(site, method, path, **kw):
    auth = (os.environ[site["wp_user_env"]], os.environ[site["wp_pass_env"]])
    url = site["url"].rstrip("/") + "/wp-json/wp/v2/" + path
    r = requests.request(method, url, auth=auth, headers=UA, timeout=90, **kw)
    r.raise_for_status()
    return r.json()


def existing_posts(site):
    posts = wp(site, "GET", "posts", params={"per_page": 100, "status": "publish,draft,future,pending",
                                              "orderby": "date", "_fields": "title,link"})
    return [{"title": html.unescape(p["title"]["rendered"]), "url": p["link"]} for p in posts]


def category_id(site, name):
    for c in wp(site, "GET", "categories", params={"per_page": 100, "_fields": "id,name"}):
        if html.unescape(c["name"]).lower() == name.lower():
            return c["id"]
    return None


def category_links(site):
    cats = wp(site, "GET", "categories", params={"per_page": 100, "hide_empty": "true", "_fields": "name,link"})
    return [{"title": html.unescape(c["name"]) + " (category)", "url": c["link"]} for c in cats]


# --------------------------------------------------------------------------- state
def load_state(key):
    f = STATE_DIR / f"{key}.json"
    return json.loads(f.read_text()) if f.exists() else {"used": []}


def save_state(key, state):
    STATE_DIR.mkdir(exist_ok=True)
    (STATE_DIR / f"{key}.json").write_text(json.dumps(state, indent=2))


# --------------------------------------------------------------------------- steps
def pick_topic(site, niche, titles, state, avoid=()):
    used = [u["question"] for u in state["used"]][-150:] + list(avoid)
    system = ("You are a senior SEO/AEO editor. You find real questions people actually ask, "
              "then choose the one with clear search intent that the site has not covered.")
    user = f"""Today is {today():%B %d, %Y}.
Site: {site['url']}  |  Audience: {site['audience']}
Niche for today: {niche}
Scope rule: {site['avoid']}

Search Reddit threads, forums and Google 'People Also Ask' style questions for REAL, current questions in this niche.
Pick ONE question that:
- is asked repeatedly or is trending right now
- has a clear, answerable intent (what/how/why/is/does/should/vs)
- is NOT already covered by these existing titles: {json.dumps(titles[:150])}
- is NOT in this list of earlier questions: {json.dumps(used)}

Reply ONLY with:
<json>{{"question": "...", "primary_keyword": "...", "search_intent": "...", "template": "how-to|comparison|faq-knowledge|news-analysis|product-review|listicle"}}</json>"""
    text, _, _ = ask(system, user, max_tokens=4000, need_json=True)
    return extract_json(text)


def fetch_text(url):
    try:
        r = requests.get(url, headers=UA, timeout=20)
        if r.status_code >= 400 or "html" not in r.headers.get("content-type", ""):
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            t.decompose()
        return " ".join(soup.get_text(" ", strip=True).split())
    except requests.RequestException:
        return ""


def auth_domains(site):
    if site.get("ymyl"):
        return set(site.get("trusted_domains", []))
    extra = [d for lst in site.get("niche_authority", {}).values() for d in lst]
    return set(site.get("authority_domains", [])) | set(extra)


def is_authority(site, h):
    return h.endswith((".gov", ".edu", ".gov.uk", ".ac.uk", ".gov.in")) or any(
        h == d or h.endswith("." + d) for d in auth_domains(site))


def research_pack(site, topic, want=6, niche=None):
    """Free web research: DuckDuckGo -> authoritative domains first -> download pages -> keep readable text.
    Every URL in the pack was downloaded successfully, so it is a real, live page."""
    try:
        from ddgs import DDGS
    except Exception as e:
        print("  research: ddgs not installed:", e)
        return []
    base = site.get("trusted_domains", []) if site.get("ymyl") else site.get("authority_domains", [])
    lead = (site.get("niche_authority", {}).get(niche or "", []) or base)[:9]  # best domains for TODAY's niche
    queries = [f'{topic["question"]} (' + " OR ".join("site:" + d for d in lead[i:i + 3]) + ")"
               for i in range(0, len(lead), 3)]
    if not site.get("ymyl"):
        queries += [topic["question"], f'{topic["primary_keyword"]} {today().year} statistics']
    me = host_of(site["url"])
    found, seen_hosts = [], set()
    for q in queries:
        time.sleep(1.2)
        try:
            results = DDGS().text(q, max_results=8) or []
        except Exception as e:
            print("  research: search error:", str(e)[:120])
            continue
        for r in results:
            url = r.get("href") or r.get("url") or ""
            h = host_of(url) if url else ""
            if not url or root_domain(h) in seen_hosts or root_domain(h) == root_domain(me) or any(h == b or h.endswith("." + b) for b in BAD_SRC):
                continue
            if site.get("ymyl") and not is_authority(site, h):
                continue
            seen_hosts.add(root_domain(h))
            found.append((url, r.get("title", ""), is_authority(site, h)))
    found.sort(key=lambda x: not x[2])  # authority first
    pack = []
    for url, title, auth in found:
        txt = fetch_text(url)
        if len(txt) > 600:
            pack.append({"url": url, "title": title, "text": txt[:3000], "authority": auth})
        if len(pack) >= want:
            break
    return pack


def tag(text, name):
    m = re.search(rf"<{name}>(.*?)</{name}>", text, re.S)
    return m.group(1).strip() if m else ""


def parse_article(text):
    art = {k: tag(text, k) for k in ("title", "slug", "meta_description", "excerpt", "tags", "image_query")}
    body = tag(text, "article_html")
    body = re.sub(r"^```(?:html)?|```$", "", body, flags=re.M).strip()
    if not body or not art["title"]:
        raise ValueError("missing <title> or <article_html> block")
    art["content_html"] = body
    art["tags"] = [t.strip() for t in art["tags"].split(",") if t.strip()]
    art["slug"] = re.sub(r"[^a-z0-9-]+", "-", (art["slug"] or art["title"]).lower()).strip("-")[:80]
    soup = BeautifulSoup(body, "html.parser")
    faq = []
    h2 = next((h for h in soup.find_all("h2") if "frequently asked" in h.get_text().lower()), None)
    if h2:
        for el in h2.find_next_siblings():
            if el.name == "h2":
                break
            if el.name == "h3":
                p = el.find_next_sibling("p")
                if p:
                    faq.append({"q": el.get_text(" ", strip=True), "a": p.get_text(" ", strip=True)})
    art["faq"] = faq
    return art


def fix_internal_links(site, html_, allowed_map):
    """Models sometimes mangle our own URLs. Snap each internal link to the closest real URL, or drop it."""
    soup = BeautifulSoup(html_, "html.parser")
    me, keys = host_of(site["url"]), sorted(allowed_map)
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("http") and host_of(href) == me and norm(href) not in allowed_map:
            m = difflib.get_close_matches(norm(href), keys, n=1, cutoff=0.6)
            if m:
                print(f"  fixed internal link {href} -> {allowed_map[m[0]]}")
                a["href"] = allowed_map[m[0]]
            else:
                print(f"  dropped unknown internal link {href}")
                a.unwrap()
    return str(soup)


def ensure_links(site, art, pack):
    """Guarantee 3-4 distinct external links: trim extras, top up from the downloaded sources."""
    soup = BeautifulSoup(art["content_html"], "html.parser")
    me = host_of(site["url"])
    seen = {}
    for a in soup.find_all("a", href=True):
        h = a["href"]
        if h.startswith("http") and host_of(h) != me:
            seen.setdefault(norm(h), []).append(a)
    if len(seen) > 4:
        keep = list(seen)[:4]
        for k, anchors in seen.items():
            if k not in keep:
                for a in anchors:
                    a.unwrap()
        seen = {k: seen[k] for k in keep}
    if len(seen) < 3 and pack:
        hosts = {root_domain(host_of(k)) for k in seen}
        extra = []
        for p in sorted(pack, key=lambda p: not p.get("authority")):
            if norm(p["url"]) in seen or root_domain(host_of(p["url"])) in hosts:
                continue
            extra.append(p)
            hosts.add(root_domain(host_of(p["url"])))
            if len(seen) + len(extra) >= 3:
                break
        if extra:
            li = "".join(f'<li><a href="{p["url"]}" target="_blank" rel="noopener">'
                         f'{html.escape((p["title"] or host_of(p["url"]))[:90])}</a> ({host_of(p["url"])})</li>'
                         for p in extra)
            soup.append(BeautifulSoup(f"<h2>Further reading</h2><ul>{li}</ul>", "html.parser"))
            print(f"  added {len(extra)} 'Further reading' source(s) to reach 3 external links")
    art["content_html"] = str(soup)


def write_article(site, topic, link_pool, feedback=None, pack=None, prev=None):
    ymyl_rules = ""
    if site.get("ymyl"):
        ymyl_rules = (
            "\n- HEALTH CONTENT (YMYL): cite only government, university, major medical-organisation or "
            "peer-reviewed sources. Give NO personal dosing, diagnosis or treatment instructions. Say when to "
            "talk to a doctor. Include one short 'not medical advice' paragraph before the FAQ. "
            "Describe uncertainty honestly; never overstate study results."
        )
    system = ("You write accurate, original, genuinely useful articles optimised for Google and for AI answer "
              "engines (AEO/GEO). Every factual claim must come from pages you actually found with Google Search. "
              "Never invent statistics, quotes, studies or URLs. Write like a knowledgeable human editor.")
    pool = "\n".join(f"- {p['title']}: {p['url']}" for p in link_pool[:60])
    fb = ""
    if feedback:
        fb = ("\n\nYOUR PREVIOUS DRAFT FAILED THESE CHECKS - FIX ALL OF THEM:\n- " + "\n- ".join(feedback))
        if prev:
            fb += "\n\nPREVIOUS DRAFT (revise and EXTEND it, keep what works):\n" + prev
    if pack:
        src = "\n\n".join(f"[{i+1}] {'(AUTHORITY) ' if p.get('authority') else ''}{p['title']}\nURL: {p['url']}\nEXCERPT: {p['text']}" for i, p in enumerate(pack))
        research_block = ("SOURCE MATERIAL (downloaded today; the ONLY facts you may state as fact come from here or "
                          "are plain common knowledge). Paraphrase in your own words, never copy sentences. "
                          "Do not call any tools or functions.\n\n" + src)
        ext_rule = ("EXACTLY 3 or 4 external links, each URL copied EXACTLY from the SOURCE MATERIAL list above "
                    "(different domains), and at least one must be marked (AUTHORITY), preferably two. Never use any other external URL.")
    else:
        research_block = "Research first: search for current (2025-2026) data and authoritative sources."
        ext_rule = "EXACTLY 3 or 4 external links to DIFFERENT authoritative domains you actually read."
    user = f"""Today is {today():%B %d, %Y}. Write one article for {site['url']}.
Question to answer: {topic['question']}
Primary keyword: {topic['primary_keyword']}  |  Intent: {topic['search_intent']}  |  Template: {topic['template']}
Audience: {site['audience']}

{research_block}

HARD RULES for content_html (HTML only, no <html>/<body>, no <h1>):
1. The FIRST element must be a <p> of 40-60 words that directly answers the question (this is the AEO answer).
2. Then a Key Takeaways box: <div class="key-takeaways"><strong>Key takeaways</strong><ul>3-5 <li></ul></div>
3. 4-6 <h2> sections (question-style where natural), <h3> only inside them. Short paragraphs, plain language,
   at least one list or table where it helps. Write about {site['min_words'] + 400} words (never fewer than {site['min_words'] + 100}): develop each section with examples, steps and numbers from the sources.
4. {ext_rule} Plain <a href="..."> with
   target="_blank" rel="noopener" (dofollow: never nofollow/sponsored/ugc). Name the source in the sentence.
   Put them inside the text where the claim is made, and list them again under a final <h2>Sources</h2>.
5. 3 or 4 internal links, chosen ONLY from this list, URL copied exactly, with descriptive anchor text:
{pool}
6. An <h2>Frequently Asked Questions</h2> with 3-5 related questions (<h3> question + <p> answer, 30-60 words each).
7. Add original value: a concrete example, a decision rule, or a common mistake - not just a summary of sources.
8. Do not use these phrases: {', '.join(BANNED)}.{ymyl_rules}

Reply ONLY in this exact tag format (HTML goes directly inside <article_html>; no JSON, no markdown fences):
<title>title, max 60 chars, includes the keyword</title>
<slug>url-slug</slug>
<meta_description>max 155 chars</meta_description>
<excerpt>1-2 sentences</excerpt>
<tags>3-5 comma-separated tags</tags>
<image_query>2-4 plain words describing a concrete, photographable scene for the featured image (e.g. 'woman typing laptop office'); no brands, no text, no logos</image_query>
<article_html>
...the full article HTML...
</article_html>{fb}"""
    text, grounded, domains = ask(system, user, max_tokens=24000, check=parse_article, search=not pack)
    art = parse_article(text)
    art["_grounded"], art["_domains"] = grounded, domains
    if pack:  # self-sourced: links must be exactly the pages we downloaded
        art["_grounded"] = True
        art["_domains"] = {host_of(p["url"]) for p in pack}
        art["_pack_urls"] = {norm(p["url"]) for p in pack}
        art["_min_auth"] = min(2, sum(1 for p in pack if p.get("authority")))
    return art


def url_alive(url):
    try:
        r = requests.get(url, headers=UA, timeout=20, allow_redirects=True, stream=True)
        return r.status_code < 400 or r.status_code in (401, 403, 429)  # bot-blocks are not dead links
    except requests.RequestException:
        return False


def validate(site, art, internal_ok):
    problems = []
    soup = BeautifulSoup(art["content_html"], "html.parser")
    words = len(soup.get_text(" ", strip=True).split())
    if words < site["min_words"]:
        problems.append(f"too short: {words} words, need {site['min_words']}+")
    first = soup.find(True)
    if not first or first.name != "p":
        problems.append("first element must be the direct-answer <p>")
    elif not 25 <= len(first.get_text().split()) <= 70:
        problems.append(f"answer paragraph is {len(first.get_text().split())} words; make it 40-60")
    if soup.find("h1"):
        problems.append("remove <h1>")
    if len(soup.find_all("h2")) < 4:
        problems.append("need at least 4 <h2> sections")
    if len(art.get("faq", [])) < 3:
        problems.append("need 3+ FAQ items")
    low = soup.get_text(" ").lower()
    for b in BANNED:
        if b in low:
            problems.append(f"banned phrase: {b}")

    me = host_of(site["url"])
    ext, internal = {}, set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.startswith("http"):
            continue
        h = host_of(href)
        if h == me:
            internal.add(norm(href))
        else:
            rel = " ".join(a.get("rel", []))
            if any(x in rel for x in ("nofollow", "sponsored", "ugc")):
                problems.append(f"external link must be dofollow: {href}")
            ext[norm(href)] = (h, href)
    if not 3 <= len(ext) <= 4:
        problems.append(f"need 3-4 distinct external links, found {len(ext)}")
    if len({root_domain(h) for h, _ in ext.values()}) < len(ext):
        problems.append("external links must go to different domains")
    if len(internal) < 2:
        problems.append("need at least 2 internal links")
    for i in internal:
        if i not in internal_ok:
            problems.append(f"internal link not in allowed list: {i}")
    for _, href in ext.values():
        if not url_alive(href):
            problems.append(f"source URL does not load: {href}")
    if art.get("_pack_urls"):
        need = art.get("_min_auth", 2)
        if sum(1 for h, _ in ext.values() if is_authority(site, h)) < need:
            problems.append(f"at least {need} external links must be marked (AUTHORITY) in the SOURCE MATERIAL")
        for k, (h, href) in ext.items():
            if k not in art["_pack_urls"]:
                problems.append(f"external link is not one of the provided SOURCE MATERIAL urls: {href}")
    elif art.get("_grounded"):
        seen = art["_domains"]
        for h, href in ext.values():
            if not any(h == d or h.endswith("." + d) or d.endswith(h) for d in seen):
                problems.append(f"link not found in your search results (possible invented source): {href}")
    if site.get("ymyl"):
        trusted = site.get("trusted_domains", [])
        good = [h for h, _ in ext.values() if any(h == t or h.endswith("." + t) for t in trusted)]
        if len(good) < 2:
            problems.append("health post needs 2+ sources from trusted medical/government domains")
    return problems


def schema_block(site, art, url_hint):
    graph = [{
        "@context": "https://schema.org", "@type": "Article", "headline": art["title"],
        "description": art["meta_description"], "datePublished": today().isoformat(),
        "dateModified": today().isoformat(),
        "author": {"@type": "Person", "name": site["author_name"]},
        "publisher": {"@type": "Organization", "name": host_of(site["url"])},
        "mainEntityOfPage": url_hint,
    }, {
        "@context": "https://schema.org", "@type": "FAQPage",
        "mainEntity": [{"@type": "Question", "name": f["q"],
                        "acceptedAnswer": {"@type": "Answer", "text": f["a"]}} for f in art["faq"]],
    }]
    return "\n".join(f'<script type="application/ld+json">{json.dumps(g, ensure_ascii=False)}</script>' for g in graph)


# --------------------------------------------------------------------------- images
def find_images(queries, used_ids, n=2, keyword=""):
    """Free stock photos from Pexels (needs PEXELS_API_KEY). Returns up to n different photos."""
    key = os.getenv("PEXELS_API_KEY")
    if not key:
        print("  image: PEXELS_API_KEY not set, posting without images")
        return []
    out, seen = [], set(used_ids)
    for q in [q for q in queries if q]:
        if len(out) >= n:
            break
        try:
            r = requests.get("https://api.pexels.com/v1/search", timeout=30, headers={"Authorization": key, **UA},
                             params={"query": q, "per_page": 15, "orientation": "landscape", "size": "large"})
            r.raise_for_status()
            for p in r.json().get("photos", []):
                if p["id"] in seen or p.get("width", 0) < 1600:
                    continue
                seen.add(p["id"])
                alt = (p.get("alt") or "").strip() or f"{q} - {keyword}".strip(" -")
                out.append({"id": p["id"], "url": p["src"]["large2x"], "photographer": p["photographer"],
                            "page": p["url"], "alt": alt[:125], "query": q})
                if len(out) >= n:
                    break
        except Exception as e:
            print("  image search error:", str(e)[:150])
    return out


def credit_html(img):
    return (f'Photo by <a href="{img["page"]}" target="_blank" rel="noopener">{html.escape(img["photographer"])}</a>'
            ' on <a href="https://www.pexels.com" target="_blank" rel="noopener">Pexels</a>')


def upload_image(site, img, slug):
    """Upload to the WordPress media library with alt text, title and credit caption."""
    data = requests.get(img["url"], headers=UA, timeout=90).content
    auth = (os.environ[site["wp_user_env"]], os.environ[site["wp_pass_env"]])
    r = requests.post(site["url"].rstrip("/") + "/wp-json/wp/v2/media", auth=auth, timeout=180, data=data,
                      headers={**UA, "Content-Disposition": f'attachment; filename="{slug}.jpg"',
                               "Content-Type": "image/jpeg"})
    r.raise_for_status()
    j = r.json()
    wp(site, "POST", f"media/{j['id']}", json={"alt_text": img["alt"], "title": img["alt"],
                                              "caption": credit_html(img), "description": img["alt"]})
    md = j.get("media_details") or {}
    return {"id": j["id"], "url": j.get("source_url") or img["url"], "w": md.get("width"), "h": md.get("height")}


def figure_html(src, img, w=None, h=None):
    size = f' width="{w}" height="{h}"' if w and h else ""
    return (f'<figure class="wp-block-image size-large"><img src="{src}" alt="{html.escape(img["alt"], quote=True)}"'
            f'{size} loading="lazy" /><figcaption>{credit_html(img)}</figcaption></figure>')


def insert_figure(html_, fig):
    """Put the image after the first section (before the 2nd <h2>); the answer paragraph stays first."""
    soup = BeautifulSoup(html_, "html.parser")
    node = BeautifulSoup(fig, "html.parser")
    h2s = soup.find_all("h2")
    if len(h2s) >= 2:
        h2s[1].insert_before(node)
    else:
        p = soup.find("p")
        (p.insert_after(node) if p else soup.append(node))
    return str(soup)


def tag_ids(site, names):
    ids = []
    for n in names[:5]:
        n = n.strip()[:40]
        if not n:
            continue
        try:
            found = wp(site, "GET", "tags", params={"search": n, "per_page": 20, "_fields": "id,name"})
            tid = next((t["id"] for t in found if html.unescape(t["name"]).lower() == n.lower()), None)
            if tid is None:
                try:
                    tid = wp(site, "POST", "tags", json={"name": n})["id"]
                except requests.HTTPError as e:  # tag already exists -> WordPress returns its id
                    tid = ((e.response.json() if e.response is not None else {}).get("data") or {}).get("term_id")
            if tid:
                ids.append(tid)
        except Exception as e:
            print(f"  tag '{n}' skipped:", str(e)[:120])
    return ids


# --------------------------------------------------------------------------- per-site run
def run_site(site, dry_run):
    key = site["key"]
    print(f"\n=== {key} ===")
    state = load_state(key)
    niches = list(site["niches"].items())
    niche, cat_name = niches[today().toordinal() % len(niches)]

    posts = existing_posts(site) if not dry_run or os.getenv(site["wp_pass_env"]) else []
    titles = [p["title"] for p in posts]
    link_pool = posts + (category_links(site) if posts else [])
    internal_ok = {norm(p["url"]) for p in link_pool} | {norm(site["url"])}
    allowed_map = {norm(p["url"]): p["url"] for p in link_pool}
    allowed_map[norm(site["url"])] = site["url"]

    tried, pack, topic = [], None, None
    for i in range(3):
        topic = pick_topic(site, niche, titles, state, avoid=tried)
        print(f"niche={niche} | question={topic['question']}")
        pk = research_pack(site, topic, niche=niche)
        auth_n = sum(1 for p in pk if p.get("authority"))
        print(f"research pack: {len(pk)} sources ({auth_n} authoritative) -> {[host_of(p['url']) for p in pk]}")
        if len(pk) >= 3 and auth_n >= (2 if i < 2 else 1):  # last try accepts a single authority source
            pack = pk
            break
        tried.append(topic["question"])
        if not NO_GROUND:
            break  # Gemini's own Google Search can source this topic
        print("  research: not enough real sources for this question, trying another one")
    else:
        raise RuntimeError(f"{key}: no verifiable sources found for 3 different questions; nothing was written")

    feedback, art, prev = None, None, None
    for attempt in range(1, 4):
        art = write_article(site, topic, link_pool, feedback, pack, prev)
        art["content_html"] = fix_internal_links(site, art["content_html"], allowed_map)
        if pack:
            ensure_links(site, art, pack)
        prev = art["content_html"]
        problems = validate(site, art, internal_ok)
        if not problems:
            break
        print(f"attempt {attempt} failed: {problems}")
        feedback = problems
    else:
        raise RuntimeError(f"{key}: article failed validation after 3 attempts: {problems}")

    want = 2 if site.get("inline_image", True) else 1
    imgs = find_images([art.get("image_query"), topic["primary_keyword"], niche], set(state.get("images", [])),
                       n=want, keyword=topic["primary_keyword"])
    print(f"  images found: {[i['url'] for i in imgs] or 'none'}")
    featured, inline = (imgs + [None, None])[:2]
    uploaded = {}
    if not dry_run:  # upload now so the in-article image uses your own media-library URL
        for name, im in (("featured", featured), ("inline", inline)):
            if im:
                try:
                    uploaded[name] = upload_image(site, im, f"{art['slug']}-{name}")
                    state.setdefault("images", []).append(im["id"])
                except Exception as e:
                    print(f"  {name} image upload failed:", str(e)[:200])
    body = art["content_html"]
    if inline:
        u = uploaded.get("inline")
        if u or dry_run:
            body = insert_figure(body, figure_html(u["url"] if u else inline["url"], inline,
                                                   u["w"] if u else None, u["h"] if u else None))
    content = body + "\n" + schema_block(site, art, site["url"])
    OUT_DIR.mkdir(exist_ok=True)
    out_file = OUT_DIR / f"{today():%Y-%m-%d}-{key}-{art['slug']}.html"
    out_file.write_text(f"<!-- {art['title']} | {art['meta_description']} | tags: {', '.join(art['tags'])} | "
                        f"featured image: {featured['url'] if featured else 'none'} -->\n{content}")

    if dry_run:
        print(f"DRY RUN saved {out_file}")
        return

    status = "draft" if site.get("ymyl") and site["status"] != "publish" else site["status"]
    if not art.get("_grounded"):
        status = "draft"  # no live search this run -> a human must fact-check
        print("WARNING: no search grounding this run; saving as DRAFT only")
    payload = {"title": art["title"], "slug": art["slug"], "content": content, "excerpt": art["excerpt"],
               "status": status}
    cid = category_id(site, cat_name)
    if cid:
        payload["categories"] = [cid]
    if "featured" in uploaded:
        payload["featured_media"] = uploaded["featured"]["id"]
    tids = tag_ids(site, art["tags"] or [topic["primary_keyword"]])
    if tids:
        payload["tags"] = tids
    res = wp(site, "POST", "posts", json=payload)
    print(f"{status.upper()} -> {res.get('link')}")
    state["used"].append({"date": today().isoformat(), "question": topic["question"],
                          "title": art["title"], "url": res.get("link")})
    save_state(key, state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--site")
    args = ap.parse_args()
    cfg = json.loads((ROOT / "sites.json").read_text())
    failed, ran = [], False
    for site in cfg["sites"]:
        if args.site and site["key"] != args.site:
            continue
        if not os.getenv(site["wp_user_env"]) or not os.getenv(site["wp_pass_env"]):
            print(f"\n=== {site['key']} === SKIPPED: secrets {site['wp_user_env']} / {site['wp_pass_env']} are empty")
            continue
        try:
            if ran:
                time.sleep(SITE_DELAY)
            ran = True
            run_site(site, args.dry_run)
        except Exception:
            traceback.print_exc()
            failed.append(site["key"])
    if failed:
        print("FAILED:", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
