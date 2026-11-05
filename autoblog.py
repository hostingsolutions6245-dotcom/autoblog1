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

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))


# --------------------------------------------------------------------------- helpers
def today():
    return dt.date.today()


def host_of(url):
    return urlparse(url).netloc.lower().replace("www.", "")


def norm(url):
    return url.split("#")[0].rstrip("/").lower()


def _resolve(uri):
    try:
        return requests.get(uri, headers=UA, timeout=15, allow_redirects=True, stream=True).url
    except requests.RequestException:
        return uri


def ask(system, user, max_tokens=12000, searches=0):
    """Call Gemini (free tier). Tries Google Search grounding first, then falls back to no-search.
    Returns (text, grounded, domains_seen_in_search)."""
    last = None
    for model in MODELS:
        for grounded in (True, False):
            for attempt in range(3):
                try:
                    cfg = types.GenerateContentConfig(
                        system_instruction=system, max_output_tokens=max_tokens, temperature=0.6,
                        tools=[types.Tool(google_search=types.GoogleSearch())] if grounded else None)
                    r = client.models.generate_content(model=model, contents=user, config=cfg)
                    domains = set()
                    if grounded:
                        gm = getattr(r.candidates[0], "grounding_metadata", None)
                        for ch in (getattr(gm, "grounding_chunks", None) or []):
                            if getattr(ch, "web", None):
                                if ch.web.title:
                                    domains.add(ch.web.title.lower().replace("www.", ""))
                                if ch.web.uri:
                                    domains.add(host_of(_resolve(ch.web.uri)))
                    return r.text or "", bool(grounded and domains), domains
                except Exception as e:  # rate limit, quota, unsupported tool on this model...
                    last = e
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        time.sleep(25 * (attempt + 1))
                        continue
                    break
    raise RuntimeError(f"Gemini unavailable (free quota may be used up): {last}")


def extract_json(text):
    m = re.search(r"<json>(.*?)</json>", text, re.S)
    raw = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    return json.loads(raw)


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
def pick_topic(site, niche, titles, state):
    used = [u["question"] for u in state["used"]][-150:]
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
    text, _, _ = ask(system, user, max_tokens=3000)
    return extract_json(text)


def write_article(site, topic, link_pool, feedback=None):
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
    fb = f"\n\nYOUR PREVIOUS DRAFT FAILED THESE CHECKS - FIX ALL OF THEM:\n- " + "\n- ".join(feedback) if feedback else ""
    user = f"""Today is {today():%B %d, %Y}. Write one article for {site['url']}.
Question to answer: {topic['question']}
Primary keyword: {topic['primary_keyword']}  |  Intent: {topic['search_intent']}  |  Template: {topic['template']}
Audience: {site['audience']}

Research first: search for current (2025-2026) data and authoritative sources.

HARD RULES for content_html (HTML only, no <html>/<body>, no <h1>):
1. The FIRST element must be a <p> of 40-60 words that directly answers the question (this is the AEO answer).
2. Then a Key Takeaways box: <div class="key-takeaways"><strong>Key takeaways</strong><ul>3-5 <li></ul></div>
3. 4-6 <h2> sections (question-style where natural), <h3> only inside them. Short paragraphs, plain language,
   at least one list or table where it helps. Minimum {site['min_words']} words.
4. EXACTLY 3 or 4 external links to DIFFERENT authoritative domains you actually read. Plain <a href="..."> with
   target="_blank" rel="noopener" (dofollow: never nofollow/sponsored/ugc). Name the source in the sentence.
   Put them inside the text where the claim is made, and list them again under a final <h2>Sources</h2>.
5. 3 or 4 internal links, chosen ONLY from this list, URL copied exactly, with descriptive anchor text:
{pool}
6. An <h2>Frequently Asked Questions</h2> with 3-5 related questions (<h3> question + <p> answer, 30-60 words each).
7. Add original value: a concrete example, a decision rule, or a common mistake - not just a summary of sources.
8. Do not use these phrases: {', '.join(BANNED)}.{ymyl_rules}

Also return: title (<=60 chars, includes keyword), slug, meta_description (<=155 chars), excerpt (1-2 sentences),
tags (3-5), faq (same Q&A as in the HTML, as list of objects with q and a), sources (name + url for each external link).{fb}

Reply ONLY with:
<json>{{"title":"","slug":"","meta_description":"","excerpt":"","tags":[],"content_html":"","faq":[{{"q":"","a":""}}],"sources":[{{"name":"","url":""}}]}}</json>"""
    text, grounded, domains = ask(system, user, max_tokens=14000)
    art = extract_json(text)
    art["_grounded"], art["_domains"] = grounded, domains
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
    if len({h for h, _ in ext.values()}) < len(ext):
        problems.append("external links must go to different domains")
    if len(internal) < 2:
        problems.append("need at least 2 internal links")
    for i in internal:
        if i not in internal_ok:
            problems.append(f"internal link not in allowed list: {i}")
    for _, href in ext.values():
        if not url_alive(href):
            problems.append(f"source URL does not load: {href}")
    if art.get("_grounded"):
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
    # prefer topically recent posts but always allow category pages
    topic = pick_topic(site, niche, titles, state)
    print(f"niche={niche} | question={topic['question']}")

    feedback, art = None, None
    for attempt in range(1, 4):
        art = write_article(site, topic, link_pool, feedback)
        problems = validate(site, art, internal_ok)
        if not problems:
            break
        print(f"attempt {attempt} failed: {problems}")
        feedback = problems
    else:
        raise RuntimeError(f"{key}: article failed validation after 3 attempts: {problems}")

    content = art["content_html"] + "\n" + schema_block(site, art, site["url"])
    OUT_DIR.mkdir(exist_ok=True)
    out_file = OUT_DIR / f"{today():%Y-%m-%d}-{key}-{art['slug']}.html"
    out_file.write_text(f"<!-- {art['title']} | {art['meta_description']} -->\n{content}")

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
