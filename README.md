# Auto-blog: 1 post per day on 4 WordPress sites

Runs free on GitHub Actions. Each day, for every site, it finds a real question
(Reddit / People-Also-Ask style), researches it, writes an answer-first article with
3-4 dofollow source links + internal links + FAQ + schema, validates it, and posts it.

## One-time setup (about 20 minutes)

1. **WordPress (do on all 4 sites):** Users -> Profile -> Application Passwords ->
   name it "autoblog" -> copy the password. The user must be Editor or Admin.
   (Needs HTTPS. If you get 401/403 later, a security plugin or Cloudflare rule is blocking the REST API.)
2. **Free Gemini API key:** go to aistudio.google.com -> Get API key. No billing needed. Open the key's project and check your limits in AI Studio (Google no longer publishes fixed numbers).
3. **GitHub:** create a PRIVATE repo and upload everything in this folder
   (keep the `.github/workflows/daily.yml` path exactly).
4. **Repo -> Settings -> Secrets and variables -> Actions -> New secret**, add 9 secrets:
   GEMINI_API_KEY
   WP_TECHYBIO_USER / WP_TECHYBIO_PASS
   WP_FLOWCHARTTECH_USER / WP_FLOWCHARTTECH_PASS
   WP_KNOWGROWHEALTH_USER / WP_KNOWGROWHEALTH_PASS
   WP_DIYCRAFTSNYOU_USER / WP_DIYCRAFTSNYOU_PASS
5. **Test:** Actions tab -> daily-posts -> Run workflow -> mode `dry-run`.
   Open the generated files in `out/` and read them.
6. **Go live as drafts:** run again with mode `live`. Check the 4 drafts in WordPress.
7. **Automatic:** the schedule (03:30 UTC daily) now runs on its own.

## Draft vs auto-publish
`sites.json` -> `"status": "draft"` (default) or `"publish"`.
Health site (`ymyl: true`) is forced to draft unless you set `"publish"` on purpose.
Recommended: 1-2 weeks of drafts, then switch the low-risk sites to `publish`.

## What it will NOT do yet
- No featured image (add via a plugin that auto-sets one, or tell me and I'll add Pexels/Unsplash).
- Meta description goes in the post *excerpt*. For Yoast/Rank Math fields, tell me which plugin each site uses.
- JSON-LD is inserted in the post body; it needs a user with `unfiltered_html` (Admin) or it gets stripped.
  If your SEO plugin already outputs Article schema, delete the `schema_block` line.

## Safety checks built in
Rejects a post unless: answer paragraph is 25-70 words and comes first, 3-4 distinct external dofollow links that
load, 2+ internal links from your real URLs, 4+ H2s, 3+ FAQs, length minimum, no AI cliches.
Health posts must also cite 2+ government/medical domains. Failed posts are retried 3 times with feedback,
then the run fails and GitHub emails you.

## Free-tier behaviour
- Uses `gemini-2.5-flash`, then `gemini-2.5-flash-lite` as backup (change with env `GEMINI_MODELS`).
- Tries Google Search grounding first. If a run has no grounding, the post is saved as a DRAFT only, never published.
- When grounded, every external link's domain must appear in the search results the model actually used.
- Each site needs about 2 calls (topic + article), more if a rewrite is needed. A 45-second pause between sites keeps you under per-minute limits.
- If the daily free quota runs out, that site fails and GitHub emails you; the next day's run continues normally.
- Free-tier content may be used by Google to improve its products. Fine for public blog posts.
GitHub pauses scheduled runs after 60 idle days; the daily state commit keeps the repo active.
