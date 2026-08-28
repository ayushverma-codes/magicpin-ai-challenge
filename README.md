# Vera — magicpin AI Challenge Submission

## About the challenge

[magicpin's AI Challenge](https://magicpin.com/vera/ai-challenge) asks candidates to build the **message engine behind Vera**,
magicpin's AI assistant for local merchants. Vera helps merchants improve listings, run campaigns,
and reply to customers faster. The deliverable is a deterministic `compose(category, merchant,
trigger, customer?)` function, exposed over HTTP, that decides the next WhatsApp message Vera
should send — its CTA, its send-as identity, a suppression key, and the reasoning behind it.

Submissions are scored 0–10 on five dimensions (**decision quality, specificity, category fit,
merchant fit, engagement compulsion**) by an LLM judge that first replays the same 30 canonical
(merchant, trigger) pairs everyone starts from, then — after submission — injects **fresh** digest
items, metric shifts, triggers, and customer scopes the bot has never seen, specifically to test
generalization rather than memorization of the 30 pairs. Full detail in `challenge-brief.md` and
`challenge-testing-brief.md`.

## About this submission

This repo is my submission: the bot server (`bot.py`), the challenge's starter dataset + generator,
the official local judge simulator, my own local test runs against it, and supporting docs.

---

## What's in this repo

```
bot.py                       # FastAPI server — the actual submission (5 endpoints, see below)
requirements.txt              # bot dependencies

judge_simulator.py             # official local test harness (provided) — runs the 30 canonical
                                # test pairs against a running bot and scores it on 5 rubric dims
test_groq.py, test_async_groq.py, test_gemini.py
                                # tiny standalone scripts to sanity-check LLM provider credentials
                                # before running the bot or the simulator

dataset/
  categories/                  # 5 category contexts (voice, vocab, peer stats, digest items)
  merchants_seed.json          # 10 seed merchants (2 per category)
  customers_seed.json          # 15 seed customers
  triggers_seed.json           # 25 seed triggers
  generate_dataset.py           # deterministic expansion of the seeds (fixed seed, same output
                                # for every candidate)

expanded/                      # generated output of generate_dataset.py (committed for convenience)
  categories/                  # 5 files
  merchants/                   # 50 files
  customers/                   # 200 files
  triggers/                    # 100 files
  test_pairs.json               # 30 canonical (merchant, trigger) pairs used for local scoring

examples/
  api-call-examples.md          # sample request/response payloads for all 5 endpoints
  case-studies.md               # 10 judge-scored reference messages, with rationale

challenge-brief.md              # the challenge's product brief (what Vera is, the rubric)
challenge-testing-brief.md      # the technical contract: endpoints, harness behavior, constraints
engagement-design.md            # notes on message levers / engagement design
engagement-research.md          # supporting research notes
```

---

## The task, in one paragraph

Vera watches a merchant's category, business state, and live triggers (a performance dip, a
renewal coming due, a competitor opening nearby, a research digest, a customer recall) and decides
when to proactively message the merchant (or, on the merchant's behalf, a customer) — and how to
keep the conversation moving once they reply. A bot is judged on **decision quality**
(picking the right signal at the right time), **specificity** (real numbers/offers/dates, no
filler), **category fit** (voice, register, taboo words), **merchant fit** (their real
name/numbers/language), and **engagement compulsion** (one clear, low-friction next action). Full
rubric: `challenge-brief.md`.

---

## How the bot is built

`bot.py` is a single FastAPI service exposing the 5 endpoints the judge harness calls
(`challenge-testing-brief.md` §2):

| Endpoint | Purpose |
|---|---|
| `POST /v1/context` | Idempotent context push (`scope` + `context_id` + `version`). Stores category/merchant/customer/trigger payloads in memory, keyed by `(scope, context_id)`. Equal-version re-posts are a no-op success; a strictly lower version is rejected as `stale_version`. |
| `POST /v1/tick` | Called every simulated tick with a list of `available_triggers`. Composes an action (or none) for each trigger, concurrently, and returns them. |
| `POST /v1/reply` | Continues an existing conversation. Detects opt-out/hostility keywords, WhatsApp auto-reply loops, and explicit merchant commitment ("yes", "go ahead", "haan") to decide `send` / `wait` / `end`. |
| `GET /v1/healthz` | Uptime + count of loaded contexts per scope. |
| `GET /v1/metadata` | Team/model/approach metadata returned to the judge. |

### Message composition pipeline (the core of `/v1/tick`)

For each trigger, `compose_action()`:

1. Resolves the trigger → merchant → category chain (and customer, if the trigger is
   customer-scoped) from whatever context has been pushed so far. Missing context → skip that
   trigger rather than fabricate one.
2. Builds a per-request prompt (`build_user_prompt`) with four labeled blocks — CATEGORY,
   MERCHANT, TRIGGER, CUSTOMER — plus resolved digest items, active offers, recent conversation
   history, and a language instruction (English / Hindi / natural hi-en code-mix / light-Hindi,
   picked from the merchant's languages **and** the category's own code-mix voice setting, not one
   or the other alone).
3. Sends that prompt to an LLM (`COMPOSER_SYSTEM_PROMPT`) that is scored against the same 5
   dimensions the judge uses, with hard rules: never invent a fact/number/offer, exactly one CTA
   naming a concrete next action, never repeat prior history, never narrate its own reasoning
   ("based on the payload…").
4. Runs the output through `looks_low_quality()` — a cheap, dataset-agnostic gate (empty/too
   short/too long, a taboo word leaked, internal jargon leaked) — and retries once on a different
   pooled model if flagged.
5. Falls back to a fully data-driven **rule-based composer** (`build_fallback_action`, no LLM
   required) if every LLM attempt fails or the tick's time budget runs out — every fact it states
   is read straight from the same context objects, so the bot degrades gracefully instead of going
   silent or generic.

### Why multiple Groq models instead of one

`openai/gpt-oss-20b` alone has an 8,000 TPM budget, and a single tick can carry up to 20 triggers
composed concurrently — that's easily 3–4x the available budget in one window. Since Groq's free
TPM budgets are tracked **per model**, `bot.py` pools several models it has verified access to and
routes each call through `TokenBudgetLimiter`, a rolling 60s token ledger that picks whichever
pooled model currently has headroom for the estimated request cost (mirroring how Groq's own
admission check reserves `prompt_tokens + max_tokens` up front). This multiplies real throughput
instead of just capping in-flight request count. See the comment block at the top of `bot.py` for
the full reasoning and the caveat that the model pool/budgets should be re-verified against your
own Groq console.

### Other notable design choices

- **Idempotent context store** keyed by `(scope, context_id, version)` — matches the harness's
  documented replay/idempotency behavior exactly (a same-version re-post is a no-op success, not a
  conflict).
- **Customer-scoped triggers with no `CustomerContext` pushed** are deliberately *not* phrased as a
  receptionist's task ticket ("a patient's cleaning is due 12 Nov, slots: …") — the prompt
  explicitly steers toward a business-owner framing ("your recall list has a patient's cleaning
  window just opened — want me to message them?"), since that was the single most common quality
  failure mode found while iterating.
- **Mojibake sanitization** for ₹ and smart quotes, since some smaller hosted models mis-encode
  them.
- **`/v1/reply` never leaves the merchant hanging**: opt-out/hostility → `end`; three identical
  repeated messages → treated as a WhatsApp auto-reply loop and ended; explicit commitment → skips
  straight to action mode instead of re-qualifying.

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repo root:

```bash
GROQ_API_KEY1=your_groq_key_here
GROQ_MODEL=openai/gpt-oss-20b        # default model if GROQ_MODEL_POOL isn't set
GROQ_MODEL_POOL=                     # optional: comma-separated list to override the default pool
```

Run the bot:

```bash
uvicorn bot:app --host 0.0.0.0 --port 8080
```

Generate (or regenerate) the expanded dataset from the seeds — deterministic, so every run
produces the same 50 merchants / 200 customers / 100 triggers / 30 canonical test pairs:

```bash
python3 dataset/generate_dataset.py --seed-dir dataset --out expanded
```

---

## Testing locally

The official `judge_simulator.py` (provided as part of the challenge pack) runs a deterministic
dry-run against a live bot: health/metadata checks, base context load, then the 30 canonical test
pairs, scoring each of the 5 rubric dimensions.

1. Start the bot (`uvicorn bot:app ...`, above).
2. Open `judge_simulator.py` and set `BOT_URL`, `LLM_PROVIDER`, and `LLM_API_KEY` (the simulator
   uses its own LLM call to judge the bot's output — this is separate from the bot's own Groq
   key).
3. Run it:

```bash
python judge_simulator.py
```

Output is deterministic for the same input and simulator settings. Note that this local run scores
against the 30 canonical pairs only — the real judge harness injects **fresh** digest items, metric
shifts, triggers, and customer scopes after submission specifically to test generalization, not
memorization of the 30 pairs. See `examples/api-call-examples.md` for raw request/response payloads
per endpoint, and `examples/case-studies.md` for 10 judge-scored example messages with rationale.

---

## Local test results

Two runs against the local bot, judged by Gemini 3.5 Flash Lite (`LLM_PROVIDER = "gemini"` in
`judge_simulator.py` — a separate judge LLM from the Groq models the bot itself composes with).

### `PHASE2_SHORT` — 3 messages (dentist merchant, all triggers)

| Message (trigger) | Specificity | Category Fit | Merchant Fit | Decision Quality | Engagement | Total |
|---|---|---|---|---|---|---|
| Research digest (fluoride varnish study) | 9 | 9 | 9 | 8 | 8 | **43/50** |
| Regulatory alert (DCI radiograph dose limit) | 9 | 9 | 8 | 8 | 8 | **42/50** |
| Recall due (6-month cleaning) | 8 | 9 | 7 | 8 | 8 | **40/50** |

**Average: 41/50 (82%) — rated EXCELLENT**

### `FULL_EVALUATION` — 25 messages across 5 merchants (1 per category) × 5 triggers

| Avg dimension | Score |
|---|---|
| Specificity | 8/10 |
| Category Fit | 7/10 |
| Merchant Fit | 8/10 |
| Decision Quality | 7/10 |
| Engagement | 7/10 |

**Average: 37/50 (74%) — rated GOOD**

Full per-message range: 29/50 to 45/50. Highest: a pharmacy recall-list message for Ramesh
(45/50 — 9/9/10/9/8). Lowest: two gym messages for Padma (29/50 and 30/50) and a restaurant
IPL-match trigger for Suresh (34/50).

**What the spread shows:**

- **Dentists and pharmacies scored strongest and most consistently** (40–45/50) — these
  categories' voice/vocab constraints in `dataset/categories/` are tight and clinical, which plays
  to the composer's "anchor on a concrete fact" instruction.
- **Gyms and salons were the weakest and most variable category** (29–41/50). The lowest two
  scores (Padma, 29 and 30/50) were both customer-scoped triggers with no `CustomerContext` pushed
  — exactly the "receptionist task-ticket" failure mode `build_user_prompt`'s `orphan_customer_scope`
  branch is meant to prevent, and it clearly isn't fully solved: Category Fit and Merchant Fit
  dropped hardest there (4–7/10), suggesting the gym category's voice constraints aren't being
  honored as reliably as the health categories' are.
- **The same (merchant, trigger) pair scored differently across the two runs** — e.g. the
  recall_due message for Dr. Meera's patient was 40/50 in `PHASE2_SHORT` but 35/50 in
  `FULL_EVALUATION` (Decision Quality dropped from 8 to 4/10 for what reads as a similar message).
  Some of this is judge-side variance (Gemini re-composing its own critique each run), but some is
  the bot itself: the Groq model pool (`GROQ_MODEL_POOL`) means two runs of the *same* trigger can
  legitimately be composed by two different pooled models, and `looks_low_quality()`'s retry-on-flag
  logic only catches unambiguous breakage, not this kind of subjective quality drift between models.
- **Decision Quality (7/10 avg) and Engagement (7/10 avg) were the softest dimensions overall**,
  both dragged down by the same handful of gym/salon/restaurant messages above rather than a
  uniform weakness — the dentist and pharmacy messages hit 8–9/10 on both.

**Caveat:** this is a local, self-judged dry run (Gemini standing in for the real harness, and only
customer-facing/merchant-facing single-turn compositions — no live `/v1/reply` conversation was
scored here). It's a useful signal for where the composer is strong vs. shaky, not the actual
submission score, since the real harness uses different (fresh, unseen) scenarios and its own
scoring logic per `judge_simulator.py`.

---

## Reference docs

- `challenge-brief.md` — product brief: what Vera is, what a strong message looks like, the
  5-dimension scoring rubric.
- `challenge-testing-brief.md` — the technical contract: endpoint schemas, harness flow (warmup →
  test window → adaptive injection → replay test → score report), timeouts/rate/payload/tick-cap
  constraints.
- `engagement-design.md` / `engagement-research.md` — working notes on which engagement levers
  (loss aversion, social proof, curiosity, reciprocity, binary asks) were used where and why.