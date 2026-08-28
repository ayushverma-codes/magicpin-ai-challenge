import time
import json
import os
import re
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Response
from pydantic import BaseModel
from groq import AsyncGroq

app = FastAPI(title="Vera Bot - magicpin AI Challenge")
START_TIME = time.time()

# ==========================================
# GROQ SETUP
# ==========================================
# Load variables from .env
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY1", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

groq_client = AsyncGroq(
    api_key=GROQ_API_KEY if GROQ_API_KEY else "dummy_key",
    base_url="https://api.groq.com",  # SDK appends /openai/v1/chat/completions itself — do not include it here
)

# ------------------------------------------------------------------------------------------------
# WHY THIS EXISTS: openai/gpt-oss-20b alone gives a single 8000 TPM (tokens/minute) budget, and the
# Groq rate limiter reserves prompt_tokens + max_tokens up front (not actual usage) when deciding
# whether a request admits. A tick can carry up to 20 triggers (judge spec: "Tick cap — 20 actions
# per tick") composed concurrently, each reserving ~1300-1800 tokens (≈700-tok system prompt + ≈550-
# 800-tok dynamic context + max_tokens headroom) — that's 25k-35k tokens of demand landing inside a
# ~20-30s window against an 8000/60s budget. A plain concurrency cap (old: Semaphore(3)) only throttles
# *how many requests are in flight*, not *how many tokens they're worth*, so it still blows the TPM
# ceiling and most triggers silently fall back to the rule-based composer.
#
# FIX: Groq's free-tier TPM budgets are tracked per model, not pooled across models. Spreading calls
# across a small pool of models — picking whichever pooled model currently has headroom — multiplies
# usable throughput instead of stacking every request against one bucket.
#
# The pool below is deliberately restricted to what actually showed up on this account's Groq
# console (Settings -> Limits): llama-3.1-8b-instant / llama-3.3-70b-versatile are NOT listed there
# (the list runs groq/* -> meta-llama/* -> openai/* -> qwen/* alphabetically with no llama-3.x entry
# in between) and calling a model outside your account's permissions just trades a TPM 429 for a
# model-access 404. gpt-oss-safeguard-20b is deliberately excluded even though it's on the console —
# it's OpenAI's moderation/classifier model, not tuned for open-ended composition. groq/compound(-mini)
# have a much bigger 70K TPM budget but are agentic systems that can invoke built-in tools (web
# search, code exec) on their own, which risks unpredictable latency/output shape for a task that
# needs a clean, fast JSON reply — left out of the default pool, but addable via GROQ_MODEL_POOL if
# you've verified they behave for this prompt.
#
# Re-verify this table against your own console before relying on it — Groq's published limits (and
# which models your account can even see) can change.
# ------------------------------------------------------------------------------------------------
GROQ_MODEL_TPM_BUDGETS: Dict[str, int] = {
    "openai/gpt-oss-20b": 8000,
    "openai/gpt-oss-120b": 8000,
    "qwen/qwen3.6-27b": 8000,
    "qwen/qwen3.8-27b": 8000,
}
_pool_env = os.environ.get("GROQ_MODEL_POOL", "")
GROQ_MODEL_POOL: List[str] = (
    [m.strip() for m in _pool_env.split(",") if m.strip()]
    if _pool_env else [GROQ_MODEL] + [m for m in GROQ_MODEL_TPM_BUDGETS if m != GROQ_MODEL]
)
# Any pooled model missing a known budget gets a conservative default rather than being dropped.
GROQ_MODEL_TPM_BUDGETS = {m: GROQ_MODEL_TPM_BUDGETS.get(m, 6000) for m in GROQ_MODEL_POOL}


class TokenBudgetLimiter:
    """Rolling 60s token ledger per pooled model. acquire() picks whichever model currently has
    headroom for the estimated cost of a call (mirroring Groq's prompt+max_tokens admission check)
    and reserves it; only waits when every pooled model is saturated. This replaces a plain
    concurrency semaphore, which caps how many calls are in flight but not how many tokens they're
    worth against the per-minute ceiling."""

    def __init__(self, budgets: Dict[str, int], window_s: float = 60.0):
        self.budgets = budgets
        self.window_s = window_s
        self.ledger: Dict[str, List[List[float]]] = {m: [] for m in budgets}  # [timestamp, tokens]
        self.lock = asyncio.Lock()

    def _used(self, model: str, now: float) -> float:
        cutoff = now - self.window_s
        entries = self.ledger[model]
        while entries and entries[0][0] < cutoff:
            entries.pop(0)
        return sum(t for _, t in entries)

    async def acquire(self, estimated_tokens: int, exclude: Optional[set] = None) -> str:
        """Blocks until some pooled model has headroom for estimated_tokens, reserves it, and
        returns which model to use. `exclude` lets a caller force a different model than one it
        already tried (used for the quality-gate retry below) — if excluding would leave zero
        candidate models, the exclusion is dropped rather than deadlocking forever."""
        exclude = exclude or set()
        if exclude and not (set(self.budgets) - exclude):
            exclude = set()
        while True:
            async with self.lock:
                now = time.time()
                best_model, best_headroom, soonest_free = None, -1.0, None
                for model, budget in self.budgets.items():
                    if model in exclude:
                        continue
                    headroom = budget - self._used(model, now)
                    if headroom >= estimated_tokens and headroom > best_headroom:
                        best_model, best_headroom = model, headroom
                    entries = self.ledger[model]
                    if entries:
                        free_at = entries[0][0] + self.window_s - now
                        if soonest_free is None or free_at < soonest_free:
                            soonest_free = free_at
                if best_model:
                    self.ledger[best_model].append([now, float(estimated_tokens)])
                    return best_model
            await asyncio.sleep(max(0.5, min(soonest_free or 1.0, 5.0)))

    async def release(self, model: str, reserved_tokens: int, actual_tokens: Optional[int]):
        """Correct the ledger after a call finishes. On failure (actual_tokens=None) the
        reservation is dropped entirely — a failed call didn't spend real budget. On success we
        never reduce the reservation below what was estimated (Groq's admission check appears to
        count the declared max_tokens regardless of how much was actually generated), but we do
        raise it if actual usage exceeded the estimate."""
        async with self.lock:
            entries = self.ledger[model]
            for entry in reversed(entries):
                if entry[1] == float(reserved_tokens):
                    if actual_tokens is None:
                        entries.remove(entry)
                    elif actual_tokens > reserved_tokens:
                        entry[1] = float(actual_tokens)
                    return


GROQ_LIMITER = TokenBudgetLimiter(GROQ_MODEL_TPM_BUDGETS)

# Concurrency is now a safety cap on in-flight HTTP calls (not the TPM guard — GROQ_LIMITER is), so
# it can be sized to the model pool: a couple of concurrent calls per pooled model.
GROQ_CONCURRENCY = asyncio.Semaphore(max(3, 2 * len(GROQ_MODEL_POOL)))

_CHARS_PER_TOKEN = 4  # rough estimator; good enough for admission-check purposes, not billing


def estimate_tokens(*texts: str) -> int:
    return sum(len(t) for t in texts) // _CHARS_PER_TOKEN + 1


def completion_budget(model: str) -> int:
    # gpt-oss models burn tokens on hidden reasoning before the visible answer even at low effort,
    # so they need more max_tokens headroom than a model with reasoning turned fully off.
    if "gpt-oss" in model and "safeguard" not in model:
        return 400
    return 260


def reasoning_kwargs(model: str) -> Dict[str, Any]:
    """Per-model reasoning_effort support differs and passing an unsupported value 400s the call
    (confirmed against Groq's /docs/reasoning): gpt-oss-20b/120b only accept low/medium/high (no
    'none'); qwen3.6-27b only accepts none/default (no low/medium/high); qwen3.8-27b accepts all
    five. We always want reasoning OFF or minimal here — this task is a short WhatsApp message +
    JSON wrapper, not a task that benefits from a hidden chain-of-thought, and every reasoning
    token spent is TPM budget not spent on throughput."""
    if "gpt-oss" in model and "safeguard" not in model:
        return {"reasoning_effort": "low"}  # lowest value gpt-oss actually accepts
    if model.startswith("qwen/"):
        return {"reasoning_effort": "none"}  # both qwen3.6 and qwen3.8 accept "none"
    return {}

# ==========================================
# COMPOSER PROMPT
#
# NOTE: this prompt must never bake in facts about any *specific* merchant
# (no fixed view/call counts, no fixed offer names). All concrete numbers
# must come from the dynamic context block built per-request in
# `build_user_prompt`. Baking in one merchant's numbers (the original bug
# here) meant every category/merchant got Dr. Meera's dental stats, which
# tanks category fit, merchant fit, specificity and triggers fabrication
# penalties on every message that isn't actually about that one merchant.
# ==========================================
COMPOSER_SYSTEM_PROMPT = """You are Vera, magicpin's WhatsApp assistant for local merchants in India.
You get 4 context blocks (CATEGORY, MERCHANT, TRIGGER, CUSTOMER) and compose ONE WhatsApp message.
Scored on 5 dimensions — optimize for all:

1. SPECIFICITY: anchor on a concrete fact FROM THE CONTEXT (number/date/citation/peer stat). No
   "grow your business" filler. "<service> @ ₹<price>" beats "X% off". If the trigger itself is thin
   on merchant-specific numbers (source="external": a local event/match/festival/competitor move,
   or anything where the payload is just world-facts, not this merchant's own data) — the external
   fact alone is NOT enough for a high score. Pair it with ONE real fact pulled from MERCHANT
   CONTEXT (an actual offer price, a performance number, a signal, a menu item, their locality) so
   the message is grounded in something specific to THIS business, not generic event commentary any
   nearby merchant could receive unchanged.
2. CATEGORY FIT: match voice.tone/register, use allowed vocab naturally, NEVER a taboo word, only
   category-appropriate offers/services.
3. MERCHANT FIT: use their real name (owner first name + category salutation style, e.g. "Dr. X" for
   dentists), only their real numbers/signals/offers, honor language preference.
4. TRIGGER RELEVANCE / DECISION QUALITY: obviously exists because of THIS trigger's actual payload —
   not a generic nudge. If the payload includes a WHY (a driver/cause/theme/reason field explaining
   what caused the signal — not just the number itself), the recommended next action must connect to
   that cause specifically, not just restate the metric. E.g. a spike caused by a specific post/offer
   → suggest building on that same thing, not a generic "want help with a promotion?".
5. ENGAGEMENT: one lever (loss aversion, social proof, effort externalization, curiosity, reciprocity,
   a question, or binary Reply YES/STOP). Exactly ONE CTA, and it must name the CONCRETE next action —
   never vague ("take care of it", "help you out"). E.g. "reply YES and I'll send the checklist", not
   "reply YES if you'd like me to take care of it".

HARD RULES:
- Never invent a fact/number/offer/citation not in the context. This includes DERIVED numbers — don't
  multiply/estimate a headcount or total unless that exact figure is literally given.
- Exactly one CTA, last sentence, concrete (see #5).
- No preambles. Don't re-introduce yourself if conversation_history is non-empty. Never repeat a
  message already in conversation_history.
- Talk like a person, not a system narrating its own logic — avoid "I can pair it with X", "combining
  trigger Y with offer Z", "based on the payload". Say things naturally.
- If merchant languages include both hi+en (or category voice says code-mix is natural), ACTUALLY
  WRITE in Hindi-English mix — real Hindi words mixed in, not just English with code-mix "allowed" in
  name only. E.g. "Aapka CTR peer median se kam hai", not "Your CTR is below peer median". English-only
  merchants get English.
- CustomerContext present → send_as="merchant_on_behalf", written in the merchant's voice to their
  customer, never mentioning Vera, respecting category customer-facing taboos (no medical claims, no
  "guaranteed"). No CustomerContext → send_as="vera", merchant-facing.
- cta: "binary" (YES/STOP or 2-choice), "open_ended" (invites reply, no fixed options), or "none".
- 2-4 short lines, WhatsApp not email.
- Research/regulation/trend triggers: tie the insight to something actionable this week (an offer, a
  patient conversation, a protocol note), not just an abstract citation.

Output JSON ONLY, no markdown fences:
{"body": "...", "cta": "binary|open_ended|none", "send_as": "vera|merchant_on_behalf", "suppression_key": "...", "rationale": "..."}
"""

# ==========================================
# STATE MANAGEMENT
# ==========================================
contexts: Dict[tuple, Dict] = {}
conversations: Dict[str, List[Dict[str, str]]] = {}
conv_meta: Dict[str, Dict[str, Any]] = {}  # conversation_id -> {merchant_id, customer_id, trigger_id}


class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: List[str] = []


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


# ==========================================
# CONTEXT HELPERS
# ==========================================

def get_payload(scope: str, ctx_id: Optional[str]) -> Optional[Dict]:
    if not ctx_id:
        return None
    entry = contexts.get((scope, ctx_id))
    return entry["payload"] if entry else None


_MOJIBAKE_FIXES = {
    "â‚¹": "₹", "Ã¢â€šÂ¹": "₹", "â‚¬": "€", "â€™": "’", "â€œ": "\u201c", "â€\x9d": "\u201d",
    "\ufffd": "₹",  # unicode replacement char showing up where a rupee sign was expected
}


def sanitize_text(s: str) -> str:
    """Fix common UTF-8 mis-decoding (mojibake) that some small hosted models introduce for ₹ and quotes."""
    if not s:
        return s
    for bad, good in _MOJIBAKE_FIXES.items():
        s = s.replace(bad, good)
    return s


def active_offers(merchant: Dict) -> List[str]:
    return [o.get("title") for o in merchant.get("offers", []) if o.get("status") == "active" and o.get("title")]


def resolve_digest_item(category: Dict, item_id: Optional[str]) -> Optional[Dict]:
    if not item_id:
        return None
    for item in category.get("digest", []):
        if item.get("id") == item_id:
            return item
    return None


def salutation_name(category: Dict, merchant: Dict) -> str:
    """Pick a category-appropriate way to address the merchant, from real identity data only."""
    identity = merchant.get("identity", {})
    first = identity.get("owner_first_name")
    name = identity.get("name", "there")
    examples = category.get("voice", {}).get("salutation_examples", [])
    if first and any("Dr." in ex for ex in examples):
        return f"Dr. {first}"
    return first or name


def recent_history(merchant: Dict, n: int = 3) -> List[Dict]:
    hist = merchant.get("conversation_history", [])
    return hist[-n:] if hist else []


def language_hint(merchant: Dict, customer: Optional[Dict], category: Optional[Dict] = None) -> str:
    """Returns a short code the caller maps to full instructions — never free text with substrings
    like 'mix' baked in, since that used to get accidentally matched by callers doing `if "mix" in
    lang`. Values: 'hi', 'en', 'hi_en_natural', 'en_light_hindi', or a customer's raw language_pref
    string when customer-facing (that field is free text set per-customer, not a fixed vocabulary)."""
    if customer:
        return customer.get("identity", {}).get("language_pref", "en")

    langs = merchant.get("identity", {}).get("languages", [])
    has_hi = "hi" in langs
    has_en = "en" in langs
    if not has_en:
        return "hi" if has_hi else "en"
    if not has_hi:
        return "en"

    # Merchant can read both — now the DEGREE of code-mixing is the category's call, not a fact
    # derivable from which languages the merchant happens to speak. Conflating the two used to force
    # a full "natural hi-en code-mix" instruction onto every bilingual merchant regardless of what
    # their category's voice actually wants (e.g. gyms' voice.code_mix="english_primary_some_hindi"
    # explicitly wants MOSTLY English — forcing heavy code-mix there directly contradicted the
    # category spec and was tanking Category Fit for exactly that reason).
    code_mix = (category or {}).get("voice", {}).get("code_mix", "")
    if code_mix == "english_primary_some_hindi":
        return "en_light_hindi"
    # "hindi_english_natural" or any unrecognized/missing value on a bilingual merchant — natural
    # code-mix is the safer default since it's what most of this dataset's categories use, but this
    # only fires when the category doesn't explicitly say otherwise.
    return "hi_en_natural"


LANGUAGE_INSTRUCTIONS = {
    "hi_en_natural": (
        "hi-en mix (natural code-mix, not pure Hindi or pure English)\n"
        "This is a REQUIREMENT, not a suggestion: if it says hi-en mix, the body text itself must "
        "contain natural Hindi words/phrasing mixed with English (e.g. \"aapka\", \"hai\", \"kar "
        "sakti hoon\"), not just English with an occasional Hindi word."
    ),
    "en_light_hindi": (
        "English-primary, with only occasional light Hindi words for warmth (e.g. a single \"aapka\" "
        "or \"bhai\" here and there) — this category's voice wants MOSTLY English, not a full natural "
        "code-mix. Don't overcorrect into heavy Hindi phrasing."
    ),
    "hi": "hi",
    "en": "en",
}


def build_user_prompt(category: Dict, merchant: Dict, trigger: Dict, customer: Optional[Dict]) -> str:
    trigger_payload = trigger.get("payload", {})
    digest_item = None
    item_id = trigger_payload.get("top_item_id") or trigger_payload.get("digest_item_id")
    if item_id:
        digest_item = resolve_digest_item(category, item_id)

    # A trigger can be scope="customer" (about one of the merchant's customers) even when we were
    # never given a full CustomerContext for it. In that case this must NOT be phrased as if the
    # customer's due-date/event were the merchant's own task — it should stay merchant-facing but
    # clearly be *about* their customer, offering to handle the outreach on the merchant's behalf.
    orphan_customer_scope = trigger.get("scope") == "customer" and customer is None

    voice = category.get("voice", {})
    parts = []

    parts.append("=== CATEGORY CONTEXT ===")
    parts.append(f"slug: {category.get('slug')}")
    parts.append(f"voice: tone={voice.get('tone')}, register={voice.get('register')}, code_mix={voice.get('code_mix')}")
    parts.append(f"vocab_allowed: {voice.get('vocab_allowed', [])[:6]}")
    parts.append(f"vocab_taboo (NEVER use): {voice.get('vocab_taboo', [])}")
    parts.append(f"peer_stats: {category.get('peer_stats', {})}")
    if digest_item:
        parts.append(f"RESOLVED digest item referenced by this trigger: {json.dumps(digest_item, ensure_ascii=False)}")

    parts.append("\n=== MERCHANT CONTEXT (this specific business — do not use any other merchant's data) ===")
    identity = merchant.get("identity", {})
    parts.append(f"identity: {json.dumps(identity, ensure_ascii=False)}")
    parts.append(f"subscription: {merchant.get('subscription', {})}")
    parts.append(f"performance (their real numbers): {merchant.get('performance', {})}")
    parts.append(f"active offers: {active_offers(merchant)}")
    parts.append(f"signals: {merchant.get('signals', [])[:4]}")
    parts.append(f"customer_aggregate: {merchant.get('customer_aggregate', {})}")
    review_themes = merchant.get("review_themes", [])[:1]
    if review_themes:
        parts.append(f"review_themes: {json.dumps(review_themes, ensure_ascii=False)}")
    hist = recent_history(merchant, n=2)
    if hist:
        hist_trimmed = [{"from": h.get("from"), "body": (h.get("body") or "")[:140]} for h in hist]
        parts.append(f"last conversation turns (do not repeat, don't re-introduce yourself): {json.dumps(hist_trimmed, ensure_ascii=False)}")

    parts.append("\n=== TRIGGER CONTEXT (the reason for this message, right now) ===")
    parts.append(f"kind: {trigger.get('kind')}, scope: {trigger.get('scope')}, source: {trigger.get('source')}, urgency: {trigger.get('urgency')}/5")
    parts.append(f"payload: {json.dumps(trigger_payload, ensure_ascii=False)}")
    parts.append(f"suppression_key (reuse exactly): {trigger.get('suppression_key')}")

    if orphan_customer_scope:
        noun = {"dentists": "patient", "pharmacies": "patient", "gyms": "member",
                "salons": "client", "restaurants": "customer"}.get(category.get("slug"), "customer")
        parts.append(
            "\nIMPORTANT: trigger scope=\"customer\" — about ONE OF THE MERCHANT'S CUSTOMERS "
            f"(a {noun}, for this category), no CustomerContext pushed (no name — don't invent one). "
            "send_as=\"vera\", merchant-facing.\n"
            "This is the #1 way these messages go wrong, so read carefully: being specific (the exact "
            "date/window from the payload should still show up somewhere) is NOT the same as reading "
            "like a data row or an admin task ticket. The merchant is a business owner getting a "
            "platform insight/opportunity, not a receptionist getting a reminder to personally chase "
            "one customer's file. This applies to EVERY kind of customer-scoped trigger — a recall, a "
            "trial follow-up, a lapsed/win-back nudge, a refill reminder, a wedding/event follow-up, "
            "or anything else this format shows up as — the underlying pattern is always the same, "
            "shown here across two different categories so you generalize it rather than copy the "
            "wording:\n"
            "WRONG (reads like an internal task log handed to a receptionist — never write like this, "
            "in any category, for any kind of customer trigger):\n"
            "  \"Ek patient ka 6-month cleaning due hai 12 Nov ko, last visit 12 May tha. Available "
            "slots: Mon 3pm, Wed 11am, Fri 5pm.\"\n"
            "  \"Ek member ka trial session complete ho gaya 22 April ko, agla slot Sat 3pm available "
            "hai.\"\n"
            "RIGHT (same underlying facts, framed as a business opportunity to a peer, not a task "
            "list):\n"
            "  \"Dr. Meera, aapke recall list mein ek patient ka 6-month cleaning window abhi khula hai "
            "(last visit 12 May) — chahen to main unhe seedha message karke book kar doon?\"\n"
            "  \"Rohan, aapka ek trial member follow-up ke liye ready hai (trial 22 April ko tha) — "
            "chahen to main unhe agle available slot ka message bhej doon?\"\n"
            "Fold the service+timing fact into ONE natural sentence a real practitioner/owner would say "
            "about their own business, in the category's own voice (clinical urgency/continuity-of-care "
            "for health categories, warm relationship urgency for lifestyle ones), then offer to "
            "personally handle the outreach yourself — never list raw slots/options for the merchant to "
            "relay themselves, that's the receptionist-task-ticket failure mode above.\n"
            "If you reach for an active offer here, check it actually fits this specific customer's "
            "situation first — e.g. a new-customer trial/acquisition offer looks tone-deaf pitched at "
            "someone who was already a paying regular (a lapsed/win-back case), it reads as if you "
            "don't know your own customer. Skip the offer entirely rather than force a mismatched one."
        )

    if customer:
        parts.append("\n=== CUSTOMER CONTEXT (drafting on behalf of the merchant, to this customer) ===")
        parts.append(f"identity: {customer.get('identity', {})}")
        parts.append(f"relationship: {customer.get('relationship', {})}")
        parts.append(f"state: {customer.get('state')}")
        parts.append(f"preferences: {customer.get('preferences', {})}")
    else:
        parts.append("\n=== CUSTOMER CONTEXT ===\nNone — this is a merchant-facing message (send_as=vera).")

    lang = language_hint(merchant, customer, category)
    # Customer-facing: lang is the customer's raw free-text language_pref (e.g. "hi-en mix"), which
    # isn't a key in LANGUAGE_INSTRUCTIONS — use it as-is. Merchant-facing: lang is one of the fixed
    # short codes from language_hint's docstring, which IS a key — map it to full instructions.
    lang_instruction = LANGUAGE_INSTRUCTIONS.get(lang, lang)
    parts.append(f"\n=== LANGUAGE TO WRITE IN ===\n{lang_instruction}")
    parts.append(f"\n=== HOW TO ADDRESS THEM ===\n{salutation_name(category, merchant) if not customer else customer.get('identity', {}).get('name', 'there')}")

    return "\n".join(parts)


# ==========================================
# RULE-BASED FALLBACK (no LLM required)
# Used when the LLM call fails after retries. Still fully data-driven —
# every fact it states comes straight out of the actual context objects.
# ==========================================

def humanize_key(k: str) -> str:
    return k.replace("_", " ")


def sentence_case(s: str) -> str:
    """Capitalize only the first character; never lowercase the rest (preserves acronyms/citations)."""
    if not s:
        return s
    return s[0].upper() + s[1:]


def fallback_trigger_fact(trigger: Dict, category: Dict) -> str:
    kind = trigger.get("kind", "")
    payload = trigger.get("payload", {})

    item_id = payload.get("top_item_id") or payload.get("digest_item_id")
    if item_id:
        item = resolve_digest_item(category, item_id)
        if item:
            src = f" ({item.get('source')})" if item.get("source") else ""
            n = f" — {item.get('trial_n')} patients/cases studied" if item.get("trial_n") else ""
            return f"{item.get('title')}{n}{src}"

    if kind in ("perf_dip", "seasonal_perf_dip"):
        metric = payload.get("metric", "performance")
        delta = payload.get("delta_pct")
        window = payload.get("window", "recently")
        seasonal_note = ", which tracks with the seasonal pattern" if payload.get("is_expected_seasonal") else ""
        if delta is not None:
            return f"your {metric} dropped {abs(delta) * 100:.0f}% over the last {window}{seasonal_note}"
    if kind == "perf_spike":
        metric = payload.get("metric", "performance")
        delta = payload.get("delta_pct")
        driver = payload.get("likely_driver")
        driver_txt = f", likely from {humanize_key(driver)}" if driver else ""
        if delta is not None:
            return f"your {metric} jumped {delta * 100:.0f}% recently{driver_txt}"
    if kind == "renewal_due":
        days = payload.get("days_remaining", payload.get("days"))
        plan = payload.get("plan")
        amount = payload.get("renewal_amount")
        plan_txt = f" {plan}" if plan else ""
        amount_txt = f" (₹{amount})" if amount else ""
        return f"your{plan_txt} subscription renews{amount_txt}{f' in {days} days' if days else ''}"
    if kind == "recall_due":
        service = payload.get("service_due", "a follow-up")
        due = payload.get("due_date", "")
        return f"{humanize_key(service)} is due{f' ({due})' if due else ''}"
    if kind == "milestone_reached":
        metric = humanize_key(payload.get("metric", "a milestone"))
        value_now = payload.get("value_now")
        milestone_value = payload.get("milestone_value")
        if value_now is not None and milestone_value is not None:
            gap = milestone_value - value_now
            if gap > 0:
                return f"you're at {value_now} {metric}, just {gap} short of {milestone_value}"
            return f"you just crossed {milestone_value} {metric}"
        return "you're closing in on a new milestone"
    if kind == "review_theme_emerged":
        theme = payload.get("theme")
        occ = payload.get("occurrences_30d")
        trend = payload.get("trend")
        trend_txt = f", and it's {trend}" if trend else ""
        return f"{occ} recent reviews mention \"{humanize_key(theme)}\"{trend_txt}" if theme else "a new review theme is emerging"
    if kind == "competitor_opened":
        name = payload.get("competitor_name")
        dist = payload.get("distance_km")
        offer = payload.get("their_offer")
        offer_txt = f", running {offer}" if offer else ""
        who = f"{name} " if name else "a new competitor "
        return f"{who}opened {dist}km away{offer_txt}" if dist else f"{who}opened nearby{offer_txt}"
    if kind == "winback_eligible":
        days = payload.get("days_since_expiry")
        lapsed = payload.get("lapsed_customers_added_since_expiry")
        bits = []
        if days is not None:
            bits.append(f"{days} days since your last offer expired")
        if lapsed is not None:
            bits.append(f"{lapsed} customers have lapsed since")
        return ", ".join(bits) if bits else "a batch of customers are win-back eligible"
    if kind == "dormant_with_vera":
        days = payload.get("days_since_last_merchant_message", payload.get("days_inactive"))
        topic = payload.get("last_topic")
        topic_txt = f" — we were last talking about {humanize_key(topic)}" if topic else ""
        return f"it's been {days} days since we last spoke{topic_txt}" if days else "it's been a while since we last spoke"
    if kind == "festival_upcoming":
        festival = payload.get("festival", "a festival")
        date = payload.get("date")
        days_until = payload.get("days_until")
        when = f" on {date}" if date else ""
        lead = f" ({days_until} days out)" if days_until else ""
        return f"{festival} is coming up{when}{lead}"
    if kind == "curious_ask_due":
        template = payload.get("ask_template", "")
        if "demand" in template:
            return "curious what's been in most demand at your place this week"
        return "worth a quick check-in on how things are going"
    if kind == "active_planning_intent":
        topic = humanize_key(payload.get("intent_topic", "that idea"))
        return f"you mentioned wanting to explore {topic} recently"
    if kind == "supply_alert":
        molecule = payload.get("molecule")
        batches = payload.get("affected_batches", [])
        manufacturer = payload.get("manufacturer")
        batch_txt = f" (batches {', '.join(batches)})" if batches else ""
        mfr_txt = f" from {manufacturer}" if manufacturer else ""
        return f"{molecule}{mfr_txt} is under a recall alert{batch_txt}"
    if kind == "category_seasonal":
        season = humanize_key(payload.get("season", "this season"))
        trends = payload.get("trends", [])
        trend_txt = "; ".join(t.replace("_", " ") for t in trends[:3])
        return f"{season} demand is shifting: {trend_txt}" if trend_txt else f"{season} demand is shifting"
    if kind == "gbp_unverified":
        uplift = payload.get("estimated_uplift_pct")
        uplift_txt = f" — verifying could lift visibility ~{uplift * 100:.0f}%" if uplift else ""
        return f"your Google Business Profile isn't verified yet{uplift_txt}"
    if kind == "wedding_package_followup":
        days = payload.get("days_to_wedding")
        next_step = payload.get("next_step_window_open")
        step_txt = f" — {humanize_key(next_step)} is a natural next step" if next_step else ""
        return f"a bride's wedding is {days} days out and she's just finished her trial{step_txt}" if days else "a bride just finished her trial and is worth a timely follow-up"
    if kind == "trial_followup":
        options = payload.get("next_session_options", [])
        slot_txt = f", with {options[0].get('label')} open" if options else ""
        return f"someone completed a trial session and there's a natural next slot{slot_txt}"
    if kind == "customer_lapsed_hard":
        days = payload.get("days_since_last_visit")
        focus = payload.get("previous_focus")
        focus_txt = f" (their focus was {humanize_key(focus)})" if focus else ""
        return f"a member hasn't visited in {days} days{focus_txt}" if days else "a member has gone quiet for a while"
    if kind == "chronic_refill_due":
        molecules = payload.get("molecule_list", [])
        runs_out = payload.get("stock_runs_out_iso", "")[:10]
        med_txt = ", ".join(molecules[:3]) if molecules else "their regular medication"
        return f"a regular patient's {med_txt} runs out around {runs_out}" if runs_out else f"a regular patient is due a refill of {med_txt}"
    if kind == "ipl_match_today":
        match = payload.get("match")
        venue = payload.get("venue")
        return f"{match} is on tonight at {venue} — good night for walk-in traffic" if match else "there's a big match on tonight — good night for walk-in traffic"

    # Generic: surface the 1-2 most informative payload fields, in natural prose (not a field dump)
    bits = []
    for k, v in payload.items():
        if isinstance(v, (str, int, float)) and k not in ("category",):
            bits.append(f"{humanize_key(str(v)) if isinstance(v, str) else v}")
        if len(bits) >= 2:
            break
    kind_txt = humanize_key(kind) or "something worth flagging"
    if bits:
        return f"{kind_txt} — {', '.join(str(b) for b in bits)}"
    return kind_txt
    bits = []
    for k, v in list(payload.items())[:2]:
        if isinstance(v, (str, int, float)):
            bits.append(f"{humanize_key(k)}: {v}")
    kind_txt = humanize_key(kind) or "an update on your account"
    return f"there's an update on {kind_txt}" + (f" ({', '.join(bits)})" if bits else "")


def fallback_customer_scope_phrase(trigger: Dict, noun: str) -> str:
    """Natural-language phrasing for a customer-scoped trigger with no CustomerContext available.
    Avoids the 'field: value' data-dump feel of fallback_trigger_fact."""
    kind = trigger.get("kind", "")
    payload = trigger.get("payload", {})

    if kind == "recall_due":
        service = humanize_key(payload.get("service_due", "a follow-up"))
        due = payload.get("due_date")
        when = f" around {due}" if due else " soon"
        return f"one of your {noun}s is coming up for their {service}{when}"
    if kind == "customer_lapsed_hard":
        days = payload.get("days_since_last_visit")
        focus = payload.get("previous_focus")
        focus_txt = f" (they were focused on {humanize_key(focus)})" if focus else ""
        when = f" in {days} days" if days else " in a while"
        return f"one of your {noun}s hasn't come by{when}{focus_txt}"
    if kind in ("winback_eligible", "customer_lapsed_soft"):
        return f"a few of your {noun}s have gone quiet and are worth a win-back nudge"
    if kind == "wedding_package_followup":
        days = payload.get("days_to_wedding")
        when = f", {days} days out from the wedding," if days else ""
        return f"a bride{when} just finished her trial — a natural moment to follow up before she books elsewhere"
    if kind == "trial_followup":
        return f"one of your {noun}s just finished a trial session — worth catching them for the next one before they book elsewhere"
    if kind == "chronic_refill_due":
        molecules = payload.get("molecule_list", [])
        med_txt = f" of {', '.join(molecules[:2])}" if molecules else ""
        return f"one of your regular {noun}s is about to run out{med_txt}"
    # Generic natural fallback
    return f"one of your {noun}s has a timely reason to hear from you right now"


def build_fallback_action(category: Dict, merchant: Dict, trigger: Dict, customer: Optional[Dict],
                           merchant_id: str, customer_id: Optional[str], trigger_id: str) -> Dict:
    fact = fallback_trigger_fact(trigger, category)
    offers = active_offers(merchant)
    urgency = trigger.get("urgency", 1)

    if customer:
        cust_name = customer.get("identity", {}).get("name", "there")
        merch_name = merchant.get("identity", {}).get("name", "")
        offer_line = f" {offers[0]}." if offers else ""
        body = f"Hi {cust_name}, {merch_name} here. {sentence_case(fact)}.{offer_line} Reply and let us know what works for you."
        send_as = "merchant_on_behalf"
        cta = "open_ended"
    elif trigger.get("scope") == "customer":
        # We know this is about a customer, but have no CustomerContext (no name/preferences) —
        # keep it merchant-facing, framed as a retention opportunity, not a raw payload dump.
        who = salutation_name(category, merchant)
        noun = {"dentists": "patient", "pharmacies": "patient", "gyms": "member",
                "salons": "client", "restaurants": "customer"}.get(category.get("slug"), "customer")
        phrase = fallback_customer_scope_phrase(trigger, noun)
        # Lapsed/win-back trigger kinds are specifically about a customer who ALREADY has a history
        # with this merchant. Whatever offer happens to be active right now is very often a
        # new-customer acquisition offer, and auto-attaching it here reads as if the merchant doesn't
        # recognize their own returning customer — skip the offer line for these kinds rather than
        # risk a mismatched pitch (this template can't judge offer relevance the way the LLM path can).
        suppress_offer = trigger.get("kind", "") in (
            "customer_lapsed_hard", "customer_lapsed_soft", "winback_eligible"
        )
        offer_line = f" Might be worth mentioning {offers[0]}." if (offers and not suppress_offer) else ""
        body = f"{who}, {phrase}.{offer_line} Want me to draft the message?"
        send_as = "vera"
        cta = "binary"
    else:
        who = salutation_name(category, merchant)
        offer_line = f" You currently have {offers[0]} live." if offers else ""
        binary = urgency >= 3
        ask = "Reply YES and I'll draft it for your review." if binary else "Want me to pull the details?"
        body = f"{who}, {sentence_case(fact)}.{offer_line} {ask}"
        send_as = "vera"
        cta = "binary" if binary else "open_ended"

    return {
        "conversation_id": f"conv_{merchant_id}_{trigger_id}",
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "send_as": send_as,
        "trigger_id": trigger_id,
        "body": body,
        "cta": cta,
        "suppression_key": trigger.get("suppression_key", trigger_id),
        "rationale": f"Fallback (LLM unavailable): templated from trigger kind '{trigger.get('kind')}' using real merchant/category data.",
    }


# ==========================================
# ENDPOINTS
# ==========================================
@app.get("/healthz")
@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _), _ in contexts.items():
        counts[scope] = counts.get(scope, 0) + 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START_TIME), "contexts_loaded": counts}


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Team Vera Pro",
        "team_members": ["Your Name"],
        "model": "+".join(GROQ_MODEL_POOL),
        "approach": "Dynamic per-request context synthesis (category/merchant/trigger/customer) with an LLM composer and a data-driven rule-based fallback",
        "contact_email": "team@example.com",
        "version": "2.0.0",
        "submitted_at": datetime.utcnow().isoformat() + "Z"
    }


VALID_SCOPES = {"category", "merchant", "customer", "trigger"}


@app.post("/v1/context")
async def push_context(body: CtxBody, response: Response):
    # Per the testing brief: "Idempotent by (context_id, version). Re-posting the same version is
    # a no-op" — a no-op must still return accepted:true (the desired state IS stored), it just
    # shouldn't rewrite the payload again. The old `>=` comparison here treated an equal-version
    # re-post the same as a stale/older one and returned accepted:false, which is what tripped every
    # CONTEXT PUSH [FAIL] in phase2_short: judge_simulator.py always posts version=1 (it never
    # increments), so any run against a server that already holds version=1 from an earlier run
    # (warm state, server not restarted between runs) got incorrectly rejected as "stale" even
    # though it was posting the exact same context again. Only a version STRICTLY LOWER than what's
    # stored is a genuine conflict.
    if body.scope not in VALID_SCOPES:
        response.status_code = 400
        return {"accepted": False, "reason": "invalid_scope",
                "details": f"scope must be one of {sorted(VALID_SCOPES)}, got '{body.scope}'"}

    key = (body.scope, body.context_id)
    cur = contexts.get(key)

    if cur and body.version == cur["version"]:
        # Idempotent no-op: already have exactly this version, nothing to change, still a success.
        return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}", "stored_at": datetime.utcnow().isoformat() + "Z"}

    if cur and cur["version"] > body.version:
        response.status_code = 409
        return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}

    contexts[key] = {"version": body.version, "payload": body.payload}
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}", "stored_at": datetime.utcnow().isoformat() + "Z"}


def looks_low_quality(body: str, category: Dict) -> Optional[str]:
    """Cheap, domain-agnostic red flags that catch a clearly broken/low-effort composition without
    hardcoding to any specific dataset's wording — this exists because different pooled models (see
    GROQ_MODEL_POOL) vary noticeably in how reliably they follow the style rules above (confirmed
    empirically: the exact same trigger scored anywhere from 26 to 43/50 across separate runs purely
    from which pooled model happened to compose it, with identical underlying data). This is a safety
    net that flags an attempt for retry-on-a-different-model, not a quality judge — it only catches
    unambiguous breakage, not subjective style. Returns a short reason string if flagged, else None."""
    if not body:
        return "empty body"
    word_count = len(body.split())
    if word_count < 6:
        return "suspiciously short (likely generic filler, not a real composition)"
    if word_count > 90:
        return "suspiciously long (likely rambling / lost the 2-4 line WhatsApp constraint)"
    lowered = body.lower()
    for taboo in category.get("voice", {}).get("vocab_taboo", []):
        if taboo and taboo.lower() in lowered:
            return f"taboo word leaked: {taboo}"
    for leaked_term in ("trigger", "payload", "suppression_key", "as an ai", "context block",
                        "system prompt"):
        if leaked_term in lowered:
            return f"leaked internal jargon: {leaked_term}"
    return None


async def try_compose_once(category: Dict, merchant: Dict, trigger: Dict, customer: Optional[Dict],
                            user_prompt: str, merchant_id: str, customer_id: Optional[str],
                            trigger_id: str, exclude: set) -> tuple:
    """One attempt at an LLM composition: pick a pooled model (excluding any already tried),
    call it, parse the result. Returns (action_or_None, model_used) so the caller can track which
    models have been tried and decide whether to retry on quality-gate failure."""
    prompt_tokens_est = estimate_tokens(COMPOSER_SYSTEM_PROMPT, user_prompt)
    estimated = prompt_tokens_est + completion_budget("openai/gpt-oss-20b")
    chosen_model = await GROQ_LIMITER.acquire(estimated, exclude=exclude)
    max_tokens = completion_budget(chosen_model)
    extra_kwargs: Dict[str, Any] = reasoning_kwargs(chosen_model)

    action = None
    call_failed = True
    try:
        async with GROQ_CONCURRENCY:
            response = await asyncio.wait_for(
                groq_client.chat.completions.create(
                    model=chosen_model,
                    messages=[
                        {"role": "system", "content": COMPOSER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=max_tokens,
                    **extra_kwargs,
                ),
                timeout=23,
            )

        usage = getattr(response, "usage", None)
        actual_tokens = getattr(usage, "total_tokens", None) if usage else None
        call_failed = False
        await GROQ_LIMITER.release(chosen_model, estimated, actual_tokens if actual_tokens else estimated)

        text_response = (response.choices[0].message.content or "").strip()
        if not text_response:
            raise ValueError("empty completion (likely reasoning-token budget exhausted)")
        llm_output = json.loads(text_response)

        action = {
            "conversation_id": f"conv_{merchant_id}_{trigger_id}",
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": llm_output.get("send_as", "merchant_on_behalf" if customer else "vera"),
            "trigger_id": trigger_id,
            "body": sanitize_text(llm_output.get("body", "").strip()),
            "cta": llm_output.get("cta", "none"),
            "suppression_key": trigger.get("suppression_key") or llm_output.get("suppression_key", trigger_id),
            "rationale": sanitize_text(llm_output.get("rationale", "No rationale provided.")),
        }
        if not action["body"]:
            action = None

    except Exception as e:
        print(f"Composer error on {trigger_id} (model={chosen_model}): {e!r}")
    finally:
        if call_failed:
            await GROQ_LIMITER.release(chosen_model, estimated, None)

    return action, chosen_model


async def compose_action(trigger_id: str) -> Optional[Dict]:
    """Build one action for one trigger. Returns None if the trigger/merchant/category can't be
    resolved (caller should skip it entirely, not fabricate one)."""
    trigger = get_payload("trigger", trigger_id)
    if not trigger:
        return None

    merchant_id = trigger.get("merchant_id")
    customer_id = trigger.get("customer_id")

    merchant = get_payload("merchant", merchant_id)
    if not merchant:
        return None

    category_slug = merchant.get("category_slug")
    category = get_payload("category", category_slug)
    if not category:
        return None

    customer = get_payload("customer", customer_id) if customer_id else None
    user_prompt = build_user_prompt(category, merchant, trigger, customer)

    # Up to 2 attempts. Tick has a hard 30s budget for ALL triggers combined (judge spec), and
    # triggers are composed concurrently (see tick()), so a second attempt costs at most one more
    # ~10-20s call before tick()'s own cancellation deadline kicks in — worst case this trigger just
    # lands on the rule-based fallback exactly like a single failed attempt would have anyway, so the
    # retry is effectively free insurance against a bad first draft, not a risk.
    tried_models: set = set()
    action = None
    for attempt in range(2):
        result, model_used = await try_compose_once(
            category, merchant, trigger, customer, user_prompt, merchant_id, customer_id, trigger_id,
            exclude=tried_models,
        )
        tried_models.add(model_used)
        if result is None:
            continue
        flag = looks_low_quality(result["body"], category)
        if flag is None:
            action = result
            break
        print(f"Composer quality gate flagged {trigger_id} on attempt {attempt + 1} "
              f"(model={model_used}): {flag}")
        action = result  # keep the latest attempt as a candidate in case the retry also gets flagged

    if not action:
        action = build_fallback_action(category, merchant, trigger, customer, merchant_id, customer_id, trigger_id)

    conv_meta[action["conversation_id"]] = {
        "merchant_id": merchant_id, "customer_id": customer_id, "trigger_id": trigger_id,
    }
    return action


@app.post("/v1/tick")
async def tick(body: TickBody):
    # Compose all triggers concurrently. compose_action() never raises — it always resolves to either
    # an LLM-composed action or a rule-based fallback action — so the only reason a task wouldn't
    # finish here is a genuinely stuck call. In that rare case, don't nuke every result in the batch:
    # salvage whatever finished and synthesize a fast rule-based fallback for whatever didn't, so one
    # slow trigger never costs the whole tick.
    tasks = {tid: asyncio.create_task(compose_action(tid)) for tid in body.available_triggers}
    done, pending = await asyncio.wait(tasks.values(), timeout=27)

    actions = []
    for tid, task in tasks.items():
        if task in pending:
            task.cancel()
            trigger = get_payload("trigger", tid)
            if not trigger:
                continue
            merchant = get_payload("merchant", trigger.get("merchant_id"))
            category = get_payload("category", merchant.get("category_slug")) if merchant else None
            if not merchant or not category:
                continue
            customer_id = trigger.get("customer_id")
            customer = get_payload("customer", customer_id) if customer_id else None
            fallback = build_fallback_action(category, merchant, trigger, customer,
                                              trigger.get("merchant_id"), customer_id, tid)
            conv_meta[fallback["conversation_id"]] = {
                "merchant_id": trigger.get("merchant_id"), "customer_id": customer_id, "trigger_id": tid,
            }
            actions.append(fallback)
            continue
        try:
            result = task.result()
        except Exception as e:
            print(f"tick: task for {tid} raised unexpectedly: {e!r}")
            result = None
        if result:
            actions.append(result)

    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    history = conversations.setdefault(body.conversation_id, [])
    history.append({"from": body.from_role, "msg": body.message})
    merchant_msg = body.message.lower().strip()

    from_messages = [m["msg"] for m in history if m["from"] == body.from_role]
    if len(from_messages) >= 3 and from_messages[-1] == from_messages[-2] == from_messages[-3]:
        return {"action": "end", "rationale": "Same message received 3x verbatim — detected WhatsApp Business auto-reply, exiting to avoid wasting turns."}

    hostile_keywords = ["stop", "spam", "unsubscribe", "useless", "annoying", "harass"]
    if any(re.search(rf"\b{re.escape(w)}\b", merchant_msg) for w in hostile_keywords):
        return {"action": "end", "rationale": "Honoring opt-out / hostility signal."}

    commitment_keywords = ["let's do it", "lets do it", "go ahead", "yes", "ok", "okay", "sure", "whats next", "what's next", "haan", "theek hai", "chalega"]
    if any(w in merchant_msg for w in commitment_keywords):
        meta = conv_meta.get(body.conversation_id, {})
        merchant = get_payload("merchant", meta.get("merchant_id") or body.merchant_id) or {}
        name = merchant.get("identity", {}).get("owner_first_name") or merchant.get("identity", {}).get("name", "")
        offers = active_offers(merchant)
        offer_txt = f" for {offers[0]}" if offers else ""
        greet = f"{name}, " if name else ""
        return {
            "action": "send",
            "body": f"{greet}done — drafting the details{offer_txt} now and will send it over to confirm.",
            "cta": "open_ended",
            "rationale": "Merchant showed explicit commitment intent; skipping further qualification and moving straight to action mode.",
        }

    return {"action": "wait", "wait_seconds": 60, "rationale": "No clear signal yet; waiting before the next nudge."}