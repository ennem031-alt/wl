"""
AA Trade Evaluator - core logic (v7).

Independent of Discord: configuration, text parsing, value list, trade history,
and the value model.

Key ideas:
  - Prices come from value_list.txt first. Units that are not listed can get an
    estimate from community votes, but only when they have enough votes, and the
    estimate is bounded so it cannot explode.
  - Units can carry stats: "dark mage with sss dmg and sss rng" is the unit "dark mage"
    plus two stat parts. Its value is the sum of the parts.
  - Filler words ("value", "what", "ano"), stat words alone, and "evo"/"unevo" alone
    are never units. Chat noise ("overpaying", "or any") is never a unit either.
  - "evo" and "unevo" are part of the unit name. An evo unit is priced separately.
  - Typos are corrected only against the value list vocabulary, never against chat.
  - Unit-like names that are not priced are logged in pending_units.json for review.
  - Percent values are stripped from names. They are not used in prices yet.
  - Saves and refits are batched ("dirty" flag).
"""

import difflib
import json
import math
import os
import re
import statistics
import sys
import threading
import time
from collections import Counter
from datetime import datetime

# ============================================================
#  PATHS AND CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
DATA_DIR = os.environ.get("DATA_DIR", BASE_DIR)
os.makedirs(DATA_DIR, exist_ok=True)
DB_FILE = os.path.join(DATA_DIR, "values_db.json")
PENDING_FILE = os.path.join(DATA_DIR, "pending_units.json")
VALUE_LIST_FILE = os.path.join(BASE_DIR, "value_list.txt")

BOT_TOKEN = ""
USER_TOKEN = ""
HISTORY_AUTH = ""
VALUE_CHANNEL_ID = 0


def load_config():
    """Read settings from environment variables first (Render), then config.json (local)."""
    global BOT_TOKEN, USER_TOKEN, HISTORY_AUTH, VALUE_CHANNEL_ID
    cfg = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            sys.exit(f"[FATAL] Could not read config.json: {e}")

    BOT_TOKEN = (os.environ.get("BOT_TOKEN") or cfg.get("bot_token", "")).strip()
    USER_TOKEN = (os.environ.get("USER_TOKEN") or cfg.get("user_token", "")).strip()
    channel = os.environ.get("VALUE_CHANNEL_ID") or cfg.get("value_channel_id", "0")
    try:
        VALUE_CHANNEL_ID = int(str(channel).strip())
    except ValueError:
        sys.exit("[FATAL] VALUE_CHANNEL_ID must be a number.")

    if not BOT_TOKEN:
        sys.exit("[FATAL] BOT_TOKEN is not set. Add it as an environment variable or in config.json.")

    HISTORY_AUTH = USER_TOKEN if USER_TOKEN else f"Bot {BOT_TOKEN}"


# ============================================================
#  TUNING
# ============================================================
BETA = 2.0
LAM_LIST = 12.0          # anchor strength for value-list units
LAM_EST = 3.0            # anchor strength for shiny estimates
LAM_UNLISTED = 1.0       # anchor strength for units that are not listed (votes must work hard)
CLAMP_FACTOR = 20.0      # unlisted estimates stay within median / 20 .. median * 20
ITERATIONS = 30
HALF_LIFE_DAYS = 60.0
FAIR_BAND = 0.07
MIN_VOTES = 3            # votes needed before a unit that is not listed gets a price
SHINY_MULT_DEFAULT = 2.0
TYPO_CUTOFF = 0.8
TYPO_MARGIN = 0.05
MIN_TYPO_LEN = 3
MIN_NAME_LEN = 4         # shortest unit name that is logged as pending

FILLER_WORDS = {
    "what", "whats", "is", "are", "the", "of", "a", "an", "how", "much",
    "ano", "anong", "ni", "ng", "sa", "si", "ba", "po", "pa", "yung", "yong",
    "magkano", "kay", "para", "value", "values", "price", "worth", "check",
    "tell", "me", "current", "now",
}

# Common chat words that are never unit names. Used to keep noise out of pending and votes.
CHAT_NOISE = {
    "or", "any", "and", "for", "my", "win", "lose", "is", "what", "value",
    "price", "trade", "trading", "tradin", "offer", "offe", "overpaying", "screenshare",
    "high", "triple", "low", "fair", "pay", "paying", "sure", "yes", "no", "ok",
    "lol", "pls", "please", "bro", "sir", "ty", "thanks", "check",
}

# ============================================================
#  TEXT PARSING
# ============================================================
SHINY_RE = re.compile(r"\bshiny\b")
MY_RE = re.compile(r"(?<!\S)my\s+")
COMMAND_PREFIX_RE = re.compile(r"^(?:!wl|is|wtf)(?:\s+|$)")
WIN_LOSE_RE = re.compile(r"\s+win\s+or\s+lose$")
SIDE_SPLIT_RE = re.compile(r"\s+(?:for|vs)\s+")
PCT_RE = re.compile(r"\(?\s*\d+(?:\.\d+)?\s*%\s*\)?")

RANKS = {"sss", "ss", "s"}
STAT_WORDS = {
    "dmg": "dmg", "damage": "dmg",
    "rng": "rng", "range": "rng",
    "spa": "spa",
    "stats": "stats", "stat": "stats",
}
STAT_ONLY_RE = re.compile(r"(?:sss|ss|s)\s+(?:dmg|damage|spa|rng|range|stats?)")
# "... and sss range": the "and" joins a stat, not a new unit.
AND_STAT_RE = re.compile(r"\s+and\s+(?=(?:sss|ss|s)\s+(?:dmg|damage|spa|rng|range|stats?)\b)")
EVO_WORDS = {"evo", "unevo"}
SKIP_WORDS = {"with", "and"}
# Words that can never be a unit on their own.
STAT_TOKENS = set(STAT_WORDS) | RANKS | {"stat", "stats", "with", "and"}


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def pct_of(raw):
    """Percent in a message, e.g. 'sss dmg 45%' -> 45.0. Kept for later pricing."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", raw)
    return float(m.group(1)) if m else None


def norm_key(raw, shiny_aware=True):
    """One key for 'shiny sakamoto', 'Sakamoto (Shiny)' and '2x Sakamoto SHINY'."""
    name = clean(raw.lower())
    name = PCT_RE.sub(" ", name)
    name = re.sub(r"^my\s+", "", name)
    name = re.sub(r"^\d+\s*x\s*|^\d+\s+", "", name)
    shiny = bool(shiny_aware and SHINY_RE.search(name))
    if shiny_aware:
        name = SHINY_RE.sub(" ", name)
    name = name.replace("'", "")
    name = re.sub(r"[^\w\s]", " ", name)
    name = clean(name)
    if not name:
        return ""
    return f"{name} shiny" if shiny else name


def strip_filler(key):
    """'what is dio value' -> 'dio'. Returns '' if nothing is left."""
    return " ".join(t for t in key.split() if t not in FILLER_WORDS)


def learn_key(raw):
    """The key a user's choice is saved under. Same normalization as resolve_query."""
    key = fix_typos(norm_key(raw))
    return strip_filler(key) or key


def split_stats(key):
    """'dark mage with sss dmg' -> ('dark mage', ['sss dmg'])."""
    base, stats = [], []
    toks = key.split()
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in SKIP_WORDS:
            i += 1
            continue
        if t in RANKS and i + 1 < len(toks) and toks[i + 1] in STAT_WORDS:
            stats.append(f"{t} {STAT_WORDS[toks[i + 1]]}")
            i += 2
            continue
        base.append(t)
        i += 1
    return " ".join(base), stats


# ============================================================
#  VALUE LIST CATALOG
# ============================================================
LIST_VALUE = {}
DISPLAY = {}
ALIAS_MAP = {}
AMBIG = {}
ALIAS_KEYS = []
COMPOSITE = {}
SHINY_MULT = SHINY_MULT_DEFAULT
DEFAULT_VALUE = 100.0
_INGEST_MEMO = {}

VOCAB = set()
FIX_CACHE = {}
SKIP_TOKENS = {"shiny"} | SKIP_WORDS | EVO_WORDS | RANKS | set(STAT_WORDS) | FILLER_WORDS

PENDING = Counter()


def build_vocab():
    """Words from the value list only. Chat is never a source, so typos don't spread."""
    VOCAB.clear()
    FIX_CACHE.clear()
    for k in LIST_VALUE:
        for t in k.split():
            if t != "shiny" and not t.isdigit():
                VOCAB.add(t)


def fix_token(tok):
    if tok in VOCAB or tok in SKIP_TOKENS or len(tok) < MIN_TYPO_LEN or tok.isdigit():
        return tok
    if tok in FIX_CACHE:
        return FIX_CACHE[tok]
    out = tok
    best = difflib.get_close_matches(tok, VOCAB, n=2, cutoff=TYPO_CUTOFF)
    if best:
        r1 = difflib.SequenceMatcher(None, tok, best[0]).ratio()
        r2 = difflib.SequenceMatcher(None, tok, best[1]).ratio() if len(best) > 1 else 0.0
        if r1 - r2 >= TYPO_MARGIN:
            out = best[0]
    FIX_CACHE[tok] = out
    return out


def fix_typos(key):
    return " ".join(fix_token(t) for t in key.split())


def corrections(raw):
    """Changed words as [(typed, fixed), ...]."""
    before = norm_key(raw).split()
    after = fix_typos(norm_key(raw)).split()
    return [(b, a) for b, a in zip(before, after) if b != a]


def shiny_estimate(name):
    if name.endswith(" shiny") and name not in LIST_VALUE:
        base = ALIAS_MAP.get(name[:-6])
        if base and base in LIST_VALUE and not base.endswith(" shiny"):
            return LIST_VALUE[base] * SHINY_MULT
    return None


def load_value_list():
    global SHINY_MULT, DEFAULT_VALUE, ALIAS_KEYS
    LIST_VALUE.clear()
    DISPLAY.clear()
    ALIAS_MAP.clear()
    AMBIG.clear()
    COMPOSITE.clear()
    _INGEST_MEMO.clear()

    if not os.path.exists(VALUE_LIST_FILE):
        print(f"[WARN] {VALUE_LIST_FILE} not found. Only community votes will be used.")
        build_vocab()
        return

    entries = []
    with open(VALUE_LIST_FILE, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "|" not in line:
                continue
            val_s, names = line.split("|", 1)
            try:
                val = float(val_s.replace(",", "").strip())
            except ValueError:
                continue
            shiny = bool(re.search(r"\(\s*shiny\s*\)", names, re.IGNORECASE))
            names = re.sub(r"\(\s*shiny\s*\)", "", names, flags=re.IGNORECASE)
            parts = [p.strip() for p in names.split("/") if p.strip()]
            aliases = [k for k in (norm_key(p, shiny_aware=False) for p in parts) if k]
            if aliases:
                entries.append((val, parts, aliases, shiny))

    alias_sets = {}
    for val, parts, aliases, shiny in entries:
        suffix = " shiny" if shiny else ""
        keys = [a + suffix for a in aliases]
        base = canonical = keys[-1]
        n = 2
        while canonical in LIST_VALUE:
            canonical = f"{base} {n}"
            n += 1
        LIST_VALUE[canonical] = val
        DISPLAY[canonical] = " / ".join(parts) + (" (Shiny)" if shiny else "")
        for k in keys + [canonical]:
            alias_sets.setdefault(k, set()).add(canonical)

    for k, s in alias_sets.items():
        if len(s) == 1:
            ALIAS_MAP[k] = next(iter(s))
        elif k in s:
            ALIAS_MAP[k] = k
        else:
            AMBIG[k] = s

    ratios = []
    for c, v in LIST_VALUE.items():
        if c.endswith(" shiny"):
            base = ALIAS_MAP.get(c[:-6])
            if base and base in LIST_VALUE and not base.endswith(" shiny") and LIST_VALUE[base] > 0:
                ratios.append(math.log(v / LIST_VALUE[base]))
    if ratios:
        SHINY_MULT = math.exp(statistics.median(ratios))
    if LIST_VALUE:
        DEFAULT_VALUE = statistics.median(LIST_VALUE.values())
    ALIAS_KEYS = list(ALIAS_MAP.keys())
    build_vocab()
    print(f"[OK] Loaded {len(LIST_VALUE)} units from value list (shiny factor x{SHINY_MULT:.2f}).")


def display_name(canonical):
    if canonical in COMPOSITE:
        return " + ".join(display_name(p) for p in COMPOSITE[canonical])
    return DISPLAY.get(canonical, canonical.title())


def is_junk(name):
    """
    True when a name is not a unit at all: only stat words, filler words, evo words,
    or 'shiny'. Composite items are junk if any of their parts is junk.
    """
    if not name:
        return True
    if name in COMPOSITE:
        return any(is_junk(p) for p in COMPOSITE[name])
    if name in LIST_VALUE:
        return False
    toks = name.replace(" shiny", "").split()
    return not toks or all(
        t in STAT_TOKENS or t in FILLER_WORDS or t in EVO_WORDS or t == "shiny"
        for t in toks)


def is_unitlike(name):
    """
    A name that could plausibly be a unit. Listed units always pass. Unlisted names must
    be long enough, short in word count, and free of chat noise and stat or evo words.
    """
    if not name or is_junk(name):
        return False
    if name in COMPOSITE:
        return all(is_unitlike(p) for p in COMPOSITE[name])
    if name in LIST_VALUE:
        return True
    base = name.replace(" shiny", "")
    toks = base.split()
    if len(base) < MIN_NAME_LEN or len(toks) > 4:
        return False
    for t in toks:
        if t.isdigit() or t in CHAT_NOISE or t in FILLER_WORDS or t in STAT_TOKENS:
            return False
    return True


def is_known(name):
    """
    A unit the bot can price: in the value list, a shiny estimate from it, or an unlisted
    unit with enough votes to have a fitted estimate.
    """
    if name in LIST_VALUE or shiny_estimate(name) is not None:
        return True
    return (name in VALUES and VOTE_COUNT.get(name, 0) >= MIN_VOTES
            and is_unitlike(name))


def priced(name):
    if name in COMPOSITE:
        return all(priced(p) for p in COMPOSITE[name])
    return is_known(name)


def build_composite(base_key, stat_keys):
    parts = []
    if base_key:
        status, name, _ = resolve_query(base_key)
        parts.append(name if status == "ok" else base_key)
    for s in stat_keys:
        status, name, _ = resolve_query(s)
        parts.append(name if status == "ok" else s)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    name = " + ".join(parts)
    COMPOSITE[name] = parts
    return name


def note_pending(name):
    """Count a unit-like name the bot cannot price yet. Reviewed with !pending."""
    if not is_unitlike(name):
        return
    with DATA_LOCK:
        PENDING[name] += 1


def top_pending(n=20):
    with DATA_LOCK:
        return PENDING.most_common(n)


def load_pending():
    if not os.path.exists(PENDING_FILE):
        return
    try:
        with open(PENDING_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        with DATA_LOCK:
            # Drop noise that older versions may have logged.
            PENDING.update({k: v for k, v in data.items() if is_unitlike(k)})
    except Exception as e:
        print(f"[WARN] Could not read pending_units.json: {e}")


def save_pending():
    with DATA_LOCK:
        snapshot = dict(PENDING)
    tmp = PENDING_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snapshot, f)
    os.replace(tmp, PENDING_FILE)


def _resolve_for_ingest(key):
    base, stats = split_stats(key)
    if key in ALIAS_MAP:
        return ALIAS_MAP[key]
    if key in AMBIG:
        return None
    if stats:
        return build_composite(base, stats)
    close = difflib.get_close_matches(key, ALIAS_KEYS, n=1, cutoff=0.9)
    return ALIAS_MAP[close[0]] if close else key


def canon_ingest(raw):
    """
    Map a name from chat to a canonical item. Returns None for empty or ambiguous names.
    Unknown names are kept as-is so votes can still be recorded.
    """
    key = fix_typos(norm_key(raw))
    if not key:
        return None
    if key in _INGEST_MEMO:
        return _INGEST_MEMO[key]
    out = _resolve_for_ingest(key)
    _INGEST_MEMO[key] = out
    return out


def parse_trade(content, track=False):
    """
    Parse a W/L post. Returns (offered, requested) or None. A trade is rejected when
    any side is empty, ambiguous, or contains junk (stat, filler, or evo words alone).
    """
    parsed = split_sides(content)
    if not parsed or not parsed[1]:
        return None
    give_raw, get_raw = parsed
    offered = [canon_ingest(x) for x in give_raw]
    requested = [canon_ingest(x) for x in get_raw]
    if not all(offered) or not all(requested):
        return None
    if any(is_junk(i) for i in offered + requested):
        return None
    if track:
        for i in offered + requested:
            if i not in COMPOSITE and not is_known(i):
                note_pending(i)
    return offered, requested


# ============================================================
#  SEGMENTATION (no comma needed)
# ============================================================
def is_known_chunk(chunk, lookup=None):
    key = fix_typos(norm_key(chunk))
    if not key:
        return False
    if lookup and lookup(key):
        return True
    if key in AMBIG:
        return True
    return is_known(ALIAS_MAP.get(key, key))


def group_by_evo(words):
    """
    'evo' ends a unit and 'unevo' starts a new one:
      'freiren evo sharkfin evo'  -> [['freiren','evo'], ['sharkfin','evo']]
      'cream evo unevo valentine' -> [['cream','evo'], ['unevo','valentine']]
    """
    groups, cur = [], []
    for w in words:
        if w == "unevo" and cur:
            groups.append(cur)
            cur = []
        cur.append(w)
        if w == "evo":
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def segment(words, lookup=None, max_len=5):
    """
    Dynamic programming over the words. A known unit scores higher the longer it is,
    so 'dark mage' is one item. Unknown words score -1 each. Stat phrases score
    strongly so they stay attached to the unit before them.
    Returns [(kind, text)] where kind is 'k' (known), 's' (stat), or '?' (unknown).
    """
    n = len(words)
    best = [(0.0, [])] * (n + 1)
    for i in range(n - 1, -1, -1):
        w = words[i]
        cands = [(best[i + 1][0] - 1.0, [("?", w)] + best[i + 1][1])]
        if w in SKIP_WORDS:
            cands.append(best[i + 1])
        for j in range(i + 1, min(n, i + max_len) + 1):
            chunk = " ".join(words[i:j])
            if STAT_ONLY_RE.fullmatch(chunk):
                cands.append((best[j][0] + 2.0 * (j - i), [("s", chunk)] + best[j][1]))
            elif is_known_chunk(chunk, lookup):
                cands.append((best[j][0] + (j - i) ** 1.5, [("k", chunk)] + best[j][1]))
        best[i] = max(cands, key=lambda c: c[0])

    out, unk = [], []
    for kind, text in best[0][1]:
        if kind == "?":
            unk.append(text)
            continue
        if unk:
            out.append(("?", " ".join(unk)))
            unk = []
        out.append((kind, text))
    if unk:
        out.append(("?", " ".join(unk)))
    return out


def split_items(text, lookup=None):
    """
    Split one side of a trade into items, with or without commas.
    Filler words are removed first. A "shiny" anywhere in a part marks the unit in that part.
    """
    # "dark mage with sss dmg and sss rng": the "and" joins stats, so keep them on the unit.
    text = AND_STAT_RE.sub(" with ", text)
    parts = re.split(r"\s+and\s+|\s*[+,&]\s*", text)
    out = []
    for p in (clean(x) for x in parts):
        if not p:
            continue
        is_shiny = bool(SHINY_RE.search(p))
        p = SHINY_RE.sub(" ", p)
        words = [w for w in fix_typos(norm_key(p, shiny_aware=False)).split()
                 if w not in FILLER_WORDS]
        shiny_done = False
        for group in group_by_evo(words):
            for kind, chunk in segment(group, lookup):
                if kind == "s" and out:
                    out[-1] = f"{out[-1]} with {chunk}"
                    continue
                if is_shiny and not shiny_done and kind != "s":
                    chunk = f"{chunk} shiny"
                    shiny_done = True
                out.append(chunk)
    return out


def split_sides(text, lookup=None):
    """
    'is my Goku for Gohan win or lose' -> (['goku'], ['gohan'])
    'is Gohan for my Goku'             -> (['goku'], ['gohan'])
    'is shiny sakamoto'                -> (['sakamoto shiny'], [])
    """
    s = clean(text.lower().replace("?", " "))
    s = COMMAND_PREFIX_RE.sub("", s)
    s = WIN_LOSE_RE.sub("", s)
    parts = SIDE_SPLIT_RE.split(s, maxsplit=1)

    if len(parts) == 1:
        items = split_items(MY_RE.sub("", parts[0]), lookup)
        return (items, []) if items else None

    left, right = parts
    left_mine = bool(MY_RE.search(left))
    right_mine = bool(MY_RE.search(right))
    left = MY_RE.sub("", left)
    right = MY_RE.sub("", right)
    give_s, get_s = (right, left) if (right_mine and not left_mine) else (left, right)
    give, get = split_items(give_s, lookup), split_items(get_s, lookup)
    if not give or not get:
        return None
    return give, get


# ============================================================
#  RESOLVE ONE QUERY
# ============================================================
def resolve_query(raw, choice_lookup=None):
    """Returns (status, name, options). status: 'ok' | 'ambiguous' | 'unknown'."""
    key = fix_typos(norm_key(raw))
    if not key:
        return "unknown", None, []

    stripped = strip_filler(key)
    candidates = [key] + ([stripped] if stripped and stripped != key else [])

    for k in candidates:
        if choice_lookup:
            hit = choice_lookup(k)
            if hit:
                return "ok", hit, []
        if k in AMBIG:
            opts = [(display_name(c), c) for c in sorted(AMBIG[k])][:4]
            return "ambiguous", None, opts
        name = ALIAS_MAP.get(k, k)
        if is_known(name):
            return "ok", name, []

    for k in [stripped or key, key]:
        base, stats = split_stats(k)
        if stats and base:
            return "ok", build_composite(base, stats), []

    for k in [stripped or key, key]:
        base, stats = split_stats(k)
        if stats and not base and len(stats) == 1:
            name = ALIAS_MAP.get(stats[0], stats[0])
            if is_known(name):
                return "ok", name, []

    return "unknown", None, suggest(stripped or key)


def suggest(key, n=4):
    tokens = {t for t in key.split() if t != "shiny"}
    if not tokens:
        return []
    want_shiny = key.endswith(" shiny")
    pool = set(ALIAS_KEYS)
    scored = []
    for cand in pool:
        ctoks = {t for t in cand.split() if t != "shiny"}
        overlap = len(tokens & ctoks)
        if not overlap:
            continue
        score = overlap / max(len(tokens), len(ctoks))
        score += 0.5 * difflib.SequenceMatcher(None, key, cand).ratio()
        if cand.endswith(" shiny") == want_shiny:
            score += 0.2
        scored.append((score, cand))
    scored.sort(key=lambda x: (-x[0], x[1]))
    out, seen = [], set()
    for _, cand in scored:
        canonical = ALIAS_MAP.get(cand, cand)
        if canonical in seen or is_junk(canonical):
            continue
        seen.add(canonical)
        out.append((display_name(canonical), canonical))
        if len(out) >= n:
            break
    return out


def source_label(name):
    if name in COMPOSITE:
        return "Unit + stats (sum of parts)"
    if name in LIST_VALUE:
        return "Value list"
    if shiny_estimate(name) is not None:
        return "Shiny estimate"
    if is_known(name):
        return "Community votes (estimate)"
    return "No price yet"


def vote_count_of(name):
    if name in COMPOSITE:
        return min((vote_count_of(p) for p in COMPOSITE[name]), default=0)
    return VOTE_COUNT.get(name, 0)


# ============================================================
#  TRADE HISTORY STORE
# ============================================================
DATA_LOCK = threading.RLock()
TRADE_DB = {}
DIRECT_INDEX = {}
DIRTY = {"flag": False}


def parse_ts(value):
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def mark_dirty():
    DIRTY["flag"] = True


def is_dirty():
    return DIRTY["flag"]


def _index_add(t):
    key = (frozenset(t["offered"]), frozenset(t["requested"]))
    DIRECT_INDEX.setdefault(key, set()).add(t["id"])


def _index_remove(t):
    key = (frozenset(t["offered"]), frozenset(t["requested"]))
    ids = DIRECT_INDEX.get(key)
    if ids:
        ids.discard(t["id"])
        if not ids:
            DIRECT_INDEX.pop(key, None)


def load_db():
    load_pending()
    if not os.path.exists(DB_FILE):
        print("[INFO] No values_db.json yet. Run !start scanning to build it.")
        return
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"[WARN] Could not read values_db.json: {e}")
        return

    loaded, skipped = {}, 0
    for key, t in raw.items():
        offered = [c for c in (canon_ingest(x) for x in t.get("offered", [])) if c]
        requested = [c for c in (canon_ingest(x) for x in t.get("requested", [])) if c]
        if not offered or not requested:
            continue
        if any(is_junk(i) for i in offered + requested):
            skipped += 1
            continue
        ts = t.get("ts") or parse_ts(t.get("timestamp", ""))
        loaded[key] = {
            "id": key,
            "offered": offered,
            "requested": requested,
            "ups": int(t.get("ups", 0)),
            "downs": int(t.get("downs", 0)),
            "timestamp": t.get("timestamp", ""),
            "author": str(t.get("author", "")),
            "ts": float(ts or 0.0),
        }
    with DATA_LOCK:
        TRADE_DB.clear()
        TRADE_DB.update(loaded)
    if skipped:
        print(f"[CLEAN] Removed {skipped} junk trades from database.")
        mark_dirty()
    print(f"[OK] Loaded {len(loaded)} trades from database.")


def save_db():
    tmp = DB_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(TRADE_DB, f)
    os.replace(tmp, DB_FILE)


def _reactions(m):
    ups = downs = 0
    for r in m.get("reactions", []):
        name = (r.get("emoji") or {}).get("name")
        if name == "👍":
            ups = r.get("count", 0)
        elif name == "👎":
            downs = r.get("count", 0)
    return ups, downs


def ingest(messages, recalc=False):
    """
    Add W/L posts and refresh reaction counts. Nothing is saved here; call flush().
    Messages without a 'reactions' key (live gateway events) never overwrite votes.
    """
    changed = 0
    with DATA_LOCK:
        for m in messages:
            content = m.get("content", "")
            if not content.lower().startswith("!wl"):
                continue
            key = str(m["id"])
            existing = TRADE_DB.get(key)

            if existing:
                if "reactions" not in m:
                    continue
                ups, downs = _reactions(m)
                if existing["ups"] != ups or existing["downs"] != downs:
                    existing["ups"], existing["downs"] = ups, downs
                    changed += 1
                continue

            parsed = parse_trade(content, track=True)
            if not parsed:
                continue
            offered, requested = parsed
            ups, downs = _reactions(m)
            ts_text = m.get("timestamp", "")
            t = {
                "id": key,
                "offered": offered,
                "requested": requested,
                "ups": ups,
                "downs": downs,
                "timestamp": ts_text,
                "author": str((m.get("author") or {}).get("id", "")),
                "ts": parse_ts(ts_text),
            }
            TRADE_DB[key] = t
            _index_add(t)
            changed += 1

        if changed:
            mark_dirty()
    if changed and recalc:
        flush()
    return changed


def bump_reaction(message_id, emoji, delta):
    if emoji not in ("👍", "👎"):
        return False
    with DATA_LOCK:
        t = TRADE_DB.get(message_id)
        if not t:
            return False
        if emoji == "👍":
            t["ups"] = max(0, t["ups"] + delta)
        else:
            t["downs"] = max(0, t["downs"] + delta)
        mark_dirty()
    return True


def remove_trade(message_id):
    with DATA_LOCK:
        t = TRADE_DB.pop(message_id, None)
        if not t:
            return False
        _index_remove(t)
        mark_dirty()
    return True


def flush():
    with DATA_LOCK:
        if not DIRTY["flag"]:
            return False
        save_db()
        save_pending()
        DIRTY["flag"] = False
    recalculate()
    return True


# ============================================================
#  VALUE MODEL
# ============================================================
VALUES = {}
VOTE_ITEMS = set()
VOTE_COUNT = {}


def prior(name):
    if name in LIST_VALUE:
        return math.log(LIST_VALUE[name]), LAM_LIST
    est = shiny_estimate(name)
    if est:
        return math.log(est), LAM_EST
    return math.log(DEFAULT_VALUE), LAM_UNLISTED


def value_of(name, theta=None):
    if name in COMPOSITE:
        return sum(value_of(p, theta) for p in COMPOSITE[name])
    source = VALUES if theta is None else theta
    if name in source:
        return math.exp(source[name])
    return math.exp(prior(name)[0])


def decay(ts, now):
    if not ts:
        return 0.5
    age_days = max(0.0, (now - ts) / 86400.0)
    return 0.5 ** (age_days / HALF_LIFE_DAYS)


def has_composite(t):
    return any(i in COMPOSITE for i in t["offered"] + t["requested"])


def has_junk(t):
    return any(is_junk(i) for i in t["offered"] + t["requested"])


def all_unitlike(t):
    return all(is_unitlike(i) for i in t["offered"] + t["requested"])


def build_groups(trades, now):
    latest = {}
    for t in trades:
        if t["ups"] == t["downs"] or not t["offered"] or not t["requested"]:
            continue
        o = tuple(sorted(set(t["offered"])))
        r = tuple(sorted(set(t["requested"])))
        who = t.get("author") or t.get("id")
        k = (who, o, r)
        if k not in latest or t.get("ts", 0) > latest[k].get("ts", 0):
            latest[k] = t

    agg = {}
    for (_, o, r), t in latest.items():
        margin = abs(t["ups"] - t["downs"])
        w = min(1 + 0.25 * (margin - 1), 2.0) * decay(t.get("ts", 0), now)
        a = agg.setdefault((o, r), [0.0, 0.0])
        a[0 if t["ups"] > t["downs"] else 1] += w
    return [(o, r, a[0], a[1]) for (o, r), a in agg.items()]


def fit_values(groups, items, priors):
    theta = {i: priors[i][0] for i in items}
    for _ in range(ITERATIONS):
        ex = {i: math.exp(theta[i]) for i in items}
        g = dict.fromkeys(items, 0.0)
        h = dict.fromkeys(items, 0.0)
        for o, r, w1, w0 in groups:
            eo = [ex[i] for i in o]
            er = [ex[i] for i in r]
            vo, vr = sum(eo), sum(er)
            z = math.log(vr) - math.log(vo)
            p = 1 / (1 + math.exp(-BETA * max(-30.0, min(30.0, z))))
            W = w1 + w0
            common = BETA * (w1 - W * p)
            curv = BETA * BETA * p * (1 - p) * W
            for i, e in zip(r, er):
                s = e / vr
                g[i] += common * s
                h[i] += curv * s * s
            for i, e in zip(o, eo):
                s = e / vo
                g[i] -= common * s
                h[i] += curv * s * s
        for i in items:
            mu, lam = priors[i]
            gi = g[i] - lam * (theta[i] - mu)
            hi = h[i] + lam
            step = max(-0.5, min(0.5, gi / hi))
            theta[i] = max(math.log(0.5), min(math.log(1e6), theta[i] + step))
    return theta


def fit_model(trades, now):
    """
    Fit log-values from voted trades. Listed units are anchored strongly to the value list.
    Unlisted units have a weak anchor and their result is clamped to a bounded range
    around the median, so a run of one-sided votes cannot push them to extreme values.
    """
    groups = build_groups(trades, now)
    items = set(LIST_VALUE) | {i for t in trades for i in t["offered"] + t["requested"]}
    priors = {i: prior(i) for i in items}
    theta = fit_values(groups, items, priors)

    lo = math.log(DEFAULT_VALUE / CLAMP_FACTOR)
    hi = math.log(DEFAULT_VALUE * CLAMP_FACTOR)
    for i in items:
        if i not in LIST_VALUE:
            theta[i] = min(max(theta[i], lo), hi)
    return theta


def recalculate():
    """
    Rebuild the direct-vote index and vote counts from TRADE_DB, then refit values.
    Only trades made of unit-like names (listed or plausible unlisted) feed the model.
    """
    global VALUES, VOTE_ITEMS, VOTE_COUNT
    with DATA_LOCK:
        trades = list(TRADE_DB.values())
        valid = [t for t in trades if t["offered"] and t["requested"] and not has_junk(t)]

        DIRECT_INDEX.clear()
        for t in valid:
            _index_add(t)

        voted = [t for t in valid
                 if t["ups"] != t["downs"] and not has_composite(t) and all_unitlike(t)]
        counts = {}
        for t in voted:
            for i in set(t["offered"]) | set(t["requested"]):
                counts[i] = counts.get(i, 0) + 1
        VOTE_COUNT = counts
        VOTE_ITEMS = set(counts)

        VALUES = fit_model(voted, time.time())


def backtest(split=0.8):
    with DATA_LOCK:
        voted = [t for t in TRADE_DB.values()
                 if t["ups"] != t["downs"] and t["offered"] and t["requested"]
                 and not has_composite(t) and not has_junk(t) and all_unitlike(t)]
    voted.sort(key=lambda t: t.get("ts", 0))
    if len(voted) < 20:
        return None

    cut = int(len(voted) * split)
    train, test = voted[:cut], voted[cut:]
    cut_ts = train[-1].get("ts") or time.time()

    theta = fit_model(train, cut_ts)

    n = correct = offerer_wins = 0
    loss = 0.0
    for t in test:
        o = sum(value_of(i, theta) for i in t["offered"])
        r = sum(value_of(i, theta) for i in t["requested"])
        z = max(-30.0, min(30.0, math.log(r) - math.log(o)))
        p = min(max(1 / (1 + math.exp(-BETA * z)), 1e-6), 1 - 1e-6)
        y = 1 if t["ups"] > t["downs"] else 0
        offerer_wins += y
        correct += int((p >= 0.5) == bool(y))
        loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
        n += 1

    if n == 0:
        return None
    share = offerer_wins / n
    return {
        "tested": n,
        "trained_on": len(train),
        "accuracy": correct / n,
        "baseline": max(share, 1 - share),
        "logloss": loss / n,
    }


def direct_votes(give, get):
    sg, st = frozenset(give), frozenset(get)
    with DATA_LOCK:
        forward = [(mid, False) for mid in DIRECT_INDEX.get((sg, st), ())]
        backward = [(mid, True) for mid in DIRECT_INDEX.get((st, sg), ())]
        rows = []
        for mid, flipped in forward + backward:
            t = TRADE_DB.get(mid)
            if t and t["ups"] != t["downs"]:
                rows.append((t, flipped))

    wins = losses = 0
    seen_authors = set()
    for t, flipped in rows:
        author = t.get("author") or t.get("id")
        if author in seen_authors:
            continue
        seen_authors.add(author)
        offerer_won = t["ups"] > t["downs"]
        if offerer_won != flipped:
            wins += 1
        else:
            losses += 1
    return wins, losses
