"""
AA Trade Evaluator - core logic.

Everything in this module is independent of Discord: configuration, text
parsing, the value-list catalog, the trade history store and the value model.
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
from datetime import datetime

# ============================================================
#  PATHS AND CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
DATA_DIR = os.environ.get("DATA_DIR", BASE_DIR)
os.makedirs(DATA_DIR, exist_ok=True)
DB_FILE = os.path.join(DATA_DIR, "values_db.json")
VALUE_LIST_FILE = os.path.join(BASE_DIR, "value_list.txt")

BOT_TOKEN = ""
HISTORY_AUTH = ""
VALUE_CHANNEL_ID = 0


def load_config():
    """
    Read settings from environment variables first (used on Render).
    config.json is only a fallback for running on your own PC.
    Secrets are never stored in the code.
    """
    global BOT_TOKEN, HISTORY_AUTH, VALUE_CHANNEL_ID
    cfg = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            sys.exit(f"[FATAL] Could not read config.json: {e}")

    BOT_TOKEN = (os.environ.get("BOT_TOKEN") or cfg.get("bot_token", "")).strip()
    channel = os.environ.get("VALUE_CHANNEL_ID") or cfg.get("value_channel_id", "0")
    try:
        VALUE_CHANNEL_ID = int(str(channel).strip())
    except ValueError:
        sys.exit("[FATAL] VALUE_CHANNEL_ID must be a number.")

    if not BOT_TOKEN:
        sys.exit("[FATAL] BOT_TOKEN is not set. Add it as an environment variable or in config.json.")

    # History is read with the bot account only. User tokens are not supported.
    HISTORY_AUTH = f"Bot {BOT_TOKEN}"


# ============================================================
#  TUNING
# ============================================================
BETA = 2.0              # steepness of the win probability curve
LAM_LIST = 12.0         # how strongly the value list anchors each unit
LAM_EST = 3.0           # anchor strength for shiny estimates
LAM_DEFAULT = 0.1       # anchor strength for units missing from the list
ITERATIONS = 30         # Newton steps per fit
HALF_LIFE_DAYS = 60.0   # older votes count less
FAIR_BAND = 0.07        # |log ratio| below this is treated as FAIR
MIN_VOTES = 3           # below this, confidence is Low
SHINY_MULT_DEFAULT = 2.0

# Words that are not part of a unit name. They are removed before matching,
# so "what is dio value" and "ano value ni dio" both resolve to "dio".
FILLER_WORDS = {
    "what", "whats", "is", "are", "the", "of", "a", "an", "how", "much",
    "ano", "anong", "ni", "ng", "sa", "si", "ba", "po", "pa", "yung", "yong",
    "magkano", "kay", "para", "value", "values", "price", "worth", "check",
    "tell", "me", "current", "now",
}

# ============================================================
#  TEXT PARSING
# ============================================================
SHINY_RE = re.compile(r"\bshiny\b")
MY_RE = re.compile(r"(?<!\S)my\s+")          # whole word "my" only ("mamy" is a unit)
COMMAND_PREFIX_RE = re.compile(r"^(?:!wl|is|wtf)(?:\s+|$)")
WIN_LOSE_RE = re.compile(r"\s+win\s+or\s+lose$")
SIDE_SPLIT_RE = re.compile(r"\s+(?:for|vs)\s+")


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def norm_key(raw, shiny_aware=True):
    """One key for 'shiny sakamoto', 'Sakamoto (Shiny)' and '2x Sakamoto SHINY'."""
    name = clean(raw.lower())
    name = re.sub(r"^my\s+", "", name)
    name = re.sub(r"^\d+\s*x\s*|^\d+\s+", "", name)   # quantity prefix
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
    """Remove filler words: 'what is dio value' -> 'dio'. Returns '' if nothing is left."""
    return " ".join(t for t in key.split() if t not in FILLER_WORDS)


def learn_key(raw):
    """The key a user's choice is saved under (filler removed when possible)."""
    key = norm_key(raw)
    return strip_filler(key) or key


def split_items(text):
    parts = re.split(r"\s+and\s+|\s*[+,&]\s*", text)
    return [p for p in (clean(x) for x in parts) if p]


def split_sides(text):
    """
    Parse a query or a W/L post.
      'is my Goku for Gohan win or lose' -> (["goku"], ["gohan"])
      'is Gohan for my Goku'             -> (["goku"], ["gohan"])   ('my' marks what YOU give)
      'is shiny sakamoto'                -> (["shiny sakamoto"], []) (single item lookup)
    Returns (give_raw, get_raw) or None.
    """
    s = clean(text.lower().replace("?", " "))
    s = COMMAND_PREFIX_RE.sub("", s)
    s = WIN_LOSE_RE.sub("", s)
    parts = SIDE_SPLIT_RE.split(s, maxsplit=1)

    if len(parts) == 1:
        items = split_items(MY_RE.sub("", parts[0]))
        return (items, []) if items else None

    left, right = parts
    left_mine = bool(MY_RE.search(left))
    right_mine = bool(MY_RE.search(right))
    left = MY_RE.sub("", left)
    right = MY_RE.sub("", right)
    give_s, get_s = (right, left) if (right_mine and not left_mine) else (left, right)
    give, get = split_items(give_s), split_items(get_s)
    if not give or not get:
        return None
    return give, get


# ============================================================
#  VALUE LIST CATALOG
# ============================================================
LIST_VALUE = {}          # canonical name -> value
DISPLAY = {}             # canonical name -> display name
ALIAS_MAP = {}           # alias key -> canonical (unambiguous only)
AMBIG = {}               # alias key -> set of canonical names (ambiguous)
ALIAS_KEYS = []
SHINY_MULT = SHINY_MULT_DEFAULT
DEFAULT_VALUE = 100.0
_INGEST_MEMO = {}


def shiny_estimate(name):
    """Estimated value for a shiny unit that is not listed, based on its non-shiny price."""
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
    _INGEST_MEMO.clear()

    if not os.path.exists(VALUE_LIST_FILE):
        print(f"[WARN] {VALUE_LIST_FILE} not found. Only community votes will be used.")
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
    print(f"[OK] Loaded {len(LIST_VALUE)} units from value list (shiny factor x{SHINY_MULT:.2f}).")


def display_name(canonical):
    return DISPLAY.get(canonical, canonical.title())


def is_known(name):
    return name in LIST_VALUE or name in VOTE_ITEMS or shiny_estimate(name) is not None


def canon_ingest(raw):
    """Map a name found in chat to a canonical unit. None = invalid or ambiguous; skip the trade."""
    key = norm_key(raw)
    if not key:
        return None
    if key in _INGEST_MEMO:
        return _INGEST_MEMO[key]
    if key in ALIAS_MAP:
        out = ALIAS_MAP[key]
    elif key in AMBIG:
        out = None
    else:
        close = difflib.get_close_matches(key, ALIAS_KEYS, n=1, cutoff=0.9)
        out = ALIAS_MAP[close[0]] if close else key
    _INGEST_MEMO[key] = out
    return out


def parse_trade(content):
    parsed = split_sides(content)
    if not parsed or not parsed[1]:
        return None
    give_raw, get_raw = parsed
    offered = [canon_ingest(x) for x in give_raw]
    requested = [canon_ingest(x) for x in get_raw]
    if not all(offered) or not all(requested):
        return None
    return offered, requested


def resolve_query(raw, choice_lookup=None):
    """
    Resolve one item name from a user query.

    choice_lookup(key) -> canonical or None, used for choices the user (or the
    community) made before. Returns (status, canonical, options) where status is
    'ok', 'ambiguous' or 'unknown'.
    """
    key = norm_key(raw)
    if not key:
        return "unknown", None, []

    stripped = strip_filler(key)
    candidates = [key]
    if stripped and stripped != key:
        candidates.append(stripped)

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

    return "unknown", None, suggest(stripped or key)


def suggest(key, n=4):
    """Close matches ranked by shared words, shiny match, and string similarity."""
    tokens = {t for t in key.split() if t != "shiny"}
    if not tokens:
        return []
    want_shiny = key.endswith(" shiny")
    pool = set(ALIAS_KEYS) | VOTE_ITEMS
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
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append((display_name(canonical), canonical))
        if len(out) >= n:
            break
    return out


def source_label(name):
    if name in LIST_VALUE and name in VOTE_ITEMS:
        return "List + votes"
    if name in LIST_VALUE:
        return "Value list"
    if name in VOTE_ITEMS:
        return "Community votes"
    if shiny_estimate(name) is not None:
        return "Shiny estimate"
    return "Unpriced"


# ============================================================
#  TRADE HISTORY STORE
# ============================================================
DATA_LOCK = threading.RLock()
TRADE_DB = {}            # message id -> trade record


def parse_ts(value):
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def load_db():
    if not os.path.exists(DB_FILE):
        print("[INFO] No values_db.json yet. Run !start scanning to build it.")
        return
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"[WARN] Could not read values_db.json: {e}")
        return

    loaded = {}
    for key, t in raw.items():
        offered = [c for c in (canon_ingest(x) for x in t.get("offered", [])) if c]
        requested = [c for c in (canon_ingest(x) for x in t.get("requested", [])) if c]
        if not offered or not requested:
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
    print(f"[OK] Loaded {len(loaded)} trades from database.")


def save_db():
    """Write to a temp file first so a crash never corrupts the database."""
    tmp = DB_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(TRADE_DB, f)
    os.replace(tmp, DB_FILE)


def ingest(messages, recalc=True):
    """Add new W/L posts and refresh votes. Returns how many records were added or updated."""
    changed = 0
    with DATA_LOCK:
        for m in messages:
            content = m.get("content", "")
            if not content.lower().startswith("!wl"):
                continue
            ups = downs = 0
            for r in m.get("reactions", []):
                name = (r.get("emoji") or {}).get("name")
                if name == "👍":
                    ups = r.get("count", 0)
                elif name == "👎":
                    downs = r.get("count", 0)

            key = str(m["id"])
            existing = TRADE_DB.get(key)
            if existing:
                if existing["ups"] == ups and existing["downs"] == downs:
                    continue
                existing["ups"], existing["downs"] = ups, downs
                changed += 1
                continue

            parsed = parse_trade(content)
            if not parsed:
                continue
            offered, requested = parsed
            ts_text = m.get("timestamp", "")
            TRADE_DB[key] = {
                "id": key,
                "offered": offered,
                "requested": requested,
                "ups": ups,
                "downs": downs,
                "timestamp": ts_text,
                "author": str((m.get("author") or {}).get("id", "")),
                "ts": parse_ts(ts_text),
            }
            changed += 1

        if changed:
            save_db()
    if changed and recalc:
        recalculate()
    return changed


# ============================================================
#  VALUE MODEL
# ============================================================
VALUES = {}              # item -> log(value) from the latest fit
VOTE_ITEMS = set()       # items that appear in voted trades
VOTE_COUNT = {}          # item -> number of voted trades containing it


def prior(name):
    """(mean log-value, anchor strength) before any votes."""
    if name in LIST_VALUE:
        return math.log(LIST_VALUE[name]), LAM_LIST
    est = shiny_estimate(name)
    if est:
        return math.log(est), LAM_EST
    return math.log(DEFAULT_VALUE), LAM_DEFAULT


def value_of(name, theta=None):
    source = VALUES if theta is None else theta
    if name in source:
        return math.exp(source[name])
    return math.exp(prior(name)[0])


def decay(ts, now):
    if not ts:
        return 0.5
    age_days = max(0.0, (now - ts) / 86400.0)
    return 0.5 ** (age_days / HALF_LIFE_DAYS)


def build_groups(trades, now):
    """
    Group voted trades by item set. Each (author, trade) pair counts once, using the
    latest vote, so one person posting many times cannot dominate the model.
    """
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
    """
    Newton fit of log-values. A group says the offered side vs the requested side
    won w1 times and lost w0 times. The value list anchors the fit.
    """
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


def recalculate():
    """Refit all values from the trade history. Call after any change to TRADE_DB."""
    global VALUES, VOTE_ITEMS, VOTE_COUNT
    with DATA_LOCK:
        trades = list(TRADE_DB.values())
        voted = [t for t in trades if t["ups"] != t["downs"] and t["offered"] and t["requested"]]

        counts = {}
        for t in voted:
            for i in set(t["offered"]) | set(t["requested"]):
                counts[i] = counts.get(i, 0) + 1
        VOTE_COUNT = counts
        VOTE_ITEMS = set(counts)

        groups = build_groups(voted, time.time())
        items = set(LIST_VALUE) | VOTE_ITEMS
        VALUES = fit_values(groups, items, {i: prior(i) for i in items})


def backtest(split=0.8):
    """
    Fit on the older trades, then predict the winner of the newer trades.
    Returns None when there are fewer than 20 voted trades.
    """
    with DATA_LOCK:
        voted = [t for t in TRADE_DB.values()
                 if t["ups"] != t["downs"] and t["offered"] and t["requested"]]
    voted.sort(key=lambda t: t.get("ts", 0))
    if len(voted) < 20:
        return None

    cut = int(len(voted) * split)
    train, test = voted[:cut], voted[cut:]
    cut_ts = train[-1].get("ts") or time.time()

    groups = build_groups(train, cut_ts)
    items = set(LIST_VALUE) | {i for t in train for i in t["offered"] + t["requested"]}
    theta = fit_values(groups, items, {i: prior(i) for i in items})

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
    """
    Exact matches only (same item sets on the same sides, or flipped).
    One vote per author. Returns (wins, losses) for the user.
    """
    sg, st = frozenset(give), frozenset(get)
    wins = losses = 0
    seen_authors = set()
    with DATA_LOCK:
        trades = list(TRADE_DB.values())
    for t in trades:
        if t["ups"] == t["downs"]:
            continue
        o, r = frozenset(t["offered"]), frozenset(t["requested"])
        if (sg, st) == (o, r):
            flipped = False
        elif (sg, st) == (r, o):
            flipped = True
        else:
            continue
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
