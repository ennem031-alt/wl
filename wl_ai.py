"""
AA Trade Evaluator - AI layer (optional).

Claude reads natural messages and decides which tool to call. The tools run
the exact calculations in wl_core, so every number the user sees comes from
the Python code, never from the model.

Setup:
  pip install -U anthropic
  config.json:  "anthropic_api_key": "sk-ant-...",   (optional) "ai_model": "claude-haiku-5-5"
"""

import json
import math
import os
import time

import wl_core as core
import wl_memory as mem

try:
    import anthropic
except ImportError:
    anthropic = None

DEFAULT_MODEL = "claude-haiku-5-5"
MAX_HISTORY_TURNS = 20        # user+assistant pairs sent to the model
MAX_TOOL_ROUNDS = 5           # safety limit on tool calls per message
REPLY_MAX_TOKENS = 600
COOLDOWN_SECONDS = 3.0        # per-user spam protection

SYSTEM_PROMPT = """You are the assistant for AA Trade Evaluator, a Discord bot that evaluates "win or lose" (W/L) unit trades in the AA game.

Rules:
- Only help with unit trades, unit values, and this bot's features. For anything else, politely say in one sentence that you only help with AA trades.
- Never state a unit's value or a trade result from memory. Always call a tool first and use only its numbers.
- In a trade, "give" is what the user gives up and "get" is what they receive. If the user wrote "my X", X goes in "give".
- If a tool reports an unrecognized unit, say so and ask which one they meant, using the suggestions the tool returned.
- If the user asks about their previous trade, call get_last_trade.
- Reply in the same language the user writes in (English or Tagalog).
- Keep replies short, under 120 words. Use short bullet points when listing units. Do not use tables.
- Mention that values are estimates when you give a verdict."""

TOOLS = [
    {
        "name": "evaluate_trade",
        "description": (
            "Evaluate a W/L trade. 'give' lists the units the user gives up, 'get' lists the units "
            "they receive. Always use this for trade questions instead of estimating values."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "give": {"type": "array", "items": {"type": "string"},
                         "description": "Unit names the user gives up"},
                "get": {"type": "array", "items": {"type": "string"},
                        "description": "Unit names the user receives"},
            },
            "required": ["give", "get"],
        },
    },
    {
        "name": "lookup_item",
        "description": "Get the value of one unit, for example 'dio' or 'shiny sakamoto'.",
        "input_schema": {
            "type": "object",
            "properties": {"item": {"type": "string", "description": "The unit name"}},
            "required": ["item"],
        },
    },
    {
        "name": "get_last_trade",
        "description": "Return the last trade this user evaluated, so it can be discussed or reversed.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

_settings = {"key": "", "model": DEFAULT_MODEL}
_client = None
_last_call = {}


# ============================================================
#  SETUP
# ============================================================
def _load_settings():
    if not os.path.exists(core.CONFIG_FILE):
        return
    try:
        with open(core.CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        _settings["key"] = str(cfg.get("anthropic_api_key", "")).strip()
        _settings["model"] = str(cfg.get("ai_model", DEFAULT_MODEL)).strip() or DEFAULT_MODEL
    except Exception as e:
        print(f"[AI] Could not read config.json: {e}")


def init_ai():
    """Create the chat-history table and report whether AI is enabled."""
    _load_settings()
    with mem.memory_conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS ai_turns (
                         id INTEGER PRIMARY KEY AUTOINCREMENT,
                         uid TEXT NOT NULL,
                         role TEXT NOT NULL,
                         content TEXT NOT NULL,
                         ts REAL NOT NULL)""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_ai_turns_uid ON ai_turns(uid, id)")

    if enabled():
        print(f"[AI] Enabled. Model: {_settings['model']}")
    elif anthropic is None:
        print("[AI] Disabled. Install it with: pip install -U anthropic")
    else:
        print("[AI] Disabled. Add anthropic_api_key to config.json to turn it on.")


def enabled():
    return anthropic is not None and bool(_settings["key"])


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=_settings["key"])
    return _client


# ============================================================
#  CHAT HISTORY (per user)
# ============================================================
def _add_turn(uid, role, text):
    with mem.memory_conn() as c:
        c.execute("INSERT INTO ai_turns (uid, role, content, ts) VALUES (?, ?, ?, ?)",
                  (uid, role, text[:4000], time.time()))
        c.execute("""DELETE FROM ai_turns WHERE uid = ? AND id NOT IN (
                         SELECT id FROM ai_turns WHERE uid = ? ORDER BY id DESC LIMIT ?)""",
                  (uid, uid, MAX_HISTORY_TURNS * 2))


def _recent_turns(uid):
    with mem.memory_conn() as c:
        rows = c.execute("SELECT role, content FROM ai_turns WHERE uid = ? ORDER BY id DESC LIMIT ?",
                         (uid, MAX_HISTORY_TURNS * 2)).fetchall()
    turns = [{"role": role, "content": text} for role, text in reversed(rows)]
    # The API requires the conversation to start with a user message
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    return turns


def forget_turns(uid):
    with mem.memory_conn() as c:
        c.execute("DELETE FROM ai_turns WHERE uid = ?", (uid,))


# ============================================================
#  TOOL IMPLEMENTATIONS (exact math lives here)
# ============================================================
def _choice_lookup(uid):
    personal = mem.learned_map(uid)

    def lookup(key):
        hit = personal.get(key) or mem.consensus(key)
        return hit if hit and core.is_known(hit) else None
    return lookup


def _resolve(uid, names):
    lookup = _choice_lookup(uid)
    resolved, problems = [], []
    for raw in names:
        status, name, opts = core.resolve_query(str(raw), lookup)
        if status == "ok":
            resolved.append(name)
        else:
            problems.append({
                "input": raw,
                "status": status,
                "suggestions": [label for label, _ in opts],
            })
    return resolved, problems


def _tool_evaluate_trade(uid, args, ctx):
    give, p1 = _resolve(uid, args.get("give", []))
    get, p2 = _resolve(uid, args.get("get", []))
    problems = p1 + p2
    if problems:
        return {"ok": False, "problems": problems,
                "note": "Ask the user which unit they meant before evaluating."}
    if not give or not get:
        return {"ok": False, "error": "A trade needs at least one unit on each side."}

    give_vals = [core.value_of(i) for i in give]
    get_vals = [core.value_of(i) for i in get]
    give_total, get_total = sum(give_vals), sum(get_vals)
    z = math.log(get_total / give_total)
    win_prob = 1 / (1 + math.exp(-core.BETA * z))
    pct = (get_total / give_total - 1) * 100

    if abs(z) < core.FAIR_BAND:
        verdict = "FAIR"
    elif z > 0:
        verdict = "WIN"
    else:
        verdict = "LOSE"

    wins, losses = core.direct_votes(give, get)
    ctx["trade"] = (give, get)

    return {
        "ok": True,
        "give": [{"unit": core.display_name(i), "value": round(v)} for i, v in zip(give, give_vals)],
        "give_total": round(give_total),
        "get": [{"unit": core.display_name(i), "value": round(v)} for i, v in zip(get, get_vals)],
        "get_total": round(get_total),
        "verdict_for_user": verdict,
        "value_difference_percent": round(pct, 1),
        "win_probability_percent": round(win_prob * 100),
        "exact_community_votes": {"wins": wins, "losses": losses},
        "limited_data_units": sorted({core.display_name(i) for i in give + get
                                      if i not in core.LIST_VALUE
                                      and core.VOTE_COUNT.get(i, 0) < core.MIN_VOTES}),
    }


def _tool_lookup_item(uid, args, ctx):
    names, problems = _resolve(uid, [args.get("item", "")])
    if problems:
        return {"ok": False, "problems": problems,
                "note": "Ask the user which unit they meant."}
    name = names[0]
    return {
        "ok": True,
        "unit": core.display_name(name),
        "value": round(core.value_of(name)),
        "source": core.source_label(name),
        "votes_on_record": core.VOTE_COUNT.get(name, 0),
    }


def _tool_get_last_trade(uid, args, ctx):
    last = mem.get_last_trade(uid)
    if not last:
        return {"ok": False, "note": "The user has not evaluated a trade yet."}
    return {
        "ok": True,
        "give": [core.display_name(i) for i in last["give"]],
        "get": [core.display_name(i) for i in last["get"]],
        "note": "Call evaluate_trade with these units to get the numbers.",
    }


TOOL_FUNCS = {
    "evaluate_trade": _tool_evaluate_trade,
    "lookup_item": _tool_lookup_item,
    "get_last_trade": _tool_get_last_trade,
}


def _run_tool(name, args, uid, ctx):
    func = TOOL_FUNCS.get(name)
    if func is None:
        return {"ok": False, "error": f"Unknown tool: {name}"}
    try:
        return func(uid, args or {}, ctx)
    except Exception as e:
        print(f"[AI] Tool {name} failed: {e}")
        return {"ok": False, "error": "The tool failed. Tell the user to try again."}


# ============================================================
#  MAIN ENTRY POINT
# ============================================================
def _on_cooldown(uid):
    now = time.time()
    if now - _last_call.get(uid, 0) < COOLDOWN_SECONDS:
        return True
    _last_call[uid] = now
    return False


def chat(uid, text):
    """
    Answer one natural-language message. Blocking call: run it with asyncio.to_thread.
    Returns (reply_text, trade) where trade is (give, get) when a trade was evaluated, else None.
    """
    if _on_cooldown(uid):
        return "Please wait a few seconds before sending another message.", None

    ctx = {"trade": None}
    messages = _recent_turns(uid) + [{"role": "user", "content": text}]
    reply = None

    try:
        client = _get_client()
        for _ in range(MAX_TOOL_ROUNDS):
            response = client.messages.create(
                model=_settings["model"],
                max_tokens=REPLY_MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
            if response.stop_reason != "tool_use":
                reply = "".join(block.text for block in response.content
                                if block.type == "text").strip()
                break

            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    output = _run_tool(block.name, block.input, uid, ctx)
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(output, ensure_ascii=False),
                    })
            messages.append({"role": "user", "content": results})

        if not reply:
            reply = "I couldn't finish that. Please try a shorter question."
    except Exception as e:
        print(f"[AI] API error: {e}")
        return "The AI service is unavailable right now. Please try again later.", None

    _add_turn(uid, "user", text)
    _add_turn(uid, "assistant", reply)
    return reply, ctx["trade"]
