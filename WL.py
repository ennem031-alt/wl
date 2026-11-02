"""
AA Trade Evaluator - Discord interface (v3).

Run:  python WL.py

Logic lives in wl_core.py (parsing, value list, trade history, value model)
and wl_memory.py (per-user memory and crowd consensus). wl_ai.py is optional.

How updates work now:
  - New W/L posts and 👍/👎 reactions arrive through the Discord gateway in real
    time. No polling is needed, which means fewer API calls.
  - Every 10 minutes, one request refreshes the latest 100 messages. This keeps
    reaction counts accurate after a restart.
  - Changes are saved in batches (every 20 seconds if anything changed).
"""

import asyncio
import math
import os
import sys
import time
from datetime import datetime, timezone

import aiohttp
import discord
from aiohttp import web
from discord.ext import commands, tasks

import wl_core as core
import wl_memory as mem
import wl_ai as ai

# ============================================================
#  DISPLAY SETTINGS
# ============================================================
COLOR_WIN = 0x2ECC71
COLOR_LOSE = 0xE74C3C
COLOR_FAIR = 0xF1C40F
COLOR_WARN = 0xF39C12
COLOR_INFO = 0x3498DB
COLOR_NEUTRAL = 0x95A5A6

GREETINGS = {"hi", "hello", "hey", "yo", "hi bot", "hello bot", "good morning",
             "good afternoon", "good evening", "kumusta", "musta"}
SWAP_WORDS = {"swap", "reverse", "flip", "baliktad", "baligtad"}
VALUES_PAGE_SIZE = 20

# key: (unicode fallback, custom emoji ID)
# A custom emoji is used only if the bot can see it (it must share a server with it).
EMOJI_DEF = {
    "exclaim":  ("⚠️", 1558495136940425317),
    "offline":  ("🔴", 1558495227893776434),
    "online":   ("🟢", 1558495192225554583),
    "arrow":    ("➡️", 1558501268803162112),
    "verified": ("✅", 1558495073895718983),
    "loading":  ("⏳", 1558495014936514642),
    "trade":    ("📊", 1558793841002160258),
}

# ============================================================
#  BOT
# ============================================================
intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True
intents.guild_messages = True
intents.guild_reactions = True
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

SCAN_STATE = {"running": False, "stop": False}
AUTH_STATE = {"failed": False}
HEALTH_STATE = {"started": False}


def emo(key):
    """Custom emoji markup if the bot can see it, otherwise the unicode fallback."""
    fallback, emoji_id = EMOJI_DEF[key]
    found = bot.get_emoji(emoji_id)
    if found is not None and (not hasattr(found, "is_usable") or found.is_usable()):
        return str(found)
    return fallback


# ============================================================
#  FORMAT HELPERS
# ============================================================
def fmt_value(v):
    return f"{v:,.0f}" if v >= 10 else f"{v:.1f}"


def fmt_date(ts):
    return datetime.fromtimestamp(ts).strftime("%b %d, %Y") if ts else "N/A"


def fmt_duration(sec):
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def confidence(items):
    score = min(core.vote_count_of(i) + (6 if i in core.LIST_VALUE else 0) for i in items)
    if score < core.MIN_VOTES:
        return f"{emo('offline')} Low"
    if score < 10:
        return "🟡 Medium"
    return f"{emo('online')} High"


def base_embed(title, color, description=None):
    embed = discord.Embed(title=title, color=color, description=description)
    embed.timestamp = datetime.now(timezone.utc)
    embed.set_footer(text="AA Trade Evaluator  •  Values are estimates, not guarantees")
    return embed


# ============================================================
#  EMBEDS
# ============================================================
def usage_embed():
    desc = (
        "**Evaluate a trade**\n"
        "`is [item A] for [item B] win or lose`\n"
        "Put `my` before the unit you are giving. It can be on either side:\n"
        "`is Gohan for my Goku`\n\n"
        "**Units with stats**\n"
        "`is dark mage with sss damage and sss range for shiny cream`\n\n"
        "**Look up one item**\n"
        "`what is shiny sakamoto`  or  `ano value ni dio`\n\n"
        "**Other**\n"
        "`swap` reverses your last trade\n\n"
        "**Commands**\n"
        "`!help`  •  `!values`  •  `!backtest`  •  `!start scanning`  •  `!stop scanning`\n"
        "`!deepscan [number]`  •  `!setup`  •  `!forget` (DM only)"
    )
    return base_embed(f"{emo('trade')} How to use", COLOR_INFO, desc)


def greeting_embed():
    desc = (
        "I evaluate W/L trades and look up unit values.\n\n"
        "**Try this:**\n"
        "`is my Goku for Gohan win or lose`\n\n"
        "**Or look up one unit:**\n"
        "`what is shiny sakamoto`\n\n"
        "Type `!help` for more."
    )
    return base_embed(f"{emo('trade')} Hello!", COLOR_INFO, desc)


def unknown_embed(unknown):
    lines = []
    for _, q, status, opts in unknown:
        if status == "ambiguous":
            lines.append(f"`{q}` matches more than one unit. Please choose one below.")
        elif opts:
            names = ", ".join(f"`{label}`" for label, _ in opts)
            lines.append(f"`{q}` is not recognized. Did you mean {names}?")
        else:
            lines.append(f"`{q}` has no data in the value list or channel history.")
    embed = base_embed(f"{emo('exclaim')} Unrecognized item", COLOR_WARN, "\n".join(lines))
    if any(opts for _, _, _, opts in unknown):
        embed.add_field(name="Next step",
                        value="Select the correct unit below. Your choice is remembered for next time.",
                        inline=False)
    return embed


def unpriced_embed(items):
    names = ", ".join(f"`{core.display_name(i)}`" for i in items)
    return base_embed(
        f"{emo('exclaim')} Cannot price this trade yet", COLOR_WARN,
        f"No price was found for: {names}\n\n"
        "These units are not in `value_list.txt` and have no community votes yet, so any number "
        "would be a guess. Add the unit to the value list, or wait until trades with it are posted "
        "and voted on in the channel.")


def items_embed(items, uid=None):
    lines, total, priced_all = [], 0.0, True
    for i in items:
        if core.priced(i):
            v = core.value_of(i)
            total += v
            lines.append(f"• `{core.display_name(i)}`  —  **{fmt_value(v)}**  _({core.source_label(i)})_")
        else:
            priced_all = False
            lines.append(f"• `{core.display_name(i)}`  —  _no price yet_")
    embed = base_embed(f"{emo('trade')} Item Value", COLOR_INFO, "\n".join(lines))
    if len(items) > 1 and priced_all:
        embed.add_field(name="Combined value", value=f"**{fmt_value(total)}**", inline=True)
    votes = min(core.vote_count_of(i) for i in items)
    embed.add_field(name="Votes on record", value=str(votes), inline=True)
    embed.add_field(name="Confidence", value=confidence(items), inline=True)
    if uid:
        top = mem.top_items(uid, 3)
        if top:
            embed.add_field(name="Your frequent lookups",
                            value=", ".join(f"`{core.display_name(i)}` ({c}×)" for i, c in top),
                            inline=False)
    embed.add_field(name="Compare a trade", value="`is [A] for [B] win or lose`", inline=False)
    return embed


def trade_embed(give, get):
    missing = [i for i in give + get if not core.priced(i)]
    if missing:
        return unpriced_embed(missing)

    give_values = [core.value_of(i) for i in give]
    get_values = [core.value_of(i) for i in get]
    give_total, get_total = sum(give_values), sum(get_values)

    z = math.log(get_total / give_total)
    p_win = 1 / (1 + math.exp(-core.BETA * z))
    pct = (get_total / give_total - 1) * 100

    if abs(z) < core.FAIR_BAND:
        verdict = "FAIR"
    elif z > 0:
        verdict = "WIN"
    else:
        verdict = "LOSE"

    wins, losses = core.direct_votes(give, get)
    posts = wins + losses
    vote_verdict = None
    if posts >= 2 and wins != losses:
        vote_verdict = "WIN" if wins > losses else "LOSE"
    conflict = vote_verdict is not None and vote_verdict != verdict

    color = {"WIN": COLOR_WIN, "LOSE": COLOR_LOSE, "FAIR": COLOR_FAIR}[verdict]
    if conflict:
        color = COLOR_WARN

    if verdict == "FAIR":
        desc = f"**FAIR**: the two sides are within {core.FAIR_BAND * 100:.0f}% of each other in value."
    elif verdict == "WIN":
        desc = f"**WIN**: you receive about **{abs(pct):.0f}% more** value than you give."
    else:
        desc = f"**LOSE**: you receive about **{abs(pct):.0f}% less** value than you give."

    if conflict:
        desc += (
            f"\n\n{emo('exclaim')} **Data conflict**: channel votes favor **{vote_verdict}**, "
            f"while the value model favors **{verdict}**. Check recent posts before trading."
        )

    embed = base_embed(f"{emo('trade')} Trade Evaluation", color, desc)

    def side_lines(items, values):
        return "\n".join(f"• `{core.display_name(i)}`  —  {fmt_value(v)}"
                         for i, v in zip(items, values))

    embed.add_field(name="You give",
                    value=f"{side_lines(give, give_values)}\n**Total: {fmt_value(give_total)}**",
                    inline=True)
    embed.add_field(name="You receive",
                    value=f"{side_lines(get, get_values)}\n**Total: {fmt_value(get_total)}**",
                    inline=True)
    embed.add_field(name="Win probability", value=f"**{p_win * 100:.0f}%**", inline=True)
    embed.add_field(name="Confidence", value=confidence(give + get), inline=True)

    if posts:
        vote_text = f"{wins} win • {losses} loss across {posts} post{'s' if posts != 1 else ''}"
    else:
        vote_text = "No exact matches in channel history"
    embed.add_field(name="Community votes", value=vote_text, inline=False)

    notes = []
    if any(i in core.COMPOSITE for i in give + get):
        notes.append("Units with stats are priced as the unit plus each stat (estimate).")
    estimated = sorted({core.display_name(i) for i in give + get
                        if core.shiny_estimate(i) is not None and i not in core.VOTE_ITEMS})
    limited = sorted({core.display_name(i) for i in give + get
                      if i not in core.LIST_VALUE and core.vote_count_of(i) < core.MIN_VOTES})
    if estimated:
        notes.append("Shiny estimate from the non-shiny price: " + ", ".join(estimated))
    if limited:
        notes.append("Limited data: " + ", ".join(limited))
    if notes:
        embed.add_field(name="Notes", value="\n".join(notes), inline=False)
    return embed


# ============================================================
#  ANSWER BUILDER
# ============================================================
def make_choice_lookup(uid):
    """Personal choices first, then the crowd consensus. Only returns units that exist."""
    personal = mem.learned_map(uid)

    def lookup(key):
        hit = personal.get(key) or mem.consensus(key)
        if hit and core.is_known(hit):
            return hit
        return None
    return lookup


def fully_resolved(uid, text):
    """True when the calculator can answer on its own (no AI needed)."""
    plain = core.clean(text.lower()).strip(" !.?")
    if plain in SWAP_WORDS:
        return True
    parsed = core.split_sides(text)
    if not parsed:
        return False
    lookup = make_choice_lookup(uid)
    return all(core.resolve_query(q, lookup)[0] == "ok" for q in parsed[0] + parsed[1])


def build_answer(uid, text, log=True):
    """Return (embed, view_or_None) for one user message."""
    if not core.LIST_VALUE and not core.TRADE_DB:
        return base_embed(f"{emo('exclaim')} Database is empty", COLOR_WARN,
                          "Run `!start scanning` in the W/L channel to build the database."), None

    plain = core.clean(text.lower()).strip(" !.?")
    if plain in SWAP_WORDS:
        last = mem.get_last_trade(uid)
        if not last:
            return base_embed(f"{emo('exclaim')} Nothing to swap", COLOR_WARN,
                              "Evaluate a trade first, then send `swap` to see it reversed."), None
        return trade_embed(last["get"], last["give"]), None

    parsed = core.split_sides(text)
    if not parsed:
        return usage_embed(), None

    give_raw, get_raw = parsed
    lookup = make_choice_lookup(uid)
    give, get, unknown = [], [], []
    for raw, bucket in ((give_raw, give), (get_raw, get)):
        for q in raw:
            status, name, opts = core.resolve_query(q, lookup)
            if status == "ok":
                bucket.append(name)
            else:
                unknown.append((core.learn_key(q), q, status, opts))

    if unknown:
        first = unknown[0]
        view = ChoiceView(uid, first[0], text, first[3]) if first[3] else None
        return unknown_embed(unknown), view

    if not get:
        if log:
            mem.log_items(uid, give)
        return items_embed(give, uid), None

    if log:
        mem.log_items(uid, give + get)
        mem.set_last_trade(uid, give, get)
    return trade_embed(give, get), None


class ChoiceView(discord.ui.View):
    """Buttons for picking the correct unit. The choice is remembered for this user and the crowd."""

    def __init__(self, uid, query_key, text, options):
        super().__init__(timeout=300)
        self.uid = uid
        self.query_key = query_key
        self.text = text
        for label, canonical in options[:4]:
            button = discord.ui.Button(label=label[:80], style=discord.ButtonStyle.secondary)
            button.callback = self._make_callback(canonical)
            self.add_item(button)

    def _make_callback(self, canonical):
        async def callback(interaction: discord.Interaction):
            if str(interaction.user.id) != self.uid:
                return await interaction.response.send_message(
                    "Only the person who asked can make this choice.", ephemeral=True)
            await asyncio.to_thread(mem.learn, self.uid, self.query_key, canonical)
            embed, view = await asyncio.to_thread(build_answer, self.uid, self.text, True)
            await interaction.response.edit_message(embed=embed, view=view)
        return callback


# ============================================================
#  CHANNEL HISTORY (uses HISTORY_AUTH - can be USER_TOKEN or BOT_TOKEN)
# ============================================================
async def fetch_messages(limit=100, before_id=None):
    headers = {"Authorization": core.HISTORY_AUTH}
    out, last = [], before_id
    async with aiohttp.ClientSession(headers=headers) as session:
        while len(out) < limit and not AUTH_STATE["failed"]:
            url = f"https://discord.com/api/v9/channels/{core.VALUE_CHANNEL_ID}/messages?limit=100"
            if last:
                url += f"&before={last}"
            try:
                async with session.get(url, timeout=15) as r:
                    if r.status == 200:
                        batch = await r.json()
                        if not batch:
                            break
                        out.extend(batch)
                        last = batch[-1]["id"]
                        await asyncio.sleep(0.4)
                    elif r.status == 429:
                        await asyncio.sleep(float(r.headers.get("Retry-After", 5)))
                    elif r.status in (401, 403):
                        AUTH_STATE["failed"] = True
                        # Binago ang log para malaman kung user token o bot token ang may problema
                        print(f"[ERROR] Discord returned {r.status}. Check your HISTORY_AUTH (BOT_TOKEN or USER_TOKEN) "
                              "and the permissions in the W/L channel. Background updates stopped.")
                        break
                    else:
                        break
            except Exception:
                break
    return out[:limit]


def live_message(m):
    """Convert a gateway message into the same shape the history API returns."""
    return {
        "id": str(m.id),
        "content": m.content,
        "timestamp": m.created_at.isoformat(),
        "author": {"id": str(m.author.id)},
    }


# ============================================================
#  SCANNING
# ============================================================
def scan_embed(st, running, error=None):
    if error:
        return base_embed(f"{emo('offline')} Scan failed", COLOR_NEUTRAL, f"```{error[:300]}```")

    if running:
        embed = base_embed(f"{emo('loading')} Scanning channel history", COLOR_INFO,
                           "Reading the channel. Use `!stop scanning` to stop early; progress is kept.")
    elif st.get("stopped"):
        embed = base_embed(f"{emo('exclaim')} Scan stopped", COLOR_WARN,
                           "Stopped on request. Everything read so far is saved.")
    else:
        embed = base_embed(f"{emo('verified')} Scan complete", COLOR_WIN,
                           "The database is up to date. Live updates are active.")

    elapsed = time.monotonic() - st["started"]
    embed.add_field(name="Messages scanned", value=f"{st['scanned']:,}", inline=True)
    embed.add_field(name="W/L posts found", value=f"{st['posts']:,}", inline=True)
    embed.add_field(name="Records saved", value=f"{st['saved']:,}", inline=True)
    embed.add_field(name=f"{emo('arrow')} Newest", value=fmt_date(st.get("newest")), inline=True)
    embed.add_field(name=f"{emo('arrow')} Oldest reached", value=fmt_date(st.get("oldest")), inline=True)
    embed.add_field(name="Elapsed", value=fmt_duration(elapsed), inline=True)

    if not running:
        voted = sum(1 for t in core.TRADE_DB.values() if t["ups"] != t["downs"])
        embed.add_field(name="Trades in database", value=f"{len(core.TRADE_DB):,}", inline=True)
        embed.add_field(name="Voted trades", value=f"{voted:,}", inline=True)
        embed.add_field(name="Units priced", value=f"{len(core.VALUES):,}", inline=True)
    return embed


async def safe_edit(message, embed):
    try:
        await message.edit(embed=embed)
    except discord.HTTPException as e:
        print(f"[WARN] Could not update status message: {e}")


def channel_allowed(ctx):
    return isinstance(ctx.channel, discord.DMChannel) or ctx.channel.id == core.VALUE_CHANNEL_ID


async def run_scan(ctx, limit=None):
    """Scan channel history (all of it, or up to `limit` messages). Can be stopped."""
    if SCAN_STATE["running"]:
        return await ctx.reply("A scan is already running. Use `!stop scanning` to stop it.")

    SCAN_STATE["running"] = True
    SCAN_STATE["stop"] = False
    st = {"scanned": 0, "posts": 0, "saved": 0, "newest": None, "oldest": None,
          "started": time.monotonic(), "stopped": False}
    status = await ctx.reply(embed=scan_embed(st, running=True))
    print("[SCAN] Started")

    last_id, buffer, last_edit, since_flush = None, [], 0.0, 0
    try:
        while limit is None or st["scanned"] < limit:
            if SCAN_STATE["stop"]:
                st["stopped"] = True
                break
            if AUTH_STATE["failed"]:
                break

            want = 100 if limit is None else min(100, limit - st["scanned"])
            batch = await fetch_messages(limit=want, before_id=last_id)
            if not batch:
                await asyncio.sleep(3)
                batch = await fetch_messages(limit=want, before_id=last_id)
                if not batch:
                    break

            if st["newest"] is None:
                st["newest"] = core.parse_ts(batch[0].get("timestamp", "")) or None
            st["oldest"] = core.parse_ts(batch[-1].get("timestamp", "")) or st["oldest"]
            st["scanned"] += len(batch)
            st["posts"] += sum(1 for m in batch if m.get("content", "").lower().startswith("!wl"))
            last_id = batch[-1]["id"]
            buffer.extend(batch)

            if len(buffer) >= 500:
                st["saved"] += await asyncio.to_thread(core.ingest, buffer, False)
                since_flush += len(buffer)
                buffer = []
                if since_flush >= 20000:
                    await asyncio.to_thread(core.flush)
                    since_flush = 0

            if time.monotonic() - last_edit >= 3:
                last_edit = time.monotonic()
                await safe_edit(status, scan_embed(st, running=True))

        if buffer:
            st["saved"] += await asyncio.to_thread(core.ingest, buffer, False)
        await asyncio.to_thread(core.flush)
        await safe_edit(status, scan_embed(st, running=False))
        print(f"[SCAN] Finished. {len(core.TRADE_DB)} trades in database.")
    except Exception as e:
        print(f"[SCAN] Error: {e}")
        await safe_edit(status, scan_embed(st, running=False, error=str(e)))
    finally:
        SCAN_STATE["running"] = False
        SCAN_STATE["stop"] = False


# ============================================================
#  BACKGROUND TASKS
# ============================================================
@tasks.loop(seconds=20)
async def flush_loop():
    """Save and refit in batches when something changed. Replaces per-message saving."""
    if core.is_dirty():
        await asyncio.to_thread(core.flush)


@tasks.loop(minutes=10)
async def refresh_loop():
    """One request every 10 minutes: refreshes reaction counts of the latest messages."""
    if AUTH_STATE["failed"]:
        refresh_loop.cancel()
        return
    messages = await fetch_messages(limit=100)
    if messages:
        await asyncio.to_thread(core.ingest, messages, False)


# ============================================================
#  HEALTH SERVER (for Render web service)
# ============================================================
async def _health(request):
    return web.Response(text="AA Trade Evaluator is running")


async def start_health_server():
    """Listen on Render's PORT so the web service is recognized as running."""
    if HEALTH_STATE["started"]:
        return
    port = int(os.environ.get("PORT", "10000"))
    app = web.Application()
    app.router.add_get("/", _health)
    app.router.add_get("/health", _health)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", port).start()
    HEALTH_STATE["started"] = True
    print(f"[HEALTH] Listening on port {port}")


# ============================================================
#  VALUES LIST
# ============================================================
def values_pages(query=""):
    """Build the pages for !values, sorted from highest to lowest value."""
    q = query.strip().lower()
    items = [(v, c) for c, v in core.LIST_VALUE.items()]
    if q:
        items = [(v, c) for v, c in items if q in core.display_name(c).lower()]
    items.sort(key=lambda x: -x[0])

    if not items:
        return [base_embed(f"{emo('exclaim')} No units found", COLOR_WARN,
                           f"Nothing in the value list matches `{query.strip()}`.")]

    total_pages = (len(items) + VALUES_PAGE_SIZE - 1) // VALUES_PAGE_SIZE
    pages = []
    for p in range(total_pages):
        chunk = items[p * VALUES_PAGE_SIZE:(p + 1) * VALUES_PAGE_SIZE]
        lines = [
            f"`{p * VALUES_PAGE_SIZE + i + 1:>3}.` **{core.display_name(c)}**  —  {fmt_value(v)}"
            for i, (v, c) in enumerate(chunk)
        ]
        embed = base_embed(f"{emo('trade')} Value List", COLOR_INFO, "\n".join(lines))
        embed.set_footer(text=f"Page {p + 1} of {total_pages}  •  {len(items)} units  •  "
                              "Values are estimates, not guarantees")
        pages.append(embed)
    return pages


class ValuesView(discord.ui.View):
    """Previous / next buttons for the !values pages."""

    def __init__(self, pages):
        super().__init__(timeout=300)
        self.pages = pages
        self.index = 0
        self._sync_buttons()

    def _sync_buttons(self):
        self.prev_page.disabled = self.index == 0
        self.next_page.disabled = self.index >= len(self.pages) - 1

    @discord.ui.button(label="◀ Previous", style=discord.ButtonStyle.secondary)
    async def prev_page(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.index = max(0, self.index - 1)
        self._sync_buttons()
        await interaction.response.edit_message(embed=self.pages[self.index], view=self)

    @discord.ui.button(label="Next ▶", style=discord.ButtonStyle.secondary)
    async def next_page(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.index = min(len(self.pages) - 1, self.index + 1)
        self._sync_buttons()
        await interaction.response.edit_message(embed=self.pages[self.index], view=self)


# ============================================================
#  COMMANDS
# ============================================================
@bot.command(name="help")
async def help_cmd(ctx):
    await ctx.reply(embed=usage_embed())


@bot.command(name="start")
async def start_cmd(ctx, *, what: str = ""):
    if what.strip().lower() != "scanning":
        return
    if not channel_allowed(ctx):
        return await ctx.reply("Run this command in the W/L channel or in DM.")
    await run_scan(ctx, limit=None)


@bot.command(name="stop")
async def stop_cmd(ctx, *, what: str = ""):
    if what.strip().lower() != "scanning":
        return await ctx.reply("Usage: `!stop scanning`")
    if not SCAN_STATE["running"]:
        return await ctx.reply(embed=base_embed(
            f"{emo('exclaim')} Nothing to stop", COLOR_WARN, "No scan is running right now."))
    SCAN_STATE["stop"] = True
    await ctx.reply(embed=base_embed(
        f"{emo('loading')} Stopping scan", COLOR_INFO,
        "The scan will stop after the current batch. Progress is saved."))


@bot.command(name="deepscan")
async def deepscan_cmd(ctx, limit: int = 5000):
    if not channel_allowed(ctx):
        return await ctx.reply("Run this command in the W/L channel or in DM.")
    limit = max(100, min(limit, 50000))
    await run_scan(ctx, limit=limit)


@bot.command(name="values")
async def values_cmd(ctx, *, query: str = ""):
    """!values            all units, highest value first
       !values shiny      only units whose name contains 'shiny'
    """
    if not core.LIST_VALUE:
        return await ctx.reply(embed=base_embed(
            f"{emo('exclaim')} Value list is empty", COLOR_WARN,
            "Check that value_list.txt is in the bot folder."))
    pages = values_pages(query)
    view = ValuesView(pages) if len(pages) > 1 else None
    await ctx.reply(embed=pages[0], view=view)


@bot.command(name="backtest")
async def backtest_cmd(ctx):
    if not channel_allowed(ctx):
        return await ctx.reply("Run this command in the W/L channel or in DM.")
    result = await asyncio.to_thread(core.backtest)
    if result is None:
        return await ctx.reply(embed=base_embed(
            f"{emo('exclaim')} Not enough data", COLOR_WARN,
            "At least 20 voted trades are needed to run a backtest."))

    beat = result["accuracy"] > result["baseline"]
    embed = base_embed(
        f"{emo('trade')} Model Backtest", COLOR_WIN if beat else COLOR_WARN,
        "Accuracy is measured on the **newest 20%** of voted trades. The model was trained only on "
        "the older 80%, so this is an honest estimate of how it performs on new trades.")
    embed.add_field(name="Trades tested", value=f"{result['tested']:,}", inline=True)
    embed.add_field(name="Trained on", value=f"{result['trained_on']:,}", inline=True)
    embed.add_field(name="Accuracy", value=f"**{result['accuracy'] * 100:.1f}%**", inline=True)
    embed.add_field(name="Baseline (always pick the majority)",
                    value=f"{result['baseline'] * 100:.1f}%", inline=True)
    embed.add_field(name="Log loss (lower is better)", value=f"{result['logloss']:.3f}", inline=True)
    if not beat:
        embed.add_field(name=f"{emo('exclaim')} Warning",
                        value="The model does not beat the baseline yet. Treat its verdicts with caution.",
                        inline=False)
    await ctx.reply(embed=embed)


@bot.command(name="setup")
async def setup_cmd(ctx):
    """Read-only status check. Secrets are never accepted or shown in chat."""
    checks = [
        ("Bot token is set", bool(core.BOT_TOKEN)),
        ("History Token is set", bool(core.HISTORY_AUTH)),
        ("Value channel ID is set", core.VALUE_CHANNEL_ID != 0),
        ("Value list loaded", bool(core.LIST_VALUE)),
        ("Trade database has data", bool(core.TRADE_DB)),
        ("Discord access is working", not AUTH_STATE["failed"]),
        ("Live updates (new posts and votes)", bot.is_ready()),
        ("AI assistant (optional)", ai.enabled()),
    ]
    lines = [f"{emo('verified') if ok else emo('offline')} {label}" for label, ok in checks]
    lines.append("")
    lines.append("To change settings, edit the environment variables in the Render dashboard. "
                 "Never paste tokens in Discord.")
    await ctx.reply(embed=base_embed(f"{emo('trade')} Setup status", COLOR_INFO, "\n".join(lines)))


@bot.command(name="forget")
async def forget_cmd(ctx):
    if not isinstance(ctx.channel, discord.DMChannel):
        return await ctx.reply("Use `!forget` in DM so your data is cleared privately.")
    await asyncio.to_thread(mem.forget, str(ctx.author.id))
    await asyncio.to_thread(ai.forget_turns, str(ctx.author.id))
    await ctx.reply(embed=base_embed(
        f"{emo('verified')} Your data was cleared", COLOR_WIN,
        "Your message history, lookups, saved unit choices, and last trade were deleted."))


# ============================================================
#  EVENTS
# ============================================================
@bot.event
async def on_ready():
    print(f"[READY] Logged in as {bot.user}")
    await start_health_server()
    await asyncio.to_thread(mem.init)
    await asyncio.to_thread(ai.init_ai)
    await asyncio.to_thread(core.load_value_list)
    await asyncio.to_thread(core.load_db)
    await asyncio.to_thread(core.recalculate)
    print(f"[READY] {len(core.TRADE_DB)} trades | {len(core.VALUES)} units priced")
    print("[LIVE] Listening for new W/L posts and 👍/👎 reactions.")
    if not flush_loop.is_running():
        flush_loop.start()
    if not refresh_loop.is_running():
        refresh_loop.start()


@bot.event
async def on_message(message):
    if message.author.bot:
        return

    # Channel messages: live capture of new W/L posts, no polling needed
    if message.guild is not None:
        if message.channel.id == core.VALUE_CHANNEL_ID and message.content.lower().startswith("!wl"):
            await asyncio.to_thread(core.ingest, [live_message(message)], False)
            return
        if message.content.startswith(bot.command_prefix):
            return await bot.process_commands(message)
        return

    # Direct messages
    if message.content.startswith(bot.command_prefix):
        return await bot.process_commands(message)

    uid = str(message.author.id)
    text = message.content.strip()
    if not text:
        return
    await asyncio.to_thread(mem.log_message, uid, text)

    if text.lower().strip(" !.?") in GREETINGS:
        return await message.reply(embed=greeting_embed())

    if ai.enabled() and not fully_resolved(uid, text):
        reply, trade = await asyncio.to_thread(ai.chat, uid, text)
        embed = trade_embed(*trade) if trade else None
        return await message.reply(content=reply[:1900], embed=embed)

    # Show a placeholder right away, then replace it with the answer
    calc = await message.reply(embed=base_embed(
        f"{emo('loading')} Calculating...", COLOR_INFO,
        "Checking the value list and trade history."))
    try:
        embed, view = await asyncio.to_thread(build_answer, uid, text, True)
    except Exception as e:
        print(f"[ERROR] build_answer failed: {e}")
        embed = base_embed(f"{emo('exclaim')} Something went wrong", COLOR_WARN,
                           "Please try again in a moment.")
        view = None
    await calc.edit(embed=embed, view=view)


@bot.event
async def on_raw_reaction_add(payload):
    if payload.channel_id != core.VALUE_CHANNEL_ID:
        return
    if bot.user and payload.user_id == bot.user.id:
        return
    await asyncio.to_thread(core.bump_reaction, str(payload.message_id), payload.emoji.name, 1)


@bot.event
async def on_raw_reaction_remove(payload):
    if payload.channel_id != core.VALUE_CHANNEL_ID:
        return
    if bot.user and payload.user_id == bot.user.id:
        return
    await asyncio.to_thread(core.bump_reaction, str(payload.message_id), payload.emoji.name, -1)


@bot.event
async def on_raw_message_delete(payload):
    if payload.channel_id != core.VALUE_CHANNEL_ID:
        return
    await asyncio.to_thread(core.remove_trade, str(payload.message_id))


# ============================================================
#  ENTRY POINT
# ============================================================
if __name__ == "__main__":
    core.load_config()
    if not core.BOT_TOKEN:
        sys.exit("[FATAL] BOT_TOKEN is missing.")
    bot.run(core.BOT_TOKEN)
