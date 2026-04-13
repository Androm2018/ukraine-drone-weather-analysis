import asyncio, os, re, csv
from datetime import datetime, timezone
from pathlib import Path
from telethon import TelegramClient
from telethon.tl.types import MessageMediaPhoto

SESSION  = '/workspaces/ukraine-drone-weather-analysis/data/telegram/session'
DATA_DIR = Path('/workspaces/ukraine-drone-weather-analysis/data/telegram')
DATA_DIR.mkdir(exist_ok=True)

api_id   = int(os.environ['TELEGRAM_API_ID'])
api_hash = os.environ['TELEGRAM_API_HASH']

START_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)

# ── Russian oblasts to scan for in mod_russia messages ──
OBLASTS = [
    "belgorod","kursk","bryansk","voronezh","rostov","krasnodar",
    "crimea","sevastopol","moscow","leningrad","pskov","smolensk",
    "kaluga","tula","ryazan","tambov","saratov","samara","orenburg",
    "tatarstan","bashkortostan","engels","lipetsk","oryol","tver",
    "yaroslavl","vladimir","nizhny novgorod","kazan","ufa",
]

# ── Target categories for dronbomber ──
TARGET_KEYWORDS = {
    "oil_refinery":   ["refinery","нпз","нефтеперерабат","нефтезавод","oil refin","petroleum"],
    "fuel_depot":     ["fuel depot","fuel storage","нефтебаза","топливо","depot","tank farm"],
    "airbase":        ["airbase","air base","airfield","аэродром","авиабаза","airport","air force"],
    "ammunition":     ["ammunition","ammo","depot","склад","арсенал","arsenal","munition"],
    "power":          ["power station","power plant","электростанция","электро","substation","энергетич"],
    "military_industrial": ["factory","завод","plant","производств","military industrial","defense plant"],
    "naval":          ["port","порт","naval","fleet","флот","ship","корабл","vessel"],
    "radar_ad":       ["radar","radar","С-300","С-400","pantsir","панцирь","air defense","ПВО"],
    "command":        ["headquarters","штаб","HQ","command","управлени"],
}

DAMAGE_KEYWORDS = {
    "destroyed":  ["destroyed","уничтожен","destroyed","direct hit","прямое попадание"],
    "fire":       ["fire","пожар","burning","горит","blaze","flame"],
    "explosion":  ["explosion","взрыв","blast","detonation"],
    "damaged":    ["damaged","повреждён","hit","struck","поражён"],
    "unconfirmed":["reported","allegedly","unconfirmed","сообщается","по данным"],
}


def parse_mod_russia(text: str, date) -> dict | None:
    """Extract intercept claims from Russian MoD messages."""
    if not text:
        return None
    text_lower = text.lower()

    # Only process messages about drone/missile interceptions
    if not any(kw in text_lower for kw in [
        "уничтожен","сбит","перехвачен","пресечена","беспилотник",
        "бпла","drone","drones","intercepted","destroyed","uav"
    ]):
        return None

    # Extract drone count
    drone_count = None
    patterns = [
        r'(\d+)\s*беспилотник',
        r'(\d+)\s*бпла',
        r'(\d+)\s*uav',
        r'(\d+)\s*drone',
        r'уничтожен[оы]?\s*(\d+)',
        r'сбит[оы]?\s*(\d+)',
        r'перехвачен[оы]?\s*(\d+)',
    ]
    for pat in patterns:
        m = re.search(pat, text_lower)
        if m:
            drone_count = int(m.group(1))
            break

    # Extract oblasts mentioned
    oblasts_found = [o for o in OBLASTS if o in text_lower]

    # Determine if this is about Ukraine attacking Russia
    ua_attack = any(kw in text_lower for kw in [
        "украин","ukraine","ukrainian","киев","kyiv","всу","зсу","афу"
    ])

    if drone_count is None and not ua_attack:
        return None

    return {
        "date":        date.strftime("%Y-%m-%d"),
        "datetime":    date.strftime("%Y-%m-%d %H:%M"),
        "drone_count": drone_count or 0,
        "oblasts":     "|".join(oblasts_found),
        "ua_attack":   int(ua_attack),
        "text_snippet": text[:200].replace("\n"," "),
    }


def parse_dronbomber(text: str, date) -> dict | None:
    """Extract confirmed strike info from dronbomber messages."""
    if not text:
        return None
    text_lower = text.lower()

    # Must mention Russia or a Russian target
    if not any(kw in text_lower for kw in [
        "russia","russian","россия","росси","moscow","moscow",
        "refinery","airbase","refinery","завод","аэродром",
        "belgorod","kursk","bryansk","voronezh","rostov",
        "krasnodar","crimea","leningrad","tatarstan","samara",
        "saratov","moscow","engels","lipetsk","oryol",
    ]):
        return None

    # Target category
    target_cat = "unknown"
    for cat, keywords in TARGET_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            target_cat = cat
            break

    # Damage assessment
    damage = "unknown"
    for dmg, keywords in DAMAGE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            damage = dmg
            break

    # Extract oblast/location
    oblasts_found = [o for o in OBLASTS if o in text_lower]

    # Confidence level
    confirmed = any(kw in text_lower for kw in [
        "confirmed","подтверждён","general staff","генштаб",
        "official","официально","ukraine confirms","sbu","гур"
    ])

    return {
        "date":        date.strftime("%Y-%m-%d"),
        "datetime":    date.strftime("%Y-%m-%d %H:%M"),
        "target_cat":  target_cat,
        "damage":      damage,
        "oblasts":     "|".join(oblasts_found) if oblasts_found else "unknown",
        "confirmed":   int(confirmed),
        "text_snippet": text[:300].replace("\n"," "),
    }


async def scrape_channel(client, channel, parser_fn, output_path, label):
    print(f"\nScraping {label}...")
    rows = []
    count = 0

    async for msg in client.iter_messages(channel, reverse=True, offset_date=START_DATE):
        if msg.date < START_DATE:
            continue
        if not msg.text:
            continue
        parsed = parser_fn(msg.text, msg.date)
        if parsed:
            rows.append(parsed)
        count += 1
        if count % 500 == 0:
            print(f"  {label}: processed {count} messages, {len(rows)} matches...")

    print(f"  {label}: done — {count} total messages, {len(rows)} relevant")

    if rows:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved → {output_path}")
    else:
        print(f"  No rows extracted for {label}")

    return rows


async def main():
    print("=" * 60)
    print("  TELEGRAM DEEP STRIKE SCRAPER")
    print(f"  From: {START_DATE.date()} → present")
    print("=" * 60)

    client = TelegramClient(SESSION, api_id, api_hash)
    await client.start()

    await scrape_channel(
        client, "mod_russia",
        parse_mod_russia,
        DATA_DIR / "mod_russia_raw.csv",
        "@mod_russia"
    )

    await scrape_channel(
        client, "dronbomber",
        parse_dronbomber,
        DATA_DIR / "dronbomber_raw.csv",
        "@dronbomber"
    )

    await client.disconnect()
    print("\nDone.")


asyncio.run(main())
