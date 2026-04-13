import asyncio, os, re, csv
from datetime import datetime, timezone
from pathlib import Path
from telethon import TelegramClient

SESSION  = '/workspaces/ukraine-drone-weather-analysis/data/telegram/session'
DATA_DIR = Path('/workspaces/ukraine-drone-weather-analysis/data/telegram')
DATA_DIR.mkdir(exist_ok=True)

api_id   = int(os.environ['TELEGRAM_API_ID'])
api_hash = os.environ['TELEGRAM_API_HASH']
START_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)

# ── Russian oblasts (Russian spelling for mod_russia) ──
RU_OBLASTS = {
    "калужской":"kaluga", "калуга":"kaluga",
    "московской":"moscow", "москов":"moscow",
    "тульской":"tula", "тула":"tula",
    "брянской":"bryansk", "брянск":"bryansk",
    "курской":"kursk", "курск":"kursk",
    "белгородской":"belgorod", "белгород":"belgorod",
    "воронежской":"voronezh", "воронеж":"voronezh",
    "орловской":"oryol", "орёл":"oryol",
    "липецкой":"lipetsk", "липецк":"lipetsk",
    "тамбовской":"tambov", "тамбов":"tambov",
    "рязанской":"ryazan", "рязань":"ryazan",
    "владимирской":"vladimir", "владимир":"vladimir",
    "ивановской":"ivanovo", "иваново":"ivanovo",
    "ярославской":"yaroslavl", "ярославль":"yaroslavl",
    "костромской":"kostroma", "кострома":"kostroma",
    "тверской":"tver", "тверь":"tver",
    "смоленской":"smolensk", "смоленск":"smolensk",
    "псковской":"pskov", "псков":"pskov",
    "ленинградской":"leningrad", "ленинград":"leningrad",
    "санкт-петербург":"spb", "петербург":"spb",
    "новгородской":"novgorod", "новгород":"novgorod",
    "ростовской":"rostov", "ростов":"rostov",
    "краснодарском":"krasnodar", "краснодар":"krasnodar",
    "ставропольском":"stavropol", "ставрополь":"stavropol",
    "саратовской":"saratov", "саратов":"saratov",
    "самарской":"samara", "самара":"samara",
    "пензенской":"penza", "пенза":"penza",
    "ульяновской":"ulyanovsk", "ульяновск":"ulyanovsk",
    "татарстан":"tatarstan", "казань":"kazan",
    "башкортостан":"bashkortostan", "уфа":"ufa",
    "оренбургской":"orenburg", "оренбург":"orenburg",
    "крым":"crimea", "севастополь":"sevastopol",
    "энгельс":"engels",
}

# ── Ukrainian oblasts/cities (Ukrainian spelling for dronbomber) ──
UA_RU_LOCATIONS = {
    "москв":"moscow", "московськ":"moscow",
    "калузьк":"kaluga", "калуга":"kaluga",
    "тверськ":"tver", "тверь":"tver", "твер":"tver",
    "брянськ":"bryansk", "брянськ":"bryansk",
    "курськ":"kursk",
    "бєлгород":"belgorod", "белгород":"belgorod",
    "вороніж":"voronezh", "воронеж":"voronezh",
    "ростов":"rostov",
    "краснодар":"krasnodar", "кубань":"krasnodar",
    "саратов":"saratov",
    "самар":"samara",
    "казань":"kazan", "татарстан":"tatarstan",
    "уфа":"ufa", "башкортостан":"bashkortostan",
    "ленінград":"leningrad", "санкт-петербург":"spb",
    "псков":"pskov",
    "смоленськ":"smolensk",
    "ліпецьк":"lipetsk", "липецьк":"lipetsk",
    "рязань":"ryazan",
    "орел":"oryol", "орьол":"oryol",
    "тамбов":"tambov",
    "пенза":"penza",
    "ульяновськ":"ulyanovsk",
    "енгельс":"engels", "engels":"engels",
    "крим":"crimea", "кримськ":"crimea",
    "севастопол":"sevastopol",
}

# ── Target categories (Ukrainian keywords) ──
UA_TARGET_CATS = {
    "oil_refinery":        ["нпз","нафтопереробн","нафтозавод","нефтезавод","нефтеперерабат","refinery"],
    "fuel_depot":          ["нафтобаза","нефтебаза","паливний склад","паливо","fuel depot","tank farm"],
    "airbase":             ["аеродром","авіабаза","аэродром","авиабаза","airbase","air base","airfield"],
    "ammunition_depot":    ["склад боєприпасів","арсенал","склад зброї","амуніці","ammunition","ammo depot"],
    "power":               ["електростанц","теплоелектр","підстанц","електропідстанц","power station","substation"],
    "military_factory":    ["військовий завод","оборонний завод","оборонне підприємств","military factory","defense plant"],
    "chemical_factory":    ["хімічний завод","хімзавод","химзавод","chemical plant","хімічн"],
    "steel_metal":         ["металург","сталевар","steel","металообробн"],
    "naval_port":          ["порт","флот","корабл","naval","судно","vessel"],
    "radar_ad":            ["пво","с-300","с-400","панцир","радар","radar","air defense"],
    "command_control":     ["штаб","командуванн","headquarters","command post"],
    "airbase_strategic":   ["ту-95","ту-22","стратегічн","strategic bomber","дальня авіація"],
}

# ── Damage keywords (Ukrainian + Russian) ──
UA_DAMAGE = {
    "destroyed":   ["знищен","уражен","зруйнован","destroyed","direct hit","прямое попадание","влучання"],
    "fire":        ["пожеж","горить","пожар","загорів","fire","burning","blaze"],
    "explosion":   ["вибух","детонац","explosion","взрыв","blast"],
    "damaged":     ["пошкоджен","damaged","повреждён","ушкоджен"],
    "confirmed":   ["підтверджен","підтвердж","генштаб","зсу підтвердж","confirmed","general staff"],
}


def parse_mod_russia_v2(text: str, date) -> dict | None:
    if not text:
        return None
    tl = text.lower()

    # STRICT FILTER: must be a daily territorial intercept report
    is_intercept = any(kw in tl for kw in [
        "над территорией",
        "над территориями",
        "пресечена попытка",
        "уничтожен",
    ]) and any(kw in tl for kw in [
        "беспилотн",
        "бпла",
        "уav",
        "дрон",
        "террористическ",
    ])

    # EXCLUDE cumulative stats ("с начала проведения специальной военной операции")
    is_cumulative = any(kw in tl for kw in [
        "с начала проведения",
        "с начала специальной",
        "за весь период",
        "всего уничтожено",
    ])

    if not is_intercept or is_cumulative:
        return None

    # Extract drone count
    drone_count = 0
    count_patterns = [
        r'уничтожен[оы]?\s+(\d+)\s+беспилотн',
        r'уничтожен[оы]?\s+(\d+)\s+бпла',
        r'(\d+)\s+беспилотн[а-я]+\s+летательн',
        r'(\d+)\s+бпла',
        r'(\d+)\s+украинск[а-я]+\s+беспилотн',
        r'пресечен[аы]\s+попытка.*?(\d+)',
        r'средствами\s+пво\s+уничтожен[оы]?\s+(\d+)',
        r'перехвачен[оы]?\s+(\d+)',
        r'сбит[оы]?\s+(\d+)',
        r'(\d+)\s+ударн[а-я]+\s+беспилотн',
    ]
    for pat in count_patterns:
        m = re.search(pat, tl)
        if m:
            drone_count = int(m.group(1))
            break

    # If still 0, try simpler — any number near key words
    if drone_count == 0:
        m = re.search(r'уничтожен[оы]?\s+(\d+)', tl)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 500:  # sanity check — exclude cumulative totals
                drone_count = n

    # Extract oblast
    oblasts_found = []
    for ru_name, en_name in RU_OBLASTS.items():
        if ru_name in tl and en_name not in oblasts_found:
            oblasts_found.append(en_name)

    return {
        "date":        date.strftime("%Y-%m-%d"),
        "datetime":    date.strftime("%Y-%m-%d %H:%M"),
        "drone_count": drone_count,
        "oblasts":     "|".join(oblasts_found) if oblasts_found else "unknown",
        "text_snippet": text[:400].replace("\n"," "),
    }


def parse_dronbomber_v2(text: str, date) -> dict | None:
    if not text:
        return None
    tl = text.lower()

    # STRICT FILTER: must be about a strike on Russian territory
    is_russia_strike = (
        any(kw in tl for kw in [
            "рф","росі","россі","росій","russia","russian",
            "територі росі","на территории","в росії",
        ]) and
        any(kw in tl for kw in [
            "удар","атак","уражен","знищен","влучання",
            "strike","hit","attack","explosion","вибух","пожеж",
            "завод","нпз","аеродром","склад","нафтобаза",
            "нафтопереробн","арсенал",
        ])
    )

    # Also catch oblast-named messages about strikes
    has_ru_location = any(loc in tl for loc in UA_RU_LOCATIONS)
    has_strike_word = any(kw in tl for kw in [
        "удар","атак","уражен","знищен","влучання","пожеж",
        "вибух","дрон","бпла","завод","нпз","аеродром",
    ])

    if not (is_russia_strike or (has_ru_location and has_strike_word)):
        return None

    # Target category
    target_cat = "unknown"
    for cat, keywords in UA_TARGET_CATS.items():
        if any(kw in tl for kw in keywords):
            target_cat = cat
            break

    # Damage
    damage = "unknown"
    for dmg, keywords in UA_DAMAGE.items():
        if any(kw in tl for kw in keywords):
            damage = dmg
            break

    # Locations
    locs = []
    for ua_name, en_name in UA_RU_LOCATIONS.items():
        if ua_name in tl and en_name not in locs:
            locs.append(en_name)

    # Confidence
    confirmed = any(kw in tl for kw in [
        "підтвердж","генштаб","зсу","офіційно","confirmed",
        "general staff","зафіксовано ураження","задокументовано",
    ])

    return {
        "date":        date.strftime("%Y-%m-%d"),
        "datetime":    date.strftime("%Y-%m-%d %H:%M"),
        "target_cat":  target_cat,
        "damage":      damage,
        "oblasts":     "|".join(locs) if locs else "unknown",
        "confirmed":   int(confirmed),
        "text_snippet": text[:500].replace("\n"," "),
    }


async def scrape_channel(client, channel, parser_fn, output_path, label):
    print(f"\nScraping {label}...")
    rows = []
    count = 0

    async for msg in client.iter_messages(
        channel, reverse=True, offset_date=START_DATE
    ):
        if msg.date < START_DATE:
            continue
        if not msg.text:
            continue
        parsed = parser_fn(msg.text, msg.date)
        if parsed:
            rows.append(parsed)
        count += 1
        if count % 1000 == 0:
            print(f"  {label}: {count} messages, {len(rows)} matches...")

    print(f"  {label}: done — {count} total, {len(rows)} matched")

    if rows:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved → {output_path}")

    return rows


async def main():
    print("=" * 60)
    print("  TELEGRAM SCRAPER v2 — FIXED PARSERS")
    print(f"  From: {START_DATE.date()} → present")
    print("=" * 60)

    client = TelegramClient(SESSION, api_id, api_hash)
    await client.start()

    await scrape_channel(
        client, "mod_russia",
        parse_mod_russia_v2,
        DATA_DIR / "mod_russia_v2.csv",
        "@mod_russia"
    )

    await scrape_channel(
        client, "dronbomber",
        parse_dronbomber_v2,
        DATA_DIR / "dronbomber_v2.csv",
        "@dronbomber"
    )

    await client.disconnect()
    print("\nDone.")


asyncio.run(main())
