import os
import base64
import json
import re
import requests
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from openai import OpenAI

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

PARK_FACTORS = {
    "coors field":        {"run_factor": 1.35},
    "wrigley field":      {"run_factor": 1.12},
    "fenway park":        {"run_factor": 1.08},
    "great american":     {"run_factor": 1.18},
    "oracle park":        {"run_factor": 0.88},
    "petco park":         {"run_factor": 0.90},
    "sutter health park": {"run_factor": 1.20},
}

FUSSBALL_SYSTEM_PROMPT = """Du bist ein datenbasierter Fussball-Wettanalyst. Analysiere die Matchups im Screenshot nach streng quantitativen Kriterien und erstelle einen Wettschein. Antworte ausschliesslich auf Deutsch.

ANALYSE-PROZESS:
1. IDENTIFIKATION: Liga + Heim/Auswaerts + Anpfiffzeit (CET)
2. MATCHUP-ANALYSE: Taktische Staerken/Schwaechen, xG-Formtrend, Value-Traps (Heavy-Favorite Quote <1.80 aber xG-Offensiv <1.5 = ausschliessen)
3. KONVERGENZ: Jeder Pick benoetigt >=3 unabhaengige Faktoren. <=2 Faktoren = sofortiger Ausschluss.

ZWINGENDE REGELN:
- Keine Picks mit Quote < 1.69
- Favoriten-Wette NUR wenn xG-for >=1.8 UND xGA-against <=1.0 gleichzeitig
- Max. 1 Wette pro Matchup
- ERLAUBT: 1X2, Europaeisches Handicap (max +-1), BTTS Ja, O/U 2.5, BTTS Ja + Over 2.5, BTTS Ja 1.HZ
- VERBOTEN: Karten, Ecken, Asian Handicap, Spieler-Props
- Jeder Pick hat Hauptwette + Alternativwette:
  REGEL A: Hauptwette = 1X2/Handicap -> Alternativwette aus {O/U 2.5, BTTS Ja, BTTS+O2.5, BTTS 1.HZ}
  REGEL B: Hauptwette = {O/U 2.5, BTTS, ...} -> Alternativwette = 1X2 oder Handicap

OUTPUT: Gib NUR valides JSON zurueck, kein Markdown, keine Erklaerung.
Format:
{
  "picks": [
    {
      "startzeit": "HH:MM",
      "spiel": "Heim vs Gast",
      "liga": "Liga-Name",
      "hauptwette": "z.B. Heim 1 @ 1.85",
      "hauptwette_markt": "1X2",
      "hauptwette_quote": 1.85,
      "alternativwette": "z.B. BTTS Ja @ 1.72",
      "alternativwette_markt": "BTTS",
      "alternativwette_quote": 1.72,
      "begruendung": "Kurze Faktoren-Begruendung",
      "faktoren_count": 3
    }
  ],
  "validierung": {
    "konvergenz": "...",
    "korrelation": "...",
    "markt_allokation": "...",
    "edge": "..."
  }
}"""

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY nicht gesetzt.")
    return OpenAI(api_key=api_key)

# ── FUSSBALL ENGINE ───────────────────────────────────────────────────────────

def analyze_fussball(image_bytes: bytes) -> dict:
    client = get_openai_client()
    b64 = base64.b64encode(image_bytes).decode("ascii")
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": FUSSBALL_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": "Analysiere alle sichtbaren Matchups im Screenshot und erstelle den Wettschein als JSON."}
            ]}
        ],
        max_tokens=2500
    )
    raw = res.choices[0].message.content.strip()
    raw = raw.encode("ascii", errors="replace").decode("ascii")
    raw = re.sub(r"```[a-z]*", "", raw).strip("`").strip()
    try:
        data = json.loads(raw)
        picks = data.get("picks", [])
        # Quote-Filter nochmals serverseitig
        picks = [p for p in picks if float(p.get("hauptwette_quote", 0)) >= 1.69]
        picks.sort(key=lambda x: x.get("startzeit", ""))
        return {"picks": picks[:6], "validierung": data.get("validierung", {})}
    except json.JSONDecodeError:
        return {"picks": [], "validierung": {}, "error": "JSON-Parsing fehlgeschlagen"}

# ── BASEBALL ENGINE ───────────────────────────────────────────────────────────

def extract_baseball_matches(image_bytes: bytes, league: str) -> list:
    client = get_openai_client()
    b64 = base64.b64encode(image_bytes).decode("ascii")
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": (
                f"Extract {league} baseball matchups from this betting screenshot as a JSON array. "
                "Each object: startzeit (HH:MM), liga, heim, gast, quote_heim (float), quote_gast (float), "
                "ou_line (float), hc_heim (string), hc_gast (string). "
                "ASCII only. Return ONLY the JSON array, no markdown."
            )},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]}
        ],
        max_tokens=1500
    )
    raw = res.choices[0].message.content.strip()
    raw = raw.encode("ascii", errors="replace").decode("ascii")
    raw = re.sub(r"```[a-z]*", "", raw).strip("`").strip()
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

def get_todays_mlb_games() -> dict:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=probablePitcher,team,venue,weather,linescore"
    try:
        r = requests.get(url, timeout=8)
        data = r.json()
    except Exception:
        return {}
    games = {}
    for date_entry in data.get("dates", []):
        for g in date_entry.get("games", []):
            h = g.get("teams", {}).get("home", {}).get("team", {}).get("name", "").lower()
            a = g.get("teams", {}).get("away", {}).get("team", {}).get("name", "").lower()
            games[(h, a)] = g
    return games

def get_pitcher_stats(player_id: int) -> dict:
    if not player_id:
        return {}
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=season&group=pitching&season={datetime.now().year}"
    try:
        r = requests.get(url, timeout=6)
        splits = r.json().get("stats", [{}])[0].get("splits", [{}])
        if not splits:
            return {}
        s = splits[0].get("stat", {})
        era = float(s.get("era", 99) or 99)
        k9  = float(s.get("strikeoutsPer9Inn", 0) or 0)
        bb9 = float(s.get("walksPer9Inn", 0) or 0)
        return {"era": era, "k_bb_per9": round(k9 - bb9, 2), "ip": float(s.get("inningsPitched", 0) or 0)}
    except Exception:
        return {}

def get_bullpen_load(team_id: int) -> float:
    if not team_id:
        return 0.0
    today = datetime.now(timezone.utc)
    total = 0.0
    for i in range(1, 4):
        d = today.replace(day=today.day - i).strftime("%Y-%m-%d")
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={d}&teamId={team_id}&hydrate=pitchers"
        try:
            r = requests.get(url, timeout=5)
            games = r.json().get("dates", [{}])[0].get("games", [])
            for g in games:
                for side in ["home", "away"]:
                    t = g.get("teams", {}).get(side, {})
                    if t.get("team", {}).get("id") == team_id:
                        total += max(0, len(t.get("pitchers", [])) - 1)
        except Exception:
            pass
    return round(total, 1)

def enrich_baseball(m: dict, mlb_games: dict) -> dict:
    heim = m.get("heim", "").lower()
    gast = m.get("gast", "").lower()
    game = None
    for (h, a), g in mlb_games.items():
        if heim in h or h in heim or gast in a or a in gast:
            game = g
            break
    if not game:
        return m
    for side, key in [("home", "pitcher_heim"), ("away", "pitcher_gast")]:
        p = game.get("teams", {}).get(side, {}).get("probablePitcher", {})
        pid = p.get("id")
        stats = get_pitcher_stats(pid) if pid else {}
        m[key] = {"name": p.get("fullName", "TBD"), **stats}
    for side, key in [("home", "bullpen_heim"), ("away", "bullpen_gast")]:
        tid = game.get("teams", {}).get(side, {}).get("team", {}).get("id")
        m[key] = get_bullpen_load(tid)
    w = game.get("weather", {})
    m["weather"] = {
        "temp_f":   int(w.get("temp", 72) or 72),
        "wind_mph": int(w.get("wind", "0 mph").split()[0]) if w.get("wind") else 0,
        "wind_dir": w.get("wind", "").split()[-1] if w.get("wind") else ""
    }
    venue = game.get("venue", {}).get("name", "").lower()
    m["venue"] = venue
    m["park_factor"] = PARK_FACTORS.get(venue, {}).get("run_factor", 1.0)
    return m

def apply_baseball_rules(matches: list, min_odds: float = 1.69) -> list:
    candidates = []
    used = set()
    for m in matches:
        mid = f"{m.get('heim','')}_{m.get('gast','')}"
        if mid in used:
            continue
        q_h = float(m.get("quote_heim", 0) or 0)
        q_a = float(m.get("quote_gast", 0) or 0)
        if max(q_h, q_a) < min_odds:
            continue
        w = m.get("weather", {})
        if w.get("temp_f", 72) < 45:
            continue
        if w.get("wind_mph", 0) > 15 and w.get("wind_dir", "") in ("in", "l-r", "r-l"):
            continue
        if m.get("bullpen_heim", 0) > 8 and m.get("bullpen_gast", 0) > 8:
            continue
        if q_h >= q_a:
            team, quote, opp = m["gast"], q_a, m["heim"]
            hc = str(m.get("hc_gast", "") or "")
        else:
            team, quote, opp = m["heim"], q_h, m["gast"]
            hc = str(m.get("hc_heim", "") or "")
        bet_type = "ML"
        bet = f"{team} ML"
        if hc and "+" in hc:
            bet = f"{team} {hc}"
            bet_type = "AH"
            quote = 2.05
        if quote < min_odds:
            continue
        p_h = m.get("pitcher_heim", {}).get("name", "TBD")
        p_a = m.get("pitcher_gast", {}).get("name", "TBD")
        w_str = f"{w.get('temp_f',72)}F, Wind {w.get('wind_mph',0)}mph {w.get('wind_dir','')}".strip()
        candidates.append({
            "time": m.get("startzeit", "--:--"),
            "game": f"{m.get('heim','')} vs {m.get('gast','')}",
            "league": m.get("liga", "MLB"),
            "bet_type": bet_type,
            "bet": bet,
            "odds": round(float(quote), 2),
            "pitcher_h": p_h,
            "pitcher_a": p_a,
            "weather": w_str
        })
        used.add(mid)
    candidates.sort(key=lambda x: x["time"])
    return candidates[:5]

# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(
    file:     UploadFile = File(...),
    sport:    str        = Form("baseball"),
    league:   str        = Form("MLB"),
    min_odds: float      = Form(1.69)
):
    try:
        content = await file.read()

        if sport == "fussball":
            result = analyze_fussball(content)
            return Response(
                content=json.dumps({"success": True, "sport": "fussball", **result}, ensure_ascii=False),
                media_type="application/json"
            )
        else:
            matches = extract_baseball_matches(content, league)
            if not matches:
                return Response(
                    content=json.dumps({"success": False, "picks": [], "message": "Keine Matchups erkannt."}, ensure_ascii=False),
                    media_type="application/json"
                )
            mlb_games = get_todays_mlb_games() if league == "MLB" else {}
            enriched  = [enrich_baseball(m, mlb_games) for m in matches]
            picks     = apply_baseball_rules(enriched, min_odds)
            return Response(
                content=json.dumps({"success": True, "sport": "baseball", "picks": picks, "count": len(picks)}, ensure_ascii=False),
                media_type="application/json"
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health():
    return {"status": "ok", "message": "SportTip API ready – Fussball + Baseball"}
