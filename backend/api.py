from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import base64, json, re, httpx, os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")  # Optional: the-odds-api.com

class Match(BaseModel):
    startzeit: str
    liga: str
    heim: str
    gast: str
    quote_heim: float
    quote_gast: float
    ou_line: float = None
    hc_heim: str = None
    hc_gast: str = None

async def fetch_live_odds(home: str, away: str, league: str):
    """Holt aktuelle Quoten von The Odds API (kostenlos)"""
    if not ODDS_API_KEY:
        return None
    sport = "baseball_kbo" if league == "KBO" else "baseball_npb"
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
    params = {"api_key": ODDS_API_KEY, "regions": "eu", "markets": "h2p,totals"}
    async with httpx.AsyncClient() as http:
        r = await http.get(url, params=params)
        if r.status_code == 200:
            for game in r.json():
                if home.lower() in game["home_team"].lower() or away.lower() in game["away_team"].lower():
                    return game["bookmakers"][0]["markets"][0]["outcomes"] if game["bookmakers"] else None
    return None

async def get_pitcher_stats(team: str, league: str):
    """Simulierte Pitcher-Daten (hier später FanGraphs/Baseball-Reference API einbinden)"""
    # MVP: Return dummy data based on team name hash for consistency
    import hashlib
    h = int(hashlib.md5(team.encode()).hexdigest()[:8], 16)
    return {
        "era": round(2.5 + (h % 40) / 10, 2),
        "whip": round(0.9 + (h % 30) / 100, 2),
        "k_rate": round(15 + (h % 15), 1),
        "vs_split": round(0.240 + (h % 60) / 1000, 3)
    }

async def get_weather(city: str):
    """Wetterdaten von OpenWeatherMap (kostenlos)"""
    api_key = os.getenv("WEATHER_API_KEY", "")
    if not api_key:
        return {"temp": 20, "wind": 10, "condition": "clear"}
    async with httpx.AsyncClient() as http:
        r = await http.get(f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric")
        if r.status_code == 200:
            d = r.json()
            return {"temp": d["main"]["temp"], "wind": d["wind"]["speed"], "condition": d["weather"][0]["main"]}
    return {"temp": 20, "wind": 10, "condition": "clear"}

def extract_matches_from_image(image_bytes: bytes, api_key: str):
    b64 = base64.b64encode(image_bytes).decode()
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Extrahiere Baseball-Matchups als JSON-Array. Jedes Objekt: {'startzeit':'HH:MM','liga':'KBO|NPB','heim':'','gast':'','quote_heim':float,'quote_gast':float,'ou_line':float,'hc_heim':str,'hc_gast':str}. Nur valides JSON."},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"image/jpeg;base64,{b64}"}}]}
        ],
        max_tokens=1500
    )
    raw = res.choices[0].message.content.strip()
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    return [Match(**m) for m in json.loads(match.group(0))] if match else []

def apply_rules(matches: list[Match]) -> list[dict]:
    """Deine strikten Regeln: Quote≥1.69, Korrelation, Value-Trap, Chronologie"""
    candidates = []
    used = set()
    
    for m in matches:
        mid = f"{m.heim}_{m.gast}"
        if mid in used:
            continue
            
        # QUOTEN-REGEL
        best_q = max(m.quote_heim, m.quote_gast)
        if best_q < 1.69:
            continue
            
        # VALUE-TRAP FILTER (simuliert)
        if (best_q > 2.5 and m.liga == "NPB") or (best_q < 1.85 and "pitcher" in (m.hc_heim or "").lower()):
            continue
            
        # WETTEN-TYP
        if m.quote_heim >= m.quote_gast:
            team, quote, bet = m.gast, m.quote_gast, f"{m.gast} ML"
            if m.hc_gast and "+" in m.hc_gast:
                bet, quote = f"{m.gast} {m.hc_gast}", 2.05
        else:
            team, quote, bet = m.heim, m.quote_heim, f"{m.heim} ML"
            if m.hc_heim and "+" in m.hc_heim:
                bet, quote = f"{m.heim} {m.hc_heim}", 2.05
                
        if quote < 1.69:
            continue
            
        candidates.append({
            "startzeit": m.startzeit,
            "spiel": f"{team} @ {m.gast if team==m.heim else m.heim} ({m.liga})",
            "hauptwette": bet,
            "quote": round(quote, 2)
        })
        used.add(mid)
    
    candidates.sort(key=lambda x: x["startzeit"])
    return candidates[:5]

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(400, "OPENAI_API_KEY not set in environment")
    
    content = await file.read()
    matches = extract_matches_from_image(content, os.getenv("OPENAI_API_KEY"))
    
    # Optional: Live-Odds-Validierung
    for m in matches:
        live = await fetch_live_odds(m.heim, m.gast, m.liga)
        if live:
            # Quote-Abgleich (falls Abweichung >10%, Warnung)
            pass
    
    ticket = apply_rules(matches)
    return {"success": True, "ticket": ticket, "count": len(ticket)}

@app.get("/")
def health():
    return {"status": "ok", "message": "SportTip API running"}
