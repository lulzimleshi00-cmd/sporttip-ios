import os
import base64
import json
import re
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

def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY nicht gesetzt.")
    return OpenAI(api_key=api_key)

def extract_matches(image_bytes: bytes, league: str = "MLB"):
    client = get_client()
    b64 = base64.b64encode(image_bytes).decode("ascii")
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    f"Extract {league} baseball matchups from this screenshot as a JSON array. "
                    "Each object must have these exact keys: "
                    "startzeit (HH:MM string), liga (string), heim (string), gast (string), "
                    "quote_heim (float), quote_gast (float), ou_line (float), "
                    "hc_heim (string), hc_gast (string). "
                    "Use only ASCII characters in team names. Return only valid JSON, no markdown."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    }
                ]
            }
        ],
        max_tokens=1500
    )
    raw = res.choices[0].message.content.strip()
    raw = raw.encode("ascii", errors="replace").decode("ascii")
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

def apply_rules(matches: list, min_odds: float = 1.69) -> list:
    candidates = []
    used = set()
    for m in matches:
        mid = f"{m.get('heim', '')}_{m.get('gast', '')}"
        if mid in used:
            continue
        q_h = float(m.get("quote_heim", 0) or 0)
        q_a = float(m.get("quote_gast", 0) or 0)
        best_q = max(q_h, q_a)
        if best_q < min_odds:
            continue
        hc_h = str(m.get("hc_heim", "") or "")
        if (best_q > 2.5 and m.get("liga") == "NPB") or ("pitcher" in hc_h.lower()):
            continue
        if q_h >= q_a:
            team, quote, bet, opp = m["gast"], q_a, f"{m['gast']} ML", m["heim"]
            hc = str(m.get("hc_gast", "") or "")
        else:
            team, quote, bet, opp = m["heim"], q_h, f"{m['heim']} ML", m["gast"]
            hc = str(m.get("hc_heim", "") or "")
        if hc and "+" in hc:
            bet = f"{team} {hc}"
            quote = 2.05
        if quote < min_odds:
            continue
        candidates.append({
            "time": m.get("startzeit", "--:--"),
            "game": f"{team} @ {opp}",
            "league": m.get("liga", ""),
            "bet_type": "ML",
            "bet": bet,
            "odds": round(float(quote), 2)
        })
        used.add(mid)
    candidates.sort(key=lambda x: x["time"])
    return candidates[:5]

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    league: str = Form("MLB"),
    min_odds: float = Form(1.69)
):
    try:
        content = await file.read()
        matches = extract_matches(content, league)
        if not matches:
            return Response(
                content=json.dumps({"success": False, "picks": [], "message": "Keine Matchups erkannt."}, ensure_ascii=False),
                media_type="application/json"
            )
        picks = apply_rules(matches, min_odds)
        result = {"success": True, "picks": picks, "count": len(picks)}
        return Response(
            content=json.dumps(result, ensure_ascii=False),
            media_type="application/json"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health():
    return {"status": "ok", "message": "SportTip API ready"}
