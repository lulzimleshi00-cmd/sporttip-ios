import os
import base64
import json
import re
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
        raise RuntimeError("OPENAI_API_KEY ist nicht gesetzt. Bitte in Vercel Environment Variables prüfen.")
    return OpenAI(api_key=api_key)

def extract_matches(image_bytes: bytes):
    client = get_client()
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
    if not match: return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

def apply_rules(matches: list) -> list:
    candidates = []
    used = set()
    for m in matches:
        mid = f"{m.get('heim','')}_{m.get('gast','')}"
        if mid in used: continue

        q_h = m.get('quote_heim', 0)
        q_a = m.get('quote_gast', 0)
        best_q = max(q_h, q_a)
        if best_q < 1.69: continue

        hc_h = m.get('hc_heim', '') or ''
        if (best_q > 2.5 and m.get('liga') == 'NPB') or ('pitcher' in hc_h.lower()):
            continue

        if q_h >= q_a:
            team, quote, bet, opp = m['gast'], q_a, f"{m['gast']} ML", m['heim']
            hc = m.get('hc_gast', '')
        else:
            team, quote, bet, opp = m['heim'], q_h, f"{m['heim']} ML", m['gast']
            hc = m.get('hc_heim', '')

        if hc and '+' in hc:
            bet, quote = f"{team} {hc}", 2.05

        if quote < 1.69: continue

        candidates.append({
            "startzeit": m['startzeit'],
            "spiel": f"{team} @ {opp} ({m['liga']})",
            "hauptwette": bet,
            "quote": round(quote, 2)
        })
        used.add(mid)

    candidates.sort(key=lambda x: x['startzeit'])
    return candidates[:5]

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        content = await file.read()
        matches = extract_matches(content)
        if not matches:
            return {"success": False, "ticket": [], "message": "Keine Matchups erkannt. Bitte Screenshot prüfen."}
        ticket = apply_rules(matches)
        return {"success": True, "ticket": ticket, "count": len(ticket)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health():
    return {"status": "ok", "message": "SportTip API ready"}
