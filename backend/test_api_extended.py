"""Extended API tests with real PDF upload."""
import asyncio
import json
import uuid
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"
PDF = Path(r"C:\Users\hp\AppData\Local\Temp\khojai_test.pdf")
EMAIL = "fulltest.user@gmail.com"
PASSWORD = "TestPass123!"


async def run():
    results = []

    def ok(name, passed, detail=""):
        results.append((passed, name, detail))
        print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    async with httpx.AsyncClient(base_url=BASE, timeout=httpx.Timeout(600.0)) as c:
        # signup (ignore if exists)
        await c.post("/auth/signup", json={"name": "Full Tester", "email": EMAIL, "password": PASSWORD})
        token = (await c.post("/auth/login", json={"email": EMAIL, "password": PASSWORD})).json()["access_token"]
        h = {"Authorization": f"Bearer {token}"}

        # upload real pdf
        with PDF.open("rb") as f:
            r = await c.post("/upload", headers=h, files={"file": ("attention_paper.pdf", f, "application/pdf")})
        ok("POST /upload real pdf", r.status_code == 200, f"{r.status_code} {r.text[:200]}")
        if r.status_code != 200:
            return results
        pdf_id = r.json()["id"]
        pdf_id_clean = str(pdf_id).replace("-", "")

        r = await c.get(f"/pdfs/{pdf_id_clean}", headers=h)
        ok("GET /pdfs/{id}", r.status_code == 200, f"summary={'yes' if r.json().get('summary') else 'no'} file_url={'yes' if r.json().get('file_url') else 'no'}")

        r = await c.post(f"/pdfs/{pdf_id_clean}/share", headers=h)
        ok("POST /pdfs/{id}/share", r.status_code == 200, r.text[:120])
        share_token = r.json()["share_token"]
        share_url = r.json()["share_url"]

        ok("share_url uses /shared/ path", "/shared/" in share_url, share_url)

        r = await c.get(f"/pdfs/shared/{share_token}")
        ok("GET /pdfs/shared/{token}", r.status_code == 200, str(r.status_code))

        r = await c.post(f"/pdfs/{pdf_id_clean}/comments", headers=h, json={"content": "Owner test comment"})
        ok("POST /pdfs/{id}/comments", r.status_code == 200, str(r.status_code))

        r = await c.get(f"/pdfs/{pdf_id_clean}/comments", headers=h)
        ok("GET /pdfs/{id}/comments", r.status_code == 200 and len(r.json()) >= 1, f"count={len(r.json())}")

        r = await c.post(f"/shared/{share_token}/comments", json={"content": "Guest comment", "guest_name": "GuestQA"})
        ok("POST /shared/{token}/comments", r.status_code == 200, str(r.status_code))

        r = await c.get(f"/shared/{share_token}/comments")
        ok("GET /shared/{token}/comments", r.status_code == 200, f"count={len(r.json())}")

        r = await c.get(f"/pdfs/{pdf_id_clean}/chat/history", headers=h)
        ok("GET /pdfs/{id}/chat/history", r.status_code == 200, str(r.status_code))

        r = await c.post(f"/shared/{share_token}/chat/history", json={"guest_name": "GuestQA"})
        ok("POST /shared/{token}/chat/history", r.status_code == 200, str(r.status_code))

        # owner query stream
        events = []
        async with c.stream("POST", f"/pdfs/{pdf_id_clean}/query", headers=h, json={"query": "What is attention?", "guest_name": None}) as s:
            async for line in s.aiter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line[5:].strip()))
        ok("POST /pdfs/{id}/query SSE", len(events) > 1 and events[-1].get("delta") == "done", f"events={len(events)}")
        ok("POST /pdfs/{id}/query has content", any(e.get("delta") not in (None, "", "done") for e in events), "")

        # guest query stream
        events2 = []
        async with c.stream("POST", f"/shared/{share_token}/query", json={"query": "What model is discussed?", "guest_name": "GuestQA"}) as s:
            async for line in s.aiter_lines():
                if line.startswith("data:"):
                    events2.append(json.loads(line[5:].strip()))
        ok("POST /shared/{token}/query SSE", len(events2) > 1, f"events={len(events2)}")

        # chat history after query
        r = await c.get(f"/pdfs/{pdf_id_clean}/chat/history", headers=h)
        msgs = r.json().get("messages", [])
        ok("chat history persisted", len(msgs) >= 2, f"messages={len(msgs)}")

        r = await c.get("/pdfs/search", params={"q": "attention"}, headers=h)
        ok("GET /pdfs/search", r.status_code == 200, f"count={len(r.json())}")

        # access control: another user cannot access pdf
        email2 = f"other_{uuid.uuid4().hex[:6]}@gmail.com"
        await c.post("/auth/signup", json={"name": "Other", "email": email2, "password": PASSWORD})
        token2 = (await c.post("/auth/login", json={"email": email2, "password": PASSWORD})).json()["access_token"]
        h2 = {"Authorization": f"Bearer {token2}"}
        r = await c.get(f"/pdfs/{pdf_id_clean}", headers=h2)
        ok("GET /pdfs/{id} forbidden for other user", r.status_code == 403, str(r.status_code))

        r = await c.get(f"/pdfs/{pdf_id_clean}/comments", headers=h2)
        ok("GET /pdfs/{id}/comments forbidden", r.status_code in (403, 404), str(r.status_code))

        # invalid token
        r = await c.get("/pdfs", headers={"Authorization": "Bearer badtoken"})
        ok("GET /pdfs invalid token", r.status_code == 401, str(r.status_code))

        # missing auth header
        r = await c.get("/pdfs")
        ok("GET /pdfs missing auth", r.status_code in (401, 422), str(r.status_code))

        # static pages with ids
        r = await c.get(f"/viewer/{pdf_id_clean}")
        ok("GET /viewer/{id}", r.status_code == 200, str(r.status_code))
        r = await c.get(f"/shared/{share_token}")
        ok("GET /shared/{token} page", r.status_code == 200, str(r.status_code))

    return results


if __name__ == "__main__":
    res = asyncio.run(run())
    fails = [x for x in res if not x[0]]
    print(f"\nFAILURES: {len(fails)}")
    for _, name, detail in fails:
        print(f"  {name}: {detail}")
