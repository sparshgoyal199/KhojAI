"""Comprehensive API smoke/integration test script."""
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

import httpx

BASE = os.getenv("TEST_BASE_URL", "http://127.0.0.1:8000")
RESULTS = []


def record(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    RESULTS.append((status, name, detail))
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


async def main():
    email = f"test_{uuid.uuid4().hex[:8]}@example.com"
    password = "TestPass123!"
    name = "API Tester"
    token = None
    pdf_id = None
    share_token = None
    session_id = None

    async with httpx.AsyncClient(base_url=BASE, timeout=httpx.Timeout(300.0)) as client:
        # Static pages
        for path in ["/", "/dashboard", "/css/styles.css", "/js/api.js"]:
            r = await client.get(path)
            record(f"GET {path}", r.status_code == 200, str(r.status_code))

        # Auth — signup
        r = await client.post("/auth/signup", json={"name": name, "email": email, "password": password})
        record("POST /auth/signup", r.status_code == 200, r.text[:120] if r.status_code != 200 else f"id={r.json().get('id')}")

        # Duplicate signup
        r2 = await client.post("/auth/signup", json={"name": name, "email": email, "password": password})
        record("POST /auth/signup duplicate", r2.status_code == 400, str(r2.status_code))

        # Login wrong password
        r3 = await client.post("/auth/login", json={"email": email, "password": "wrong"})
        record("POST /auth/login wrong password", r3.status_code == 401, str(r3.status_code))

        # Login ok
        r = await client.post("/auth/login", json={"email": email, "password": password})
        ok = r.status_code == 200 and "access_token" in r.json()
        record("POST /auth/login", ok, str(r.status_code))
        if ok:
            token = r.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        # Protected without token
        r = await client.get("/pdfs")
        record("GET /pdfs without auth", r.status_code == 401, str(r.status_code))

        # List empty
        r = await client.get("/pdfs", headers=headers)
        record("GET /pdfs", r.status_code == 200, f"count={len(r.json())}")

        # Search with no results
        r = await client.get("/pdfs/search", params={"q": "nonexistent_xyz"}, headers=headers)
        record("GET /pdfs/search", r.status_code == 200, str(r.status_code))

        # Upload non-pdf
        r = await client.post(
            "/upload",
            headers=headers,
            files={"file": ("bad.txt", b"not a pdf", "text/plain")},
        )
        record("POST /upload non-pdf", r.status_code == 400, r.text[:100])

        # Try to use existing PDFs from list if any
        r = await client.get("/pdfs", headers=headers)
        existing = r.json() if r.status_code == 200 else []

        # Upload real pdf if we can create a minimal one — use reportlab or skip
        # Minimal valid PDF bytes
        minimal_pdf = (
            b"%PDF-1.1\n1 0 obj<<>>endobj\n2 0 obj<</Length 44>>stream\n"
            b"BT /F1 12 Tf 100 700 Td (Hello KhojAI test) Tj ET\n"
            b"endstream\nendobj\n3 0 obj<</Type/Page/Parent 4 0 R/MediaBox[0 0 612 792]/Contents 2 0 R>>endobj\n"
            b"4 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"5 0 obj<</Type/Catalog/Pages 4 0 R>>endobj\n"
            b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000022 00000 n \n"
            b"0000000105 00000 n \n0000000190 00000 n \n0000000249 00000 n \n"
            b"trailer<</Size 6/Root 5 0 R>>\nstartxref\n298\n%%EOF"
        )

        print("\n--- Upload test (may take 1-3 min) ---")
        try:
            r = await client.post(
                "/upload",
                headers=headers,
                files={"file": ("api_test_doc.pdf", minimal_pdf, "application/pdf")},
                timeout=httpx.Timeout(600.0),
            )
            upload_ok = r.status_code == 200
            record("POST /upload pdf", upload_ok, f"{r.status_code} {r.text[:150]}")
            if upload_ok:
                data = r.json()
                pdf_id = data.get("id")
        except Exception as e:
            record("POST /upload pdf", False, str(e))

        # If upload failed, try first existing pdf from any user list — re-login not possible; use list from before signup... 
        # Actually after signup list is empty. Try listing again in case upload failed but user had pdfs from before — new user won't.
        if not pdf_id and existing:
            pdf_id = existing[0]["id"]

        if not pdf_id:
            # Try to login as existing user isn't available — skip pdf-dependent tests
            print("\nSkipping PDF-dependent tests (no pdf_id)")
        else:
            pdf_id_str = str(pdf_id).replace("-", "")

            r = await client.get(f"/pdfs/{pdf_id_str}", headers=headers)
            record("GET /pdfs/{id}", r.status_code == 200, f"file_url={'yes' if r.json().get('file_url') else 'no'}")

            r = await client.get(f"/pdfs/{pdf_id}", headers=headers)
            record("GET /pdfs/{id} with dashes", r.status_code == 200, str(r.status_code))

            r = await client.get("/pdfs/not-a-real-id", headers=headers)
            record("GET /pdfs/{id} 404", r.status_code == 404, str(r.status_code))

            r = await client.post(f"/pdfs/{pdf_id_str}/share", headers=headers)
            share_ok = r.status_code == 200
            record("POST /pdfs/{id}/share", share_ok, r.text[:120])
            if share_ok:
                share_token = r.json().get("share_token")
                share_url = r.json().get("share_url", "")
                record("share_url format /shared/{token}", "/shared/" in share_url, share_url)

            if share_token:
                r = await client.get(f"/pdfs/shared/{share_token}")
                record("GET /pdfs/shared/{token}", r.status_code == 200, str(r.status_code))

                r = await client.get(f"/shared/{share_token}/comments")
                record("GET /shared/{token}/comments", r.status_code == 200, str(r.status_code))

                r = await client.post(
                    f"/shared/{share_token}/comments",
                    json={"content": "Guest comment from test", "guest_name": "TestGuest"},
                )
                record("POST /shared/{token}/comments", r.status_code == 200, str(r.status_code))

                r = await client.post(
                    f"/shared/{share_token}/chat/history",
                    json={"guest_name": "TestGuest"},
                )
                record("POST /shared/{token}/chat/history", r.status_code == 200, str(r.status_code))
                if r.status_code == 200:
                    session_id = r.json().get("session_id")

                # Guest query streaming
                async with client.stream(
                    "POST",
                    f"/shared/{share_token}/query",
                    json={"query": "What is this document about?", "guest_name": "TestGuest"},
                ) as stream:
                    chunks = []
                    async for line in stream.aiter_lines():
                        if line.startswith("data:"):
                            chunks.append(line)
                    record(
                        "POST /shared/{token}/query stream",
                        len(chunks) > 0 and any("done" in c for c in chunks),
                        f"events={len(chunks)}",
                    )

                r = await client.post(
                    f"/shared/{share_token}/query",
                    json={"query": "test", "guest_name": ""},
                )
                record("POST /shared/{token}/query missing guest_name", r.status_code == 400, str(r.status_code))

            r = await client.get(f"/pdfs/{pdf_id_str}/comments", headers=headers)
            record("GET /pdfs/{id}/comments", r.status_code == 200, str(r.status_code))

            r = await client.post(
                f"/pdfs/{pdf_id_str}/comments",
                headers=headers,
                json={"content": "Owner comment from test"},
            )
            record("POST /pdfs/{id}/comments", r.status_code == 200, str(r.status_code))

            r = await client.get(f"/pdfs/{pdf_id_str}/chat/history", headers=headers)
            record("GET /pdfs/{id}/chat/history", r.status_code == 200, str(r.status_code))

            async with client.stream(
                "POST",
                f"/pdfs/{pdf_id_str}/query",
                headers=headers,
                json={"query": "Summarize in one sentence.", "guest_name": None},
            ) as stream:
                chunks = []
                async for line in stream.aiter_lines():
                    if line.startswith("data:"):
                        chunks.append(line)
                record(
                    "POST /pdfs/{id}/query stream",
                    len(chunks) > 0 and any("done" in c for c in chunks),
                    f"events={len(chunks)}",
                )

        # Viewer page route
        if pdf_id:
            r = await client.get(f"/viewer/{pdf_id}")
            record("GET /viewer/{id} page", r.status_code == 200, str(r.status_code))
        if share_token:
            r = await client.get(f"/shared/{share_token}")
            record("GET /shared/{token} page", r.status_code == 200, str(r.status_code))

    fails = [x for x in RESULTS if x[0] == "FAIL"]
    print("\n" + "=" * 60)
    print(f"TOTAL: {len(RESULTS)}  PASS: {len(RESULTS)-len(fails)}  FAIL: {len(fails)}")
    if fails:
        print("\nFailures:")
        for _, name, detail in fails:
            print(f"  - {name}: {detail}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
