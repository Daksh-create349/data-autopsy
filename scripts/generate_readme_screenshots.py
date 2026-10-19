"""Automated script to capture high-resolution screenshots of Data Autopsy for README."""

import asyncio
import base64
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import tornado.websocket

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CDP_PORT = 9222
BRAVE_PATH = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"


async def click_tab(ws, tab_name: str):
    """Click a tab button by its display text."""
    click_js = f"""
    (() => {{
        const tabs = Array.from(document.querySelectorAll('button[role="tab"]'));
        const target = tabs.find(t => t.textContent.includes('{tab_name}'));
        if (target) {{
            target.click();
            return 'clicked';
        }}
        return 'not_found';
    }})()
    """
    await ws.write_message(json.dumps({"id": 100, "method": "Runtime.evaluate", "params": {"expression": click_js}}))
    await asyncio.sleep(2.5)


async def scroll_page(ws, y_offset: int):
    """Scroll main window by vertical pixels."""
    scroll_js = f"window.scrollTo(0, {y_offset});"
    await ws.write_message(json.dumps({"id": 101, "method": "Runtime.evaluate", "params": {"expression": scroll_js}}))
    await asyncio.sleep(1.0)


async def capture(ws, filename: str, msg_id: int):
    """Capture viewport screenshot and save to disk."""
    out_path = OUTPUT_DIR / filename
    await ws.write_message(json.dumps({"id": msg_id, "method": "Page.captureScreenshot", "params": {"format": "png"}}))
    while True:
        resp = await ws.read_message()
        data = json.loads(resp)
        if data.get("id") == msg_id:
            img_bytes = base64.b64decode(data["result"]["data"])
            out_path.write_bytes(img_bytes)
            print(f"Captured {filename} ({len(img_bytes):,} bytes)")
            break


async def main():
    print(f"Starting Brave headless process at port {CDP_PORT}...")
    target_url = "http://localhost:8501/?fixture=demo_4_target_copy_leakage.csv"
    proc = subprocess.Popen([
        BRAVE_PATH,
        "--headless=new",
        f"--remote-debugging-port={CDP_PORT}",
        "--disable-gpu",
        "--window-size=1440,1050",
        target_url,
    ])

    try:
        await asyncio.sleep(2.5)
        req = urllib.request.urlopen(f"http://localhost:{CDP_PORT}/json/list")
        targets = json.loads(req.read().decode())
        page_target = next(t for t in targets if t.get("type") == "page")
        ws_url = page_target["webSocketDebuggerUrl"]
        print("Connected to WebSocket CDP:", ws_url)

        ws = await tornado.websocket.websocket_connect(ws_url)
        await ws.write_message(json.dumps({"id": 1, "method": "Page.enable"}))

        print("Waiting 6 seconds for initial Streamlit engine calculation and render...")
        await asyncio.sleep(6)

        # 1. Overview Dashboard
        print("1. Capturing Overview Dashboard...")
        await click_tab(ws, "Overview")
        await scroll_page(ws, 0)
        await capture(ws, "01_overview_dashboard.png", 10)

        # 2. Dataset Profile
        print("2. Capturing Dataset Profile...")
        await click_tab(ws, "Dataset Profile")
        await scroll_page(ws, 120)
        await capture(ws, "02_dataset_profile.png", 20)

        # 3. Forensic Findings
        print("3. Capturing Forensic Findings...")
        await click_tab(ws, "Forensic Findings")
        await scroll_page(ws, 100)
        await capture(ws, "03_forensic_findings.png", 30)

        # 4. Leakage Investigation
        print("4. Capturing Leakage Investigation & Baseline Comparison...")
        await click_tab(ws, "Leakage Investigation")
        await scroll_page(ws, 280)
        await capture(ws, "04_leakage_investigation.png", 40)

        # 5. Evidence Inspector
        print("5. Capturing Evidence Inspector...")
        await click_tab(ws, "Evidence Inspector")
        await scroll_page(ws, 100)
        await capture(ws, "05_evidence_inspector.png", 50)

        # 6. AI Investigator
        print("6. Capturing AI Investigator...")
        await click_tab(ws, "AI Investigator")
        await scroll_page(ws, 60)
        await capture(ws, "06_ai_investigator.png", 60)

        # 7. Report Export
        print("7. Capturing Report Export...")
        await click_tab(ws, "Report Export")
        await scroll_page(ws, 100)
        await capture(ws, "07_report_export.png", 70)

        ws.close()
        print("All 7 screenshots generated successfully in docs/images/!")

    finally:
        proc.kill()


if __name__ == "__main__":
    asyncio.run(main())
