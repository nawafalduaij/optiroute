"""Capture demo-UI screenshots (and the crops used by the deck) with headless Chromium.

  .venv/bin/uvicorn app:app --port 8000 &
  .venv/bin/pip install playwright && .venv/bin/python -m playwright install chromium
  .venv/bin/python scripts/capture_screenshots.py
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "deliverables" / "screenshots"
URL = "http://localhost:8000/"


async def capture() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1.5)
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        await page.goto(URL)
        await page.click("#reset")
        await page.wait_for_timeout(400)

        async def preset(label: str) -> None:
            await page.click(f".preset:has-text('{label}')")
            await page.wait_for_selector(".card.final", timeout=15000)
            await page.wait_for_timeout(700)

        for label in ("Cache seed", "Paraphrase hit", "Near-duplicate trap"):
            await preset(label)
        await page.screenshot(path=OUT / "live_trap_blocked.png", full_page=True)
        # Deck crop: query card, stage rail, pruner diff and cache-guard card of the blocked-trap run
        top = await page.locator("#qcard").bounding_box()
        guard = await page.locator("#details > div").nth(1).bounding_box()
        await page.screenshot(path=OUT / "crop_live_trap.png", full_page=True, clip={
            "x": top["x"] - 8, "y": top["y"] - 8, "width": top["width"] + 16,
            "height": guard["y"] + guard["height"] - top["y"] + 16})
        await preset("Low-overlap")
        await page.screenshot(path=OUT / "live_escalation.png", full_page=True)
        await preset("Easy · Arabic")
        await page.screenshot(path=OUT / "live_arabic.png", full_page=True)
        await page.click(".tabs button[data-v=bench]")
        await page.wait_for_timeout(1500)
        await page.screenshot(path=OUT / "benchmark.png", full_page=True)
        tau_box = await page.locator("#waveChart").bounding_box()
        await page.screenshot(path=OUT / "crop_tau.png", clip=tau_box)
        await browser.close()
    if errors:
        raise SystemExit(f"page errors: {errors}")
    print("saved:", ", ".join(sorted(f.name for f in OUT.glob("*.png"))))


if __name__ == "__main__":
    asyncio.run(capture())
