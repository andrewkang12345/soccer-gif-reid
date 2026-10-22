"""Inspect the user-provided public dataset share in a fresh browser session."""
import asyncio
import json
from pathlib import Path
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
URL = "https://pan.baidu.com/s/1F8zmO3_L4-G9rIxB8eXbbg?#list/path=%2F"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = await browser.new_context(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
        page = await context.new_page()
        await page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(3000)
        print("Title:", await page.title(), flush=True)
        print("URL:", page.url, flush=True)
        print("Inputs:", await page.locator("input").evaluate_all("els => els.map(e => ({type:e.type, placeholder:e.placeholder, id:e.id, name:e.name}))"), flush=True)
        print("Body:", (await page.locator("body").inner_text())[:7000], flush=True)
        await page.screenshot(path=str(ROOT / "reports/baidu_browser_initial.png"), full_page=True)
        # Only the explicitly user-provided public-share password is entered.
        field = page.locator('input[type="text"]:visible')
        if await field.count():
            await field.first.fill("bw6F")
            extract = page.get_by_text("提取文件", exact=True)
            if await extract.count():
                await extract.first.click()
            else:
                await field.first.press("Enter")
            await page.wait_for_timeout(5000)
        print("After password URL:", page.url, flush=True)
        print("After password body:", (await page.locator("body").inner_text())[:10000], flush=True)
        await page.screenshot(path=str(ROOT / "reports/baidu_browser_share.png"), full_page=True)
        private = ROOT / "data/baidu_browser_state.json"
        await context.storage_state(path=str(private))
        private.chmod(0o600)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
