"""Try the share's visible browser download flow; record any server/login barrier."""
import asyncio
import json
import re
from pathlib import Path
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = await browser.new_context(storage_state=str(ROOT / "data/baidu_browser_state.json"),
                                           viewport={"width": 1440, "height": 1000}, accept_downloads=True)
        page = await context.new_page()
        async def record_response(response):
            if "/api/sharedownload" in response.url:
                try:
                    data = await response.json()
                    print("Browser download API:", {"http": response.status, "errno": data.get("errno"),
                                                    "list_type": type(data.get("list")).__name__,
                                                    "show_msg": data.get("show_msg")}, flush=True)
                except Exception as error:
                    print("Browser download response error:", str(error)[:300], flush=True)
        page.on("response", record_response)
        await page.goto("https://pan.baidu.com/s/1F8zmO3_L4-G9rIxB8eXbbg?#list/path=%2F", wait_until="domcontentloaded", timeout=60000)
        for name in ["MuPNIT", "Release_versions"]:
            link = page.get_by_text(name, exact=True)
            await link.wait_for(timeout=30000)
            print("Directory link:", name, await link.evaluate("e => ({tag:e.tagName,href:e.getAttribute('href'),cls:e.className})"), flush=True)
            await link.click()
            await page.wait_for_timeout(2500)
        print("Archive listing:", (await page.locator("body").inner_text())[:10000], flush=True)
        target = page.get_by_text("MuPNIT-ReID.tar.gz", exact=True)
        await target.wait_for(timeout=30000)
        print("Archive element:", await target.evaluate("e => e.outerHTML"), flush=True)
        for row in await page.locator("dd").all():
            filename = row.locator("a.filename")
            if not await filename.count():
                continue
            name = await filename.inner_text()
            selected = "JS-item-active" in (await row.get_attribute("class") or "")
            if selected != (name == "MuPNIT-ReID.tar.gz"):
                await row.locator('span[node-type="EOGexf"]').click()
        print("Selected files:", await page.locator("dd.JS-item-active a.filename").all_text_contents(), flush=True)
        print("After archive click:", (await page.locator("body").inner_text())[:10000], flush=True)
        await page.screenshot(path=str(ROOT / "reports/baidu_browser_archive.png"), full_page=True)
        print("Download controls:", await page.get_by_text("下载", exact=True).count(), flush=True)
        controls = page.get_by_text("下载", exact=True)
        downloads = []
        page.on("download", lambda download: downloads.append(download))
        button = page.locator("span.text:visible").filter(has_text=re.compile("^下载$"))
        await button.click()
        await page.wait_for_timeout(30000)
        print("After download:", (await page.locator("body").inner_text())[:10000], flush=True)
        print("Frames:", [(frame.name, frame.url.split('?')[0]) for frame in page.frames], flush=True)
        for frame in page.frames[1:]:
            try:
                print("Frame text:", (await frame.locator("body").inner_text(timeout=3000))[:6000], flush=True)
            except Exception:
                pass
        if downloads:
            for download in downloads:
                destination = ROOT / "data" / download.suggested_filename
                print("Saving download:", destination, flush=True)
                await download.save_as(str(destination))
                print("Download bytes:", destination.stat().st_size, flush=True)
        await page.screenshot(path=str(ROOT / "reports/baidu_browser_download.png"), full_page=True)
        await context.storage_state(path=str(ROOT / "data/baidu_browser_state.json"))
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
