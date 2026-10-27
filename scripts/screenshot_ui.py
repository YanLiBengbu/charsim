"""Screenshot of the Gradio interface (Fig3.png) through the Chrome DevTools protocol.

Starts headless Chrome on a running interface (scripts/app.py), presses Start and Run to the end
with the default settings (Llama 3.1 8B, pearls, soul with all tools) and seed 100, and waits for the
scene to finish. The figure has two panels cut from the same page: (A) the settings and the
character tools, (B) the controls, the status and the 20 moves of the scene with the urgencies. The
character cards between them are left out, so that the text stays legible at the width of a page.
Run: python scripts/app.py --port 7860 &   then   python scripts/screenshot_ui.py --out paper/Fig3.png"""

import argparse
import asyncio
import base64
import io
import json
import subprocess
import time
import urllib.request

import websockets
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"


class Page:
    def __init__(self, ws):
        self.ws, self.n = ws, 0

    async def call(self, method, **params):
        self.n += 1
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == self.n:
                return msg.get("result", {})

    async def js(self, expr):
        r = await self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")

    async def click(self, text):
        return await self.js(f"""(() => {{ const b = [...document.querySelectorAll('button')]
            .find(x => x.innerText.trim() === {json.dumps(text)}); if (b) {{ b.click(); return true; }} return false; }})()""")

    async def status(self):
        return await self.js("""(() => { const t = [...document.querySelectorAll('textarea')]
            .find(x => x.closest('.block') && x.closest('.block').innerText.startsWith('Status')); return t ? t.value : ''; })()""")

    async def top(self, js_find):
        """Top of an element in page coordinates (CSS pixels)."""
        return await self.js(f"""(() => {{ const e = {js_find}; return e.getBoundingClientRect().top + window.scrollY; }})()""")


def panels(png, cuts, scale, label_px):
    """The two panels of the figure, each under a band with its letter."""
    page = Image.open(io.BytesIO(png)).convert("RGB")
    w = page.size[0]
    font = ImageFont.truetype(FONT, label_px)
    parts = []
    for letter, (a, b) in zip("AB", cuts):
        crop = page.crop((0, int(a * scale), w, int(b * scale)))
        band = Image.new("RGB", (w, int(label_px * 1.4)), "white")
        ImageDraw.Draw(band).text((int(label_px * 0.4), int(label_px * 0.2)), letter, fill="black", font=font)
        parts += [band, crop]
    out = Image.new("RGB", (w, sum(p.size[1] for p in parts)), "white")
    y = 0
    for p in parts:
        out.paste(p, (0, y))
        y += p.size[1]
    return out


async def main(args):
    chrome = subprocess.Popen(["google-chrome", "--headless=new", "--disable-gpu", "--no-first-run",
                               f"--remote-debugging-port={args.cdp}", f"--window-size={args.width},{args.height}",
                               "--hide-scrollbars", args.url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{args.cdp}/json"))
                tab = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(1)
        async with websockets.connect(tab["webSocketDebuggerUrl"], max_size=2 ** 28) as ws:
            page = Page(ws)
            await page.call("Page.enable")
            await page.call("Emulation.setDeviceMetricsOverride", width=args.width, height=args.height,
                            deviceScaleFactor=args.scale, mobile=False)
            for _ in range(60):
                if await page.js("document.querySelectorAll('button').length > 3"):
                    break
                await asyncio.sleep(2)
            # the first seed of the experiments
            await page.js(f"""(() => {{ const b = [...document.querySelectorAll('.block')]
                .find(x => x.innerText.trim().startsWith('Seed')); const i = b.querySelector('input');
                i.value = '{args.seed}'; i.dispatchEvent(new Event('input', {{bubbles: true}}));
                i.dispatchEvent(new Event('change', {{bubbles: true}})); return i.value; }})()""")
            await asyncio.sleep(1)
            await page.click("Start")
            for _ in range(300):
                if (await page.status() or "").startswith("turn 0"):
                    break
                await asyncio.sleep(2)
            await page.click("Run to the end")
            t0 = time.time()
            while time.time() - t0 < args.timeout:
                st = await page.status() or ""
                if st.startswith("turn 20 /"):
                    break
                await asyncio.sleep(5)
            print("status:", await page.status())
            await asyncio.sleep(3)
            cards = await page.top("[...document.querySelectorAll('.block')].find(x => x.innerText.trim().startsWith('Card of'))")
            buttons = await page.top("[...document.querySelectorAll('button')].find(x => x.innerText.trim() === 'Start')")
            session = await page.top("[...document.querySelectorAll('.block')].find(x => x.innerText.trim().startsWith('Session'))")
            bottom = int(session) - 8
            await page.call("Emulation.setDeviceMetricsOverride", width=args.width, height=bottom,
                            deviceScaleFactor=args.scale, mobile=False)
            await asyncio.sleep(2)
            shot = await page.call("Page.captureScreenshot", format="png", captureBeyondViewport=True,
                                   clip={"x": 0, "y": 0, "width": args.width, "height": bottom, "scale": 1})
            fig = panels(base64.b64decode(shot["data"]), [(0, cards - 14), (buttons - 14, bottom)], args.scale,
                         label_px=int(22 * args.scale))
            fig.save(args.out)
            print("figure", fig.size)
    finally:
        chrome.terminate()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    ap.add_argument("--out", default="paper/Fig3.png")
    ap.add_argument("--cdp", type=int, default=9333)
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=1500)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--seed", type=int, default=100)
    asyncio.run(main(ap.parse_args()))
