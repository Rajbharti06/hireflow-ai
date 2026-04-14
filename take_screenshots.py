"""
Screenshot capture script for HireFlow AI README.
Polls until actual app content is visible before capturing.
"""
import time
from playwright.sync_api import sync_playwright

BASE = "http://localhost:8501"
OUT  = "screenshots"
VIEWPORT = {"width": 1440, "height": 900}


def wait_content(page, timeout=90):
    """
    Poll until Streamlit has rendered real content (not just the spinner).
    We look for the HireFlow title or the sidebar usage text — either means
    the Python script finished its first run.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            # The app title appears in both the hero section and the browser tab
            page.wait_for_selector(
                'text="HireFlow AI"',
                timeout=3000,
            )
            # Extra settle: let JS / CSS finish painting
            time.sleep(2)
            return True
        except Exception:
            time.sleep(1)
    print("  WARNING: timed out waiting for content, shooting anyway")
    return False


def take(page, name: str, full=False, clip=None):
    path = f"{OUT}/{name}"
    kwargs = {"full_page": full}
    if clip:
        kwargs["clip"] = clip
    page.screenshot(path=path, **kwargs)
    print(f"  saved {path}")


def sidebar_width(page) -> int:
    """Return the pixel width of the sidebar element."""
    try:
        w = page.evaluate("""
            () => {
                const sb = document.querySelector('[data-testid="stSidebar"]');
                return sb ? sb.getBoundingClientRect().width : 240;
            }
        """)
        return int(w)
    except Exception:
        return 240


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport=VIEWPORT)
        page = ctx.new_page()

        # ── Initial load ──────────────────────────────────────────────────────
        print("Loading app and waiting for content…")
        page.goto(BASE, wait_until="domcontentloaded", timeout=60000)
        wait_content(page, timeout=120)

        sw = sidebar_width(page)
        print(f"  sidebar width detected: {sw}px")

        # ── 01 Full dashboard hero ────────────────────────────────────────────
        print("01 hero / landing")
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(0.3)
        take(page, "01_login.png")

        # ── 02 Sidebar — full strip (all sections) ────────────────────────────
        # Use a tall page screenshot cropped to sidebar width so nothing is cut off
        print("02 full sidebar strip")
        take(page, "02_dashboard.png",
             clip={"x": 0, "y": 0, "width": sw + 10, "height": 900})

        # ── 03 Main content — feature grid ────────────────────────────────────
        print("03 feature cards (main area)")
        take(page, "03_upload_panel.png",
             clip={"x": sw + 10, "y": 0, "width": VIEWPORT["width"] - sw - 10, "height": 900})

        # ── 04 Sidebar scrolled to show AI Backend + History sections ─────────
        print("04 sidebar — AI backend panel")
        # Scroll using keyboard: focus sidebar then Page Down
        try:
            sb_handle = page.locator('[data-testid="stSidebar"]')
            sb_handle.hover()
            page.mouse.wheel(0, 600)   # wheel-scroll sidebar
            time.sleep(1.2)
        except Exception:
            pass
        take(page, "04_features.png",
             clip={"x": 0, "y": 0, "width": sw + 10, "height": 900})

        # ── 05 Full-page overview ─────────────────────────────────────────────
        print("05 full page")
        page.evaluate("window.scrollTo(0, 0)")
        try:
            sb_handle = page.locator('[data-testid="stSidebar"]')
            sb_handle.hover()
            page.mouse.wheel(0, -9999)
            time.sleep(0.5)
        except Exception:
            pass
        take(page, "05_features_bottom.png", full=True)

        browser.close()
    print("All done.")


if __name__ == "__main__":
    run()
