"""
End-to-end demo: upload real JD + 3 resumes, run screening, capture all result views.
Run after the app is already running on http://localhost:8501
"""
import time
from pathlib import Path
from playwright.sync_api import sync_playwright, Page

BASE = "http://localhost:8501"
OUT  = Path("screenshots")
OUT.mkdir(exist_ok=True)

JD_PDF = r"C:\Users\rajbh\Downloads\Cyber-Security-Analyst.pdf"
RESUME_PDFS = [
    r"C:\Users\rajbh\Downloads\EllaWhiteResume.pdf",
    r"C:\Users\rajbh\Downloads\cyber-security-analyst2 - Template 18.pdf",
    r"C:\Users\rajbh\Downloads\cybersecurity-analyst-11495.pdf",
]

VIEWPORT = {"width": 1440, "height": 900}


# ─── helpers ─────────────────────────────────────────────────────────────────

def wait_content(page: Page, text="HireFlow AI", timeout=120):
    print("  waiting for content…")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            page.wait_for_selector(f'text="{text}"', timeout=3000)
            time.sleep(2)
            return
        except Exception:
            time.sleep(1)
    print("  WARNING: timed out")


def wait_spinner_gone(page: Page, timeout=30):
    """Wait until Streamlit's running spinner disappears (short, for upload acks)."""
    print("  waiting for spinner…")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            page.wait_for_selector(
                '[data-testid="stStatusWidget"]',
                state="hidden",
                timeout=3000,
            )
            time.sleep(1)
            return
        except Exception:
            time.sleep(1)


def wait_results(page: Page, timeout=360):
    """
    Wait until the results view is fully rendered.
    The results page shows the 'Screening Results' header and Overview/Candidates/Analytics tabs.
    We poll for the Overview tab text which only exists on the results page.
    """
    print("  waiting for results (AI scoring in progress…)")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            # Tabs only appear after st.rerun() posts results
            page.wait_for_selector(
                'text="Screening Results"',
                timeout=5000,
            )
            # Give Streamlit one more second to finish painting
            time.sleep(3)
            print("  results ready!")
            return True
        except Exception:
            elapsed = int(time.time() - (deadline - timeout))
            print(f"  still processing… ({elapsed}s)", end="\r")
            time.sleep(3)
    print("\n  WARNING: results timed out — screenshotting anyway")
    return False


def take(page: Page, name: str, full=False, clip=None):
    path = str(OUT / name)
    kwargs = {"full_page": full}
    if clip:
        kwargs["clip"] = clip
    page.screenshot(path=path, **kwargs)
    print(f"  >> saved {name}")


def scroll_main(page: Page, px: int):
    page.evaluate(f"window.scrollTo(0, {px})")
    time.sleep(0.6)


# ─── main flow ────────────────────────────────────────────────────────────────

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport=VIEWPORT)
        page = ctx.new_page()

        # ── 1. Load & settle ──────────────────────────────────────────────────
        print("\n[1] Loading app")
        page.goto(BASE, wait_until="domcontentloaded", timeout=60000)
        wait_content(page)

        # ── 2. Landing — hero + upload zone ──────────────────────────────────
        print("\n[2] Landing screenshot")
        scroll_main(page, 0)
        take(page, "ss_01_landing.png")

        # ── 3. Upload JD ──────────────────────────────────────────────────────
        print("\n[3] Uploading JD PDF")
        jd_inputs = page.locator('section[data-testid="stFileUploaderDropzone"] input[type="file"]')
        # First uploader = JD
        jd_inputs.first.set_input_files(JD_PDF)
        time.sleep(3)   # let Streamlit process the upload
        wait_spinner_gone(page, timeout=30)

        # ── 4. JD Quick Scan visible ──────────────────────────────────────────
        print("\n[4] JD Quick Scan")
        scroll_main(page, 0)
        time.sleep(1)
        take(page, "ss_02_jd_scan.png")

        # ── 5. Upload resumes ─────────────────────────────────────────────────
        print("\n[5] Uploading 3 resumes")
        # Second uploader = Resumes (accepts multiple)
        resume_input = jd_inputs.nth(1)
        resume_input.set_input_files(RESUME_PDFS)
        time.sleep(3)
        wait_spinner_gone(page, timeout=30)

        # ── 6. Both uploaders filled ──────────────────────────────────────────
        print("\n[6] Both uploaders filled")
        scroll_main(page, 0)
        time.sleep(1)
        take(page, "ss_03_ready_to_screen.png")

        # ── 7. Click Analyze ──────────────────────────────────────────────────
        print("\n[7] Clicking Analyze Candidates…")
        analyze_btn = page.locator('button:has-text("Analyze"), button:has-text("Quick Analyze")').first
        analyze_btn.click()
        time.sleep(2)

        # ── 8. Processing spinner ─────────────────────────────────────────────
        print("\n[8] Processing spinner")
        try:
            page.wait_for_selector('[data-testid="stStatusWidget"]', timeout=5000)
            take(page, "ss_04_processing.png")
        except Exception:
            take(page, "ss_04_processing.png")   # grab anyway

        # ── 9. Wait for results (may take up to 5 min with NVIDIA) ───────────
        print("\n[9] Waiting for results (AI processing…)")
        wait_results(page, timeout=360)

        # ── 10. Results — full overview ───────────────────────────────────────
        print("\n[10] Results — overview")
        scroll_main(page, 0)
        take(page, "ss_05_results_overview.png")

        # ── 11. Results — ranked table (scroll down to see scores) ────────────
        print("\n[11] Results — ranked list")
        scroll_main(page, 300)
        take(page, "ss_06_ranked_list.png")

        # ── 12. Candidates tab ────────────────────────────────────────────────
        print("\n[12] Candidates tab")
        scroll_main(page, 0)
        try:
            page.locator('[data-baseweb="tab"]:has-text("Candidates")').first.click()
            time.sleep(1.5)
        except Exception:
            try:
                page.locator('button[role="tab"]:has-text("Candidates")').first.click()
                time.sleep(1.5)
            except Exception:
                pass
        take(page, "ss_07_candidates_tab.png")

        # Scroll down to show first candidate card details
        scroll_main(page, 400)
        time.sleep(0.5)
        take(page, "ss_08_candidate_detail.png")

        # Scroll further to see skills gap / explanation
        scroll_main(page, 900)
        time.sleep(0.5)
        take(page, "ss_09_skills_gap.png")

        # ── 13. Analytics tab ─────────────────────────────────────────────────
        print("\n[13] Analytics tab")
        scroll_main(page, 0)
        try:
            page.locator('[data-baseweb="tab"]:has-text("Analytics")').first.click()
            time.sleep(2)
        except Exception:
            try:
                page.locator('button[role="tab"]:has-text("Analytics")').first.click()
                time.sleep(2)
            except Exception:
                pass
        take(page, "ss_10_analytics.png")
        scroll_main(page, 400)
        time.sleep(0.5)
        take(page, "ss_11_analytics_charts.png")

        # ── 14. Overview tab — pipeline board ─────────────────────────────────
        print("\n[14] Back to Overview")
        scroll_main(page, 0)
        try:
            page.locator('[data-baseweb="tab"]:has-text("Overview")').first.click()
            time.sleep(1)
        except Exception:
            pass
        take(page, "ss_12_overview_tab.png")

        # Full page composite
        scroll_main(page, 0)
        take(page, "ss_13_full_results.png", full=True)

        browser.close()
    print("\nAll screenshots saved to screenshots/")


if __name__ == "__main__":
    run()
