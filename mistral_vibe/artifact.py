"""
artifact.py — Artifact Engine

Detects React / HTML / CSS / Textual-plugin code blocks in LLM responses,
saves them to disk, scaffolds preview environments, and launches local
HTTP servers so the user can see a live preview in their browser.

React strategy
──────────────
  If npm + node are available  → scaffold Vite project, run `npm install && npm run dev`
  Otherwise                    → wrap JSX in a CDN-based HTML stub (Babel standalone)
    and serve with Python's built-in http.server

HTML / CSS strategy
───────────────────
  Save and serve immediately with Python http.server

Plugin strategy (Textual widgets)
──────────────────────────────────
  Save the Python source to disk and store the code in the panel for review.
  (User can copy-paste / execute manually — we never auto-execute plugin code.)
"""

from __future__ import annotations

import http.server
import json
import os
import re
import shutil
import socket
import socketserver
import subprocess
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .core import SystemBus, VirtualRAMDisk

# ─── Availability checks ──────────────────────────────────────────────────────

NPM_AVAILABLE = shutil.which("npm") is not None
NODE_AVAILABLE = shutil.which("node") is not None

# ─── Detection regexes ───────────────────────────────────────────────────────

_RE_REACT = re.compile(r"```(?:jsx?|tsx?|react)\n([\s\S]*?)```", re.MULTILINE)
_RE_HTML  = re.compile(r"```html\n([\s\S]*?)```",                 re.MULTILINE)
_RE_CSS   = re.compile(r"```css\n([\s\S]*?)```",                  re.MULTILINE)
_RE_PLUG  = re.compile(
    r"```python\n([\s\S]*?class\s+\w+\s*\(\s*(?:Static|Widget|App)[^)]*\)[\s\S]*?)```",
    re.MULTILINE,
)

# ─── CDN-based React HTML wrapper (no npm needed) ─────────────────────────────

_REACT_CDN_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Mistral Vibe · Artifact</title>
  <script crossorigin src="https://unpkg.com/react@18/umd/react.development.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0d1117;
      color: #e6edf3;
      min-height: 100vh;
    }}
    #root {{ padding: 1rem; }}
    /* basic github-dark reset for demo components */
    button {{
      background: #21262d; color: #e6edf3; border: 1px solid #30363d;
      padding: .4rem .8rem; border-radius: 6px; cursor: pointer;
    }}
    button:hover {{ background: #30363d; }}
    input, select, textarea {{
      background: #161b22; color: #e6edf3; border: 1px solid #30363d;
      padding: .4rem; border-radius: 6px;
    }}
    a {{ color: #58a6ff; }}
  </style>
</head>
<body>
  <div id="root"></div>
  <script type="text/babel">
{component_code}

// Mount — try default export, then App, then first exported fn
const Root = (typeof module !== 'undefined' && module.exports) ? module.exports.default :
             (typeof App !== 'undefined' ? App :
             (typeof Component !== 'undefined' ? Component : () => <div>Component</div>));

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode><Root /></React.StrictMode>
);
  </script>
</body>
</html>
"""

# ─── Vite scaffold templates ──────────────────────────────────────────────────

_VITE_PACKAGE = {
    "name": "mistral-artifact",
    "private": True,
    "version": "0.0.0",
    "type": "module",
    "scripts": {"dev": "vite", "build": "vite build", "preview": "vite preview"},
    "dependencies": {
        "react": "^18.2.0",
        "react-dom": "^18.2.0",
    },
    "devDependencies": {
        "@vitejs/plugin-react": "^4.3.0",
        "vite": "^5.2.0",
    },
}

_VITE_INDEX_HTML = """\
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Mistral Vibe · Artifact</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
"""

_VITE_CONFIG = """\
import {{ defineConfig }} from 'vite'
import react from '@vitejs/plugin-react'
export default defineConfig({{
  plugins: [react()],
  server: {{ port: {port}, open: true }},
}})
"""

_VITE_MAIN = """\
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
"""

_VITE_CSS = """\
*, *::before, *::after { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #0d1117;
  color: #e6edf3;
  min-height: 100vh;
}
#root { padding: 1rem; }
"""

# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class Artifact:
    aid: str
    kind: str          # "react" | "html" | "css" | "plugin"
    code: str
    path: Path
    port: int = 0
    status: str = "saved"    # saved | building | running | error | stopped
    url: str = ""
    error: str = ""
    created: float = field(default_factory=time.time)
    _proc: Optional[subprocess.Popen] = field(default=None, repr=False)
    _server: Optional[socketserver.TCPServer] = field(default=None, repr=False)

    @property
    def short_code(self) -> str:
        lines = self.code.strip().splitlines()
        preview = "\n".join(lines[:12])
        return preview + ("\n  ..." if len(lines) > 12 else "")

    @property
    def age_str(self) -> str:
        dt = time.time() - self.created
        if dt < 60:
            return f"{dt:.0f}s"
        if dt < 3600:
            return f"{dt/60:.0f}m"
        return f"{dt/3600:.1f}h"

    def stop(self) -> None:
        try:
            if self._proc:
                self._proc.terminate()
            if self._server:
                self._server.shutdown()
        except Exception:
            pass
        self.status = "stopped"


# ─── Port allocation ──────────────────────────────────────────────────────────

def _find_free_port(start: int = 5173) -> int:
    port = start
    while port < 9000:
        with socket.socket() as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                port += 1
    return start


# ─── Artifact Engine ──────────────────────────────────────────────────────────

class ArtifactEngine:
    """
    Core artifact system.

    Usage:
      artifacts = engine.detect_and_create(response_text)
      for art in artifacts:
          print(art.url)   # open in browser
    """

    ARTIFACTS_DIR = Path("./artifacts")

    def __init__(self, bus: SystemBus, ramdisk: VirtualRAMDisk) -> None:
        self.bus = bus
        self.ramdisk = ramdisk
        self.artifacts: Dict[str, Artifact] = {}
        self.ARTIFACTS_DIR.mkdir(exist_ok=True)

    # ── Detection & creation ──────────────────────────────────────────────────

    def detect_and_create(self, response_text: str) -> List[Artifact]:
        """Scan an LLM response; create and return all detected artifacts."""
        created: List[Artifact] = []

        for match in _RE_REACT.finditer(response_text):
            art = self._make_react(match.group(1).strip())
            if art:
                created.append(art)

        if not created:   # fallback: bare HTML
            for match in _RE_HTML.finditer(response_text):
                art = self._make_html(match.group(1).strip())
                if art:
                    created.append(art)

        for match in _RE_PLUG.finditer(response_text):
            art = self._make_plugin(match.group(1).strip())
            if art:
                created.append(art)

        return created

    # ── React artifact ────────────────────────────────────────────────────────

    def _make_react(self, code: str) -> Optional[Artifact]:
        aid = self._gen_id()
        art_dir = self.ARTIFACTS_DIR / aid
        art_dir.mkdir(parents=True, exist_ok=True)
        port = _find_free_port(5173)

        art = Artifact(aid=aid, kind="react", code=code, path=art_dir, port=port)
        self.artifacts[aid] = art
        self.ramdisk.write(f"/artifacts/{aid}/code.jsx", code, tags=["artifact", "react"])

        if NPM_AVAILABLE and NODE_AVAILABLE:
            threading.Thread(target=self._launch_vite, args=(art,), daemon=True).start()
        else:
            # CDN fallback
            html = _REACT_CDN_TEMPLATE.format(component_code=code)
            (art_dir / "index.html").write_text(html)
            self.ramdisk.write(f"/artifacts/{aid}/index.html", html, tags=["artifact"])
            threading.Thread(target=self._serve_html, args=(art,), daemon=True).start()

        self.bus.publish("artifact.created", {"aid": aid, "kind": "react", "port": port})
        return art

    def _launch_vite(self, art: Artifact) -> None:
        art.status = "building"
        src = art.path / "src"
        src.mkdir(exist_ok=True)

        # Write scaffold
        (art.path / "package.json").write_text(json.dumps(_VITE_PACKAGE, indent=2))
        (art.path / "index.html").write_text(_VITE_INDEX_HTML)
        (art.path / "vite.config.js").write_text(_VITE_CONFIG.format(port=art.port))
        (src / "main.jsx").write_text(_VITE_MAIN)
        (src / "index.css").write_text(_VITE_CSS)

        # Ensure export default
        app_code = art.code
        if "export default" not in app_code and "function App" in app_code:
            app_code += "\n\nexport default App;"
        elif "export default" not in app_code and "const App" in app_code:
            app_code += "\n\nexport default App;"
        (src / "App.jsx").write_text(f"import React from 'react'\n{app_code}")

        # npm install
        try:
            proc = subprocess.run(
                ["npm", "install"],
                cwd=art.path,
                capture_output=True,
                timeout=180,
            )
            if proc.returncode != 0:
                art.status = "error"
                art.error = proc.stderr.decode(errors="replace")[:300]
                return
        except Exception as e:
            art.status = "error"
            art.error = str(e)
            return

        # npm run dev
        try:
            art._proc = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=art.path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            time.sleep(2)   # give vite a moment to start
            art.status = "running"
            art.url = f"http://localhost:{art.port}"
            self.bus.publish("artifact.running", {"aid": art.aid, "url": art.url})
            webbrowser.open(art.url)
        except Exception as e:
            art.status = "error"
            art.error = str(e)

    # ── HTML artifact ─────────────────────────────────────────────────────────

    def _make_html(self, code: str) -> Optional[Artifact]:
        aid = self._gen_id()
        art_dir = self.ARTIFACTS_DIR / aid
        art_dir.mkdir(parents=True, exist_ok=True)
        port = _find_free_port(7000)

        # Wrap bare fragments
        if not code.strip().lower().startswith("<!doctype") and "<html" not in code.lower():
            code = (
                "<!doctype html><html><head>"
                "<meta charset='UTF-8'>"
                "<style>body{background:#0d1117;color:#e6edf3;font-family:sans-serif;padding:1rem}</style>"
                "</head><body>" + code + "</body></html>"
            )
        (art_dir / "index.html").write_text(code)

        art = Artifact(aid=aid, kind="html", code=code, path=art_dir, port=port)
        self.artifacts[aid] = art
        self.ramdisk.write(f"/artifacts/{aid}/index.html", code, tags=["artifact", "html"])
        threading.Thread(target=self._serve_html, args=(art,), daemon=True).start()
        self.bus.publish("artifact.created", {"aid": aid, "kind": "html", "port": port})
        return art

    def _serve_html(self, art: Artifact) -> None:
        orig_dir = os.getcwd()

        class _Handler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *_args): pass
            def log_error(self, *_args): pass

        try:
            os.chdir(art.path)
            with socketserver.TCPServer(("", art.port), _Handler) as httpd:
                httpd.allow_reuse_address = True
                art._server = httpd
                art.status = "running"
                art.url = f"http://localhost:{art.port}"
                self.bus.publish("artifact.running", {"aid": art.aid, "url": art.url})
                webbrowser.open(art.url)
                httpd.serve_forever()
        except Exception as e:
            art.status = "error"
            art.error = str(e)
        finally:
            os.chdir(orig_dir)

    # ── Plugin artifact ───────────────────────────────────────────────────────

    def _make_plugin(self, code: str) -> Optional[Artifact]:
        aid = self._gen_id()
        art_dir = self.ARTIFACTS_DIR / aid
        art_dir.mkdir(parents=True, exist_ok=True)
        plugin_path = art_dir / "plugin.py"
        plugin_path.write_text(code)

        art = Artifact(aid=aid, kind="plugin", code=code, path=art_dir, status="saved")
        self.artifacts[aid] = art
        self.ramdisk.write(f"/artifacts/{aid}/plugin.py", code, tags=["artifact", "plugin"])
        self.bus.publish("artifact.created", {"aid": aid, "kind": "plugin"})
        return art

    # ── Control ───────────────────────────────────────────────────────────────

    def stop(self, aid: str) -> bool:
        art = self.artifacts.get(aid)
        if art:
            art.stop()
            return True
        return False

    def open_browser(self, aid: str) -> bool:
        art = self.artifacts.get(aid)
        if art and art.url:
            webbrowser.open(art.url)
            return True
        return False

    def scaffold_plugin_template(self, name: str) -> str:
        """Return a Textual widget template for the user/LLM to fill in."""
        return f'''\
from textual.widgets import Static

class {name}(Static):
    """Auto-generated Mistral Vibe plugin: {name}"""

    def render(self) -> str:
        return "[bold cyan]{name}[/bold cyan]\\n\\nCustom content here."
'''

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _gen_id() -> str:
        return f"art_{int(time.time() * 1000) % 10_000_000:07d}"

    def list_artifacts(self) -> List[Artifact]:
        return sorted(self.artifacts.values(), key=lambda a: a.created, reverse=True)

    def stats(self) -> dict:
        arts = list(self.artifacts.values())
        return {
            "total": len(arts),
            "running": sum(1 for a in arts if a.status == "running"),
            "error": sum(1 for a in arts if a.status == "error"),
            "react": sum(1 for a in arts if a.kind == "react"),
            "html": sum(1 for a in arts if a.kind == "html"),
            "plugin": sum(1 for a in arts if a.kind == "plugin"),
        }
