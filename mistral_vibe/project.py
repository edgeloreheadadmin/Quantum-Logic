"""
project.py — Project Management System

Projects are named workspaces that combine:
  • INSTRUCTIONS.md  — custom system prompt (injected into every LLM call)
  • File context     — contents of a folder tree included as conversation context
  • Metadata         — creation time, active model override, tags

Folder ingestor
───────────────
  /load <path>   reads an entire directory tree (with .gitignore-style filtering)
  and stores the file tree + contents both on RAM disk and in the Project object.
  Contents are injected automatically into the system prompt of every subsequent
  Mistral call while the project is active.

Commands (parsed by app.py)
────────────────────────────
  /project new <name>           create a new project
  /project load <name>          activate a saved project
  /project list                 list all projects
  /load <folder-path>           ingest folder into the active project (or new one)
  /instructions <text…>         set system instructions for active project
  /unload                       deactivate the active project
"""

from __future__ import annotations

import fnmatch
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .core import SystemBus, VirtualRAMDisk

# ─── Filtering ────────────────────────────────────────────────────────────────

_IGNORE = {
    # VCS / tooling
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "node_modules", ".yarn", ".pnp.js",
    # Build / dist
    "dist", "build", "out", ".next", ".nuxt", ".output",
    "target", "*.egg-info", "*.egg", "site-packages",
    # Env / secrets
    ".env", ".env.*", "*.pem", "*.key", "*.p12",
    "venv", ".venv", "env", ".env.local",
    # Artifacts / binary
    "*.pyc", "*.pyo", "*.so", "*.o", "*.a", "*.lib", "*.dll", "*.exe",
    "*.bin", "*.dat", "*.db", "*.sqlite3",
    # Misc
    ".DS_Store", "Thumbs.db", "*.lock", "coverage", ".coverage",
    "*.min.js", "*.min.css",
}

_MAX_FILE_KB = 100       # skip files larger than this
_MAX_TOTAL_KB = 400      # stop ingesting once we've hit this limit
_TEXT_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".scss",
    ".json", ".jsonc", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".md", ".txt", ".rst", ".sh", ".bash", ".zsh", ".fish",
    ".cpp", ".c", ".h", ".hpp", ".rs", ".go", ".java", ".rb",
    ".swift", ".kt", ".cs", ".php", ".sql", ".graphql", ".proto",
    ".dockerfile", ".tf", ".hcl", ".xml", ".svg",
}


def _should_skip(p: Path) -> bool:
    name = p.name
    return any(fnmatch.fnmatch(name, pat) for pat in _IGNORE)


def _lang(p: Path) -> str:
    return {
        ".py": "python", ".js": "javascript", ".jsx": "jsx",
        ".ts": "typescript", ".tsx": "tsx", ".html": "html",
        ".css": "css", ".scss": "scss", ".json": "json",
        ".md": "markdown", ".yaml": "yaml", ".yml": "yaml",
        ".toml": "toml", ".sh": "bash", ".bash": "bash",
        ".cpp": "cpp", ".c": "c", ".h": "c", ".rs": "rust",
        ".go": "go", ".java": "java", ".rb": "ruby",
        ".sql": "sql", ".graphql": "graphql",
    }.get(p.suffix.lower(), "text")


# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class ProjectFile:
    rel_path: str
    content: str
    language: str
    size_kb: float


@dataclass
class Project:
    name: str
    root: Path
    instructions: str = ""
    files: List[ProjectFile] = field(default_factory=list)
    tree: str = ""
    created: float = field(default_factory=time.time)
    model_override: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # ── Derived props ──

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def total_kb(self) -> float:
        return sum(f.size_kb for f in self.files)

    @property
    def has_context(self) -> bool:
        return bool(self.files or self.instructions)

    # ── Context builder ───────────────────────────────────────────────────────

    def build_system_prompt(self, base_prompt: str) -> str:
        """Prepend project instructions and file context to the base system prompt."""
        parts: List[str] = []

        if self.instructions.strip():
            parts.append(f"# Project: {self.name}\n\n{self.instructions.strip()}")

        if self.files:
            parts.append(f"## File context ({self.file_count} files, {self.total_kb:.1f} KB)")
            if self.tree:
                parts.append(f"### Directory structure\n```\n{self.tree}\n```")
            for f in self.files:
                parts.append(f"### {f.rel_path}\n```{f.language}\n{f.content}\n```")

        if parts:
            return "\n\n".join(parts) + "\n\n---\n\n" + base_prompt
        return base_prompt

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_instructions(self) -> None:
        (self.root / "INSTRUCTIONS.md").write_text(self.instructions)

    def load_instructions(self) -> None:
        p = self.root / "INSTRUCTIONS.md"
        if p.exists():
            self.instructions = p.read_text(errors="replace")


# ─── Folder Ingestor ─────────────────────────────────────────────────────────

class FolderIngestor:
    """Recursively reads a folder into a list of ProjectFile objects."""

    def ingest(
        self,
        folder: Path,
        max_depth: int = 6,
    ) -> Tuple[List[ProjectFile], str]:
        """
        Returns (files, tree_string).
        Skips binary files, large files, and ignored paths.
        """
        files: List[ProjectFile] = []
        tree_lines: List[str] = [f"{folder.name}/"]
        total_kb = 0.0

        def _walk(path: Path, depth: int, prefix: str) -> None:
            nonlocal total_kb
            if depth > max_depth:
                return
            try:
                entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            except PermissionError:
                return

            for i, entry in enumerate(entries):
                if _should_skip(entry):
                    continue
                is_last = i == len(entries) - 1
                branch = "└── " if is_last else "├── "
                child_prefix = prefix + ("    " if is_last else "│   ")
                tree_lines.append(f"{prefix}{branch}{entry.name}")

                if entry.is_dir():
                    _walk(entry, depth + 1, child_prefix)
                elif entry.is_file():
                    if total_kb >= _MAX_TOTAL_KB:
                        continue
                    if entry.suffix.lower() not in _TEXT_EXTS:
                        continue
                    size_kb = entry.stat().st_size / 1024
                    if size_kb > _MAX_FILE_KB:
                        continue
                    try:
                        content = entry.read_text(encoding="utf-8", errors="replace")
                        rel = str(entry.relative_to(folder))
                        files.append(ProjectFile(
                            rel_path=rel,
                            content=content,
                            language=_lang(entry),
                            size_kb=round(size_kb, 2),
                        ))
                        total_kb += size_kb
                    except Exception:
                        pass

        _walk(folder, 0, "")
        return files, "\n".join(tree_lines)


# ─── Project Manager ──────────────────────────────────────────────────────────

PROJECTS_DIR = Path("./projects")


class ProjectManager:
    """
    CRUD for Projects.  The active project's context is automatically
    injected into every Mistral API call via build_system_prompt().
    """

    def __init__(self, bus: SystemBus, ramdisk: VirtualRAMDisk) -> None:
        self.bus = bus
        self.ramdisk = ramdisk
        self.ingestor = FolderIngestor()
        self._projects: Dict[str, Project] = {}
        self.active: Optional[Project] = None
        PROJECTS_DIR.mkdir(exist_ok=True)
        self._scan()

    # ── Init ──────────────────────────────────────────────────────────────────

    def _scan(self) -> None:
        """Load projects already saved on disk."""
        for p in sorted(PROJECTS_DIR.iterdir()):
            if p.is_dir() and not p.name.startswith("."):
                proj = Project(name=p.name, root=p)
                proj.load_instructions()
                self._projects[p.name] = proj

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def new(self, name: str, instructions: str = "") -> Project:
        root = PROJECTS_DIR / name
        root.mkdir(parents=True, exist_ok=True)
        proj = Project(name=name, root=root, instructions=instructions)
        if instructions:
            proj.save_instructions()
        self._projects[name] = proj
        self.active = proj
        self.bus.publish("project.created", {"name": name})
        return proj

    def activate(self, name: str) -> Optional[Project]:
        proj = self._projects.get(name)
        if proj:
            self.active = proj
            self.bus.publish("project.activated", {"name": name})
        return proj

    def deactivate(self) -> None:
        old = self.active.name if self.active else None
        self.active = None
        self.bus.publish("project.deactivated", {"name": old})

    def set_instructions(self, instructions: str) -> bool:
        if not self.active:
            return False
        self.active.instructions = instructions
        self.active.save_instructions()
        self.bus.publish("project.instructions_set", {
            "name": self.active.name,
            "length": len(instructions),
        })
        return True

    # ── Folder ingestion ──────────────────────────────────────────────────────

    def load_folder(
        self,
        folder_path: str,
        project_name: Optional[str] = None,
        max_depth: int = 6,
    ) -> Tuple[Project, str]:
        """
        Ingest a folder tree.  Creates a new project (or reuses existing).
        Returns (project, summary_string).
        """
        folder = Path(folder_path).expanduser().resolve()
        if not folder.exists():
            raise FileNotFoundError(f"Path not found: {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder}")

        name = project_name or folder.name
        proj = self._projects.get(name) or self.new(name)

        files, tree = self.ingestor.ingest(folder, max_depth=max_depth)
        proj.files = files
        proj.tree = tree
        self.active = proj

        # Cache in RAM disk
        self.ramdisk.write(f"/projects/{name}/tree.txt", tree, tags=["project"])
        if files:
            ctx = proj.build_system_prompt("")
            self.ramdisk.write(f"/projects/{name}/context.txt",
                               ctx[:200_000], tags=["project", "context"])

        summary = (
            f"Loaded **{len(files)} files** from `{folder}`\n"
            f"Total context: **{proj.total_kb:.1f} KB**\n"
            f"Project **{name}** is now active."
        )
        self.bus.publish("project.folder_loaded", {
            "name": name,
            "files": len(files),
            "kb": proj.total_kb,
            "folder": str(folder),
        })
        return proj, summary

    # ── System prompt injection ───────────────────────────────────────────────

    def get_system_prompt(self, base: str) -> str:
        """Return the effective system prompt (with project context if active)."""
        if self.active and self.active.has_context:
            return self.active.build_system_prompt(base)
        return base

    # ── Listing helpers ───────────────────────────────────────────────────────

    def list_all(self) -> List[Project]:
        return sorted(self._projects.values(), key=lambda p: p.created, reverse=True)

    def stats(self) -> dict:
        return {
            "count": len(self._projects),
            "active": self.active.name if self.active else None,
            "active_files": self.active.file_count if self.active else 0,
            "active_kb": round(self.active.total_kb, 1) if self.active else 0.0,
            "has_instructions": bool(
                self.active and self.active.instructions.strip()
            ) if self.active else False,
        }
