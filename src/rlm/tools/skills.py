"""Skill discovery helpers."""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import yaml

from rlm.mcp import list_skill_modules

logger = logging.getLogger(__name__)

_FRONTMATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.S)


SKILLS_ROOT = Path(os.environ.get("RLM_HOME") or Path.home() / ".rlm") / "skills"


def _find_skills_dir() -> Path | None:
    """Locate the skills root, ``$RLM_HOME/skills``."""
    return SKILLS_ROOT if SKILLS_ROOT.is_dir() else None


SKILLS_DIR = _find_skills_dir()


def _normalize_skill_name(name: str) -> str:
    """Normalize a discovered skill token to the import/CLI form."""
    return name.replace("-", "_")


def get_installed_skills() -> list[str]:
    """Return installed skill names discovered from distribution metadata."""
    skills: set[str] = set()
    prefix = "rlm-skill-"
    for dist in metadata.distributions():
        name = dist.metadata.get("Name", "")
        if name.startswith(prefix):
            skills.add(_normalize_skill_name(name[len(prefix) :]))
    return sorted(skills)


@dataclass(frozen=True)
class AuthoredSkill:
    """One package under the contract's persistent skills directory.

    An authored skill is an installed skill minus installation: the same layout,
    ``SKILL.md`` and async ``run()`` entrypoint, imported from ``sys.path`` rather than
    a distribution. ``error`` names the first contract violation found without
    importing; the kernel then binds a placeholder that reports it.
    """

    name: str
    path: str  # sys.path entry that makes ``import <name>`` work
    skill_md: str | None
    error: str | None = None


def _check_authored(package_dir: Path, name: str) -> tuple[str | None, str | None]:
    """``(SKILL.md path, error)`` for one authored package directory."""
    if not (package_dir / "src" / name / "__init__.py").is_file():
        return None, f"expected {package_dir.name}/src/{name}/__init__.py"
    skill_md = package_dir / "SKILL.md"
    if not skill_md.is_file():
        return None, f"expected {package_dir.name}/SKILL.md"
    pyproject = package_dir / "pyproject.toml"
    if pyproject.is_file():
        with open(pyproject, "rb") as f:
            project = tomllib.load(f).get("project", {})
        expected = f"rlm-skill-{package_dir.name}"
        if project.get("name") != expected:
            return str(skill_md), (
                f"{package_dir.name}/pyproject.toml must name the distribution "
                f"{expected!r}"
            )
    return str(skill_md), None


def list_authored_skills(skills_dir: str | Path | None) -> list[AuthoredSkill]:
    """Agent-authored packages under ``skills_dir``, one per subdirectory whose name
    is an identifier (hyphens map to underscores, as for installed skills)."""
    if skills_dir is None:
        return []
    root = Path(skills_dir)
    if not root.is_dir():
        return []
    authored: list[AuthoredSkill] = []
    for package_dir in sorted(root.iterdir()):
        name = _normalize_skill_name(package_dir.name)
        if not package_dir.is_dir() or not name.isidentifier():
            continue
        skill_md, error = _check_authored(package_dir, name)
        authored.append(AuthoredSkill(name, str(package_dir / "src"), skill_md, error))
    return authored


def discover_skills(
    session_dir: Path | None = None, skills_dir: str | Path | None = None
) -> list[str]:
    """Return unambiguous installed, session-local and authored skill module names."""
    installed = get_installed_skills()
    generated = list_skill_modules(session_dir) if session_dir is not None else []
    authored = [skill.name for skill in list_authored_skills(skills_dir)]
    for label, names in (("generated", generated), ("authored", authored)):
        collisions = sorted(set(installed) & set(names))
        if collisions:
            raise ValueError(
                f"skill name collision between installed and {label}: "
                + ", ".join(collisions)
            )
    collisions = sorted(set(generated) & set(authored))
    if collisions:
        raise ValueError(
            "skill name collision between generated and authored: "
            + ", ".join(collisions)
        )
    return [*installed, *generated, *authored]


@dataclass(frozen=True)
class SkillDoc:
    """A SKILL.md skill: listed in the prompt for the agent to read."""

    name: str
    description: str
    location: str


def _frontmatter(skill_md: Path, text: str) -> dict:
    """The SKILL.md frontmatter mapping, or ``{}`` when it is absent or unreadable: one
    malformed skill must not keep the agent from starting."""
    match = _FRONTMATTER.match(text)
    if match is None:
        return {}
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError as error:
        logger.warning("ignoring invalid frontmatter in %s: %s", skill_md, error)
        return {}
    if not isinstance(meta, dict):
        logger.warning("ignoring non-mapping frontmatter in %s", skill_md)
        return {}
    return meta


def _skill_doc(skill_md: Path) -> SkillDoc:
    meta = _frontmatter(skill_md, skill_md.read_text())
    return SkillDoc(
        name=str(meta.get("name") or skill_md.parent.name),
        description=" ".join(str(meta.get("description") or "").split()),
        location=str(skill_md),
    )


def discover_skill_docs(skills_dir: Path) -> list[SkillDoc]:
    """Return the skills under ``skills_dir``: every folder holding a SKILL.md,
    instruction-only skills and Python skill packages alike."""
    return [_skill_doc(skill_md) for skill_md in sorted(skills_dir.glob("*/SKILL.md"))]
