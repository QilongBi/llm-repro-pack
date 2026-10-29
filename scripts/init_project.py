"""Create the recommended project layout. Never overwrites existing files.

Usage: python init_project.py <project_dir>
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import resource_path

LAYOUT_DIRS = ("data", "outputs", "logs", "env")

GITIGNORE = """# llm-repro-pack: keep weights, secrets and local state out of version control
*.safetensors
*.gguf
*.bin
*.pt
.env
.env.*
.llmrp/
"""

ENV_README = """# env/

Put the run-time environment here, best first:

1. Docker image digest (`docker inspect --format '{{index .RepoDigests 0}}' <image> > env/image_digest.txt`)
2. Conda export (`conda env export > env/environment.yml`)
3. Lockfile or `pip freeze > env/requirements.lock`

Also save hardware info from the run machine:

- NVIDIA: `nvidia-smi > env/nvidia-smi.txt`
- Apple Silicon: `system_profiler SPHardwareDataType > env/hardware.txt`
- Any OS: `python -c "import platform; print(platform.platform())" >> env/hardware.txt`

Export these on the machine and at the time of the run when you can. Exports made
later are flagged as post-hoc.
"""


def init(project: Path) -> list[str]:
    project.mkdir(parents=True, exist_ok=True)
    created = []
    for d in LAYOUT_DIRS:
        p = project / d
        if not p.exists():
            p.mkdir()
            created.append(d + "/")
    for name, src in (("prompt.yaml", "templates/project/prompt.yaml"), ("config.yaml", "templates/project/config.yaml"),
                      ("practices.yaml", "templates/practices.yaml")):
        dst = project / name
        if not dst.exists():
            shutil.copyfile(resource_path(src), dst)
            created.append(name)
    extras = {project / "env" / "README.md": ENV_README, project / ".gitignore": GITIGNORE}
    for dst, text in extras.items():
        if not dst.exists():
            dst.write_text(text, encoding="utf-8", newline="\n")
            created.append(dst.relative_to(project).as_posix())
    return created


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", type=Path)
    args = ap.parse_args()
    created = init(args.project)
    if created:
        print("Created: " + ", ".join(created))
    else:
        print("Nothing to create; the layout already exists.")
    print("Layout: prompt.yaml, config.yaml, practices.yaml, data/, outputs/, logs/, env/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
