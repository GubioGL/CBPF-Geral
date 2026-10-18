"""Pack the files Overleaf needs into manuscript/manuscript_overleaf.zip."""
import zipfile
from pathlib import Path

MS = Path(__file__).resolve().parents[1]
files = [MS / "main.tex", MS / "references.bib", MS / "figures" / "schematic.tex"]
files += sorted(p for p in (MS / "figures").glob("*.pdf") if p.stem != "schematic_preview")
with zipfile.ZipFile(MS / "manuscript_overleaf.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f.relative_to(MS).as_posix())
print("\n".join(f.relative_to(MS).as_posix() for f in files))
print(f"-> {MS / 'manuscript_overleaf.zip'}")
