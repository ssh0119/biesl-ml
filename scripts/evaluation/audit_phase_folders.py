"""Compare phase spreadsheet entries with recordings without modifying raw data."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd

from src.io import list_recordings, parse_filename


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit(data_dir, output_dir, aliases=None):
    root, out = Path(data_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    aliases = aliases or {}
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    book = root / "Phase separation.xlsx"
    with zipfile.ZipFile(book) as z:
        shared = ["".join(si.itertext()) for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si",ns)]
        rows = []
        for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).findall(".//m:row",ns):
            cells = {}
            for cell in row.findall("m:c",ns):
                value = cell.find("m:v",ns)
                if value is not None:
                    cells["".join(c for c in cell.get("r") if c.isalpha())] = shared[int(value.text)] if cell.get("t")=="s" else value.text
            if cells.get("B") and cells["B"] != "File name":
                rows.append({"spreadsheet_name":cells["B"], "name":aliases.get(cells["B"],cells["B"]),
                             "phase1":cells["C"].replace(" ","").lower(), "phase2":cells["D"].replace(" ","").lower()})
    files = [Path(p) for p in list_recordings(str(root))]
    actual = {p.stem:p for p in files}
    errors=[]
    if len(actual)!=len(files): errors.append("Duplicate recording stems")
    if len({r['name'] for r in rows})!=len(rows): errors.append("Duplicate spreadsheet names")
    records=[]
    for row in rows:
        path=actual.get(row["name"])
        item={**row,"actual_path":str(path) if path else None,"match":False}
        if row["phase1"] != "rest": errors.append(f"Non-rest Phase 1: {row['name']}")
        if path is None:
            errors.append(f"Spreadsheet entry missing file: {row['name']}")
        elif path.parent.name.lower()!=row["phase2"]:
            errors.append(f"Folder mismatch: {row['name']}")
        else:
            item["match"]=True
        records.append(item)
    extra=sorted(set(actual)-{r["name"] for r in rows})
    errors.extend(f"File missing spreadsheet entry: {name}" for name in extra)
    manifest=[]
    for p in files:
        metadata=parse_filename(str(p))
        manifest.append({"path":str(p),"name":p.stem,"phase2":p.parent.name.lower(),"sha256":sha256(p),"plank_seconds":metadata[3]})
    hashes=Counter(m["sha256"] for m in manifest)
    duplicates=[m["name"] for m in manifest if hashes[m["sha256"]]>1]
    if duplicates: errors.append("Byte-identical recordings would leak across folds: " + ", ".join(duplicates))
    previous = pd.read_csv("output_new_1/features.csv")
    old = previous[previous.activity != "rest"].groupby("rec_name").activity.first().to_dict()
    changes=[]
    for m in manifest:
        label=old.get(m["name"])
        if label is None: changes.append({"name":m["name"],"change":"added"})
        else:
            if label!=m["phase2"]:
                changes.append({"name":m["name"],"change":"label_changed","old":label,"new":m["phase2"]})
    changes.extend({"name":name,"change":"removed"} for name in sorted(set(old)-set(actual)))
    result={"data_dir":str(root),"spreadsheet_sha256":sha256(book),"aliases":aliases,"n_recordings":len(files),
            "counts":dict(Counter(p.parent.name for p in files)),"errors":errors,"records":records,"manifest":manifest,"changes_from_data":changes}
    (out/"data_audit.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    text="# data_new 분류 점검\n\n"+f"녹화 {len(files)}개 / 표 {len(rows)}개. 정확한 파일명 또는 명시적으로 확인된 별칭으로 대조했습니다.\n\n"
    text+="\n".join(f"- {k}: {v}개" for k,v in result["counts"].items())
    text+="\n\n## 점검 결과\n\n"+("분류·파일 대응 모두 일치하며 동일 내용의 중복 파일이 없습니다." if not errors else "\n".join("- "+e for e in errors))
    text+="\n\n## 파일명 별칭\n\n"+json.dumps(aliases,ensure_ascii=False,indent=2)
    text+="\n\n## 이전 data와 차이\n\n```json\n"+json.dumps(changes,ensure_ascii=False,indent=2)+"\n```\n"
    (out/"DATA_AUDIT.md").write_text(text,encoding="utf-8")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",default="data_new")
    parser.add_argument("--output-dir",default="output_data_new")
    parser.add_argument("--aliases")
    args=parser.parse_args()
    aliases=json.loads(Path(args.aliases).read_text(encoding="utf-8")) if args.aliases else {}
    result=audit(args.data_dir,args.output_dir,aliases)
    print(json.dumps({k:result[k] for k in ("n_recordings","counts","errors","changes_from_data")},indent=2,ensure_ascii=False))
