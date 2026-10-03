from __future__ import annotations

import shutil

import pytest
import yaml

from bge_obs.service import CommandError

from .conftest import ML, PY, write


def test_index_incremental(svc, vault):
    r = svc.index_cmd()
    assert r["ok"] and r["added"] == 10 and r["notes_total"] == 10
    assert r["images"]["total"] == 1
    r = svc.index_cmd()
    assert r["added"] == 0 and r["unchanged"] == 10
    (vault / "生活/料理/cook0.md").unlink()
    write(vault / "生活/料理/cook1.md", "# 改過了\n\n完全不同的內容。")
    r = svc.index_cmd()
    assert r["removed"] == 1 and r["updated"] == 1


def test_analysis_requires_index(svc, vault):
    with pytest.raises(CommandError) as e:
        svc.classify(str(vault / "學習/機器學習/ml0.md"))
    assert e.value.code == "index_empty"


def test_ingest_new_ml_note(svc, vault):
    svc.index_cmd()
    new = write(vault / "_inbox/new.md", f"# 新筆記\n\n{ML}\n\n額外說明：交叉驗證。")
    res = svc.ingest([str(new)])["results"][0]
    assert res["ok"], res
    assert res["in_vault"] == "_inbox/new.md"
    assert res["classify"]["top"][0]["folder"] == "學習/機器學習"
    assert res["similar"][0]["path"].startswith("學習/機器學習/")
    assert any(d["kind"] in ("duplicate", "near_duplicate", "contained_in") for d in res["dedup"])
    assert [t["tag"] for t in res["tags"]["tags"]][:1] == ["ml"]  # from vault usage (no tags.yaml)
    assert res["tags"]["tags_file_missing"] is True


def test_exact_duplicate_and_self_exclusion(svc, vault):
    svc.index_cmd()
    src = vault / "程式/Python/py0.md"
    copy = vault / "_inbox/py0 copy.md"
    copy.parent.mkdir(exist_ok=True)
    shutil.copy(src, copy)
    r = svc.dedup(str(copy))
    assert r["dedup"][0] == {"kind": "exact", "target": "程式/Python/py0.md", "score": 1.0, "evidence": []}
    # A file already in the vault must not match itself.
    r = svc.dedup(str(src))
    assert all(d["target"] != "程式/Python/py0.md" for d in r["dedup"])


def test_dedup_all_finds_pairs(svc, vault):
    shutil.copy(vault / "程式/Python/py0.md", vault / "程式/py0-dup.md")
    svc.index_cmd()
    r = svc.dedup(all_=True)
    assert any(set(g) == {"程式/Python/py0.md", "程式/py0-dup.md"} for g in r["exact_groups"])
    assert r["counts"]["pairs"] >= 1


def test_redundant_cross_vault(svc, cfg, vault):
    cfg.thresholds["chunk_match"] = 0.85  # the n-gram fake embedder scores lower than the real model
    svc.index_cmd()
    new = write(vault / "_inbox/mixed.md", f"# 混合\n\n## A\n\n{PY}\n\n## B\n\n全新的段落，談論天文望遠鏡的光學設計與鏡片研磨。")
    r = svc.redundant(str(new))
    assert any(c["found_in"].startswith("程式/Python/") for c in r["cross_vault"])
    assert 0 < r["redundant_ratio"] < 1


def test_tags_from_tag_list(svc, cfg, vault):
    svc.index_cmd()
    cfg.tags_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.tags_path.write_text(yaml.safe_dump({"python": PY[:40], "cooking": "料理 食譜 番茄炒蛋"},
                                            allow_unicode=True), encoding="utf-8")
    r = svc.tags(str(vault / "程式/Python/py1.md"))
    assert r["tags"][0]["tag"] == "python"
    assert r["existing"] == ["python"]


def test_suggest_tags_clusters(svc):
    svc.index_cmd()
    r = svc.suggest_tags()
    assert r["clusters"] and r["clusters"][0]["size"] >= 3


def test_similar_text_and_image(svc, vault):
    svc.index_cmd()
    r = svc.similar(text=COOK_Q)
    assert r["notes"][0]["path"].startswith("生活/料理/")
    r = svc.similar(image=str(vault / "attachments/red.png"))
    assert r["images"][0]["path"] == "attachments/red.png"
    assert r["images"][0]["referenced_by"] == ["學習/機器學習/with_img.md"]


COOK_Q = "番茄炒蛋怎麼做？先炒蛋再炒番茄"


def test_extract_cmd(svc, vault):
    r = svc.extract_cmd(str(vault / "學習/機器學習/ml0.md"))
    assert r["kind"] == "md" and r["tags"] == ["ml"] and "監督式學習" in r["text"]
