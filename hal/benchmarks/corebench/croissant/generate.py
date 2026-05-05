"""Generate Croissant 1.0 metadata for the CORE-bench v1.1 mainline and OOD sets.

Reads the task manifests in this directory's parent and writes croissant_*.json
alongside this script. Re-run after editing manifests to keep the metadata in sync.

The output uses HuggingFace placeholder URLs for capsule tarballs and the manifest
itself; replace HF_OWNER / HF_REPO_* below with real values once datasets are
published.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
COREBENCH_DIR = HERE.parent

# ---------------------------------------------------------------------------
# Source manifests
# ---------------------------------------------------------------------------
MAINLINE_JSON = COREBENCH_DIR / "core_test.json.bak.main42"
OOD_JSON = COREBENCH_DIR / "core_test.json"  # currently the OOD set in active use

# ---------------------------------------------------------------------------
# Hosting (replace once datasets are uploaded to HuggingFace / Zenodo)
# ---------------------------------------------------------------------------
HF_OWNER = "agent-evals"
HF_REPO_MAINLINE = "core-bench-v1.1-mainline"
HF_REPO_OOD = "core-bench-v1.1-ood"

# ---------------------------------------------------------------------------
# Fixed metadata shared across both datasets
# ---------------------------------------------------------------------------
LICENSE = "https://creativecommons.org/licenses/by/4.0/"  # confirm before submission
VERSION = "1.1.0"
DATE_PUBLISHED = "2026-05-03"
CITE_AS = (
    "@inproceedings{corebench-v1-1-2026,"
    " title={CORE-bench v1.1: A Reliability Benchmark for AI Agents on Scientific Reproducibility Tasks},"
    " author={TODO},"
    " booktitle={NeurIPS Datasets and Benchmarks Track},"
    " year={2026}"
    "}"
)
CREATOR = {
    "@type": "Organization",
    "name": "TODO: lab/institution name",
    "url": "TODO: project URL",
}
KEYWORDS = [
    "AI agents",
    "scientific reproducibility",
    "benchmark",
    "Code Ocean",
    "computational science",
    "agentic evaluation",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_derived_from(tasks: list) -> list:
    """Return prov:wasDerivedFrom entries.

    The HF RAI checker's UI caps display/export at 5 slots, so we keep this
    list short and dataset-level. Per-capsule DOIs live in the recordSet
    field `capsule_doi` (one DOI per task record), which is the semantically
    correct place for per-record provenance.
    """
    n_dois = sum(1 for t in tasks if (t.get("capsule_doi") or "").strip())
    return [
        {
            "@id": "https://codeocean.com",
            "prov:label": (
                f"Code Ocean compute capsule platform. All {len(tasks)} source "
                "capsules are hosted on Code Ocean; per-capsule DOIs "
                f"({n_dois} of {len(tasks)} tasks have one) are preserved on "
                "each task record in the `capsule_doi` field of the RecordSet."
            ),
        }
    ]


def _build(
    *,
    name: str,
    description: str,
    repo: str,
    manifest_path: Path,
    record_count: int,
    split_label: str,  # e.g. "mainline" or "OOD" — used in RAI text
) -> dict:
    # Manifests are uploaded to HF as `core_test.json` regardless of their
    # local filename (mainline lives at core_test.json.bak.main42 on disk).
    manifest_url = (
        f"https://huggingface.co/datasets/{HF_OWNER}/{repo}/resolve/main/core_test.json"
    )
    capsules_template = (
        f"https://huggingface.co/datasets/{HF_OWNER}/{repo}/resolve/main/capsules/{{capsule_id}}.tar.gz"
    )

    file_object_manifest = {
        "@type": "cr:FileObject",
        "@id": "tasks-json",
        "name": "core_test.json",
        "description": "JSON array of CORE-bench tasks. One element per task.",
        "encodingFormat": "application/json",
        "contentUrl": manifest_url,
        "sha256": _sha256(manifest_path),
    }

    file_set_capsules = {
        "@type": "cr:FileSet",
        "@id": "capsules",
        "name": "capsules",
        "description": (
            "Per-task capsule tarballs. Each tarball contains the original Code Ocean "
            "capsule (code, data, environment, results) renamed to <capsule_id>.tar.gz."
        ),
        "encodingFormat": "application/gzip",
        "includes": "capsules/*.tar.gz",
    }

    fields = [
        ("capsule_id", "Code Ocean capsule identifier (e.g. 'capsule-9026204').", "sc:Text", "$[*].capsule_id"),
        ("capsule_title", "Capsule title from Code Ocean.", "sc:Text", "$[*].capsule_title"),
        ("field", "Scientific discipline (e.g. 'Computer Science').", "sc:Text", "$[*].field"),
        ("language", "Primary programming language used in the capsule.", "sc:Text", "$[*].language"),
        ("task_prompt", "Initial instruction given to the agent describing what to execute.", "sc:Text", "$[*].task_prompt"),
        ("expected_answer", "JSON object mapping question text to ground-truth answer. Multiple identical entries indicate independent reruns of the original capsule.", "sc:Text", "$[*].results"),
        ("capsule_doi", "DOI of the source paper or capsule, if available.", "sc:Text", "$[*].capsule_doi"),
        ("capsule_url", "Direct download URL for the capsule tarball (derived from capsule_id).", "sc:URL", None),
    ]

    record_set = {
        "@type": "cr:RecordSet",
        "@id": "tasks",
        "name": "tasks",
        "description": f"{record_count} CORE-bench evaluation tasks.",
        "field": [
            (
                {
                    "@type": "cr:Field",
                    "@id": f"tasks/{fname}",
                    "name": fname,
                    "description": fdesc,
                    "dataType": ftype,
                    "source": {
                        "fileObject": {"@id": "tasks-json"},
                        "extract": {"jsonPath": fpath},
                    },
                }
                if fpath is not None
                else {
                    # Derived field: not extracted from the source manifest.
                    # Materialized inline in recordSet[0].data via _augment_records.
                    "@type": "cr:Field",
                    "@id": f"tasks/{fname}",
                    "name": fname,
                    "description": fdesc,
                    "dataType": ftype,
                }
            )
            for fname, fdesc, ftype, fpath in fields
        ],
    }

    return {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "citeAs": "cr:citeAs",
            "column": "cr:column",
            "conformsTo": "dct:conformsTo",
            "cr": "http://mlcommons.org/croissant/",
            "rai": "http://mlcommons.org/croissant/RAI/",
            "data": {"@id": "cr:data", "@type": "@json"},
            "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
            "dct": "http://purl.org/dc/terms/",
            "equivalentProperty": "cr:equivalentProperty",
            "examples": {"@id": "cr:examples", "@type": "@json"},
            "prov": "http://www.w3.org/ns/prov#",
            "extract": "cr:extract",
            "field": "cr:field",
            "samplingRate": "cr:samplingRate",
            "fileProperty": "cr:fileProperty",
            "fileObject": "cr:fileObject",
            "fileSet": "cr:fileSet",
            "format": "cr:format",
            "includes": "cr:includes",
            "isLiveDataset": "cr:isLiveDataset",
            "jsonPath": "cr:jsonPath",
            "key": "cr:key",
            "md5": "cr:md5",
            "parentField": "cr:parentField",
            "path": "cr:path",
            "recordSet": "cr:recordSet",
            "references": "cr:references",
            "regex": "cr:regex",
            "repeated": "cr:repeated",
            "replace": "cr:replace",
            "sc": "https://schema.org/",
            "separator": "cr:separator",
            "source": "cr:source",
            "subField": "cr:subField",
            "transform": "cr:transform",
        },
        "@type": "Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "name": name,
        "description": description,
        "version": VERSION,
        "license": LICENSE,
        "url": f"https://huggingface.co/datasets/{HF_OWNER}/{repo}",
        "datePublished": DATE_PUBLISHED,
        "citeAs": CITE_AS,
        "creator": CREATOR,
        "keywords": KEYWORDS,
        "distribution": [file_object_manifest, file_set_capsules],
        "recordSet": [record_set],
        # ── Responsible AI metadata ────────────────────────────────────────
        "rai:dataCollection": (
            "Tasks were derived from publicly available, peer-reviewed Code Ocean "
            "compute capsules covering computational scientific research. For each "
            "capsule, authors of this benchmark executed the original code, "
            "inspected its outputs, and authored natural-language questions whose "
            "ground-truth answers can be derived from the capsule's reproduced "
            "outputs (figures, tables, printed numerics)."
        ),
        "rai:dataCollectionType": ["Manual Human Curator"],
        "rai:dataCollectionRawData": (
            "Source capsules originate from Code Ocean (https://codeocean.com), a "
            "platform for reproducible computational research. Capsule selection "
            "was constrained to capsules that ran end-to-end in a fixed-resource "
            "Linux container without GPUs (with the exception of three capsules in "
            "the mainline set that require a GPU and are distributed for "
            "completeness)."
        ),
        "rai:dataCollectionTimeframeStart": "2025-01-01",
        "rai:dataCollectionTimeframeEnd": "2026-04-30",
        "rai:dataAnnotationProtocol": (
            "For each capsule, annotators executed the capsule three independent "
            "times in its declared environment, recorded the resulting numeric/"
            "categorical outputs, and authored questions whose answers were "
            "stable across reruns (or, where outputs were stochastic, recorded "
            "all three observed values to support fuzzy/numeric tolerance "
            "matching during evaluation)."
        ),
        "rai:dataAnnotationPlatform": "Code Ocean execution environments + manual review",
        "rai:dataAnnotationAnalysis": (
            "Each numeric ground truth is a 95% prediction interval over three "
            "independent capsule reruns. Each non-numeric ground truth is the "
            "deterministic answer across three independent capsule reruns. "
            "Tasks where reruns disagreed beyond expected stochastic variation "
            "were either rejected or rephrased."
        ),
        "rai:dataAnnotationDemographics": (
            "Annotators are graduate-level researchers with a background in "
            "computer science and computational sciences. No personal data about "
            "annotators is collected or distributed."
        ),
        "rai:dataAnnotationPerItem": "1 annotator per task; cross-checked by a second annotator.",
        "rai:dataPreprocessingProtocol": (
            "Capsules are distributed unmodified except that (a) the 'results/' "
            "subdirectory of each capsule is scrubbed at evaluation time to prevent "
            "data leakage to the agent, and (b) each capsule is repackaged as a "
            "single .tar.gz archive named after its Code Ocean capsule ID."
        ),
        "rai:dataUseCases": (
            "Intended for evaluating AI agents on end-to-end scientific "
            "reproducibility tasks: reading capsule code/data, executing it in a "
            "sandboxed environment, and answering questions whose answers depend "
            "on faithfully reproduced outputs. Suitable for outcome-consistency, "
            "calibration, and tool-use studies on agentic systems."
        ),
        "rai:dataLimitations": (
            "Coverage is biased toward capsules that run in CPU-only Linux "
            "environments, which under-represents GPU-heavy ML papers. The "
            "benchmark measures execution + question answering, not novel "
            "scientific reasoning. Ground-truth answers are derived from the "
            "original authors' implementations and inherit any errors therein. "
            "Some capsules involve domain-specific terminology that may "
            "disadvantage agents without scientific pretraining."
        ),
        "rai:dataReleaseMaintenancePlan": (
            "The dataset will be hosted on HuggingFace with versioned releases. "
            "Issues, errata, and capsule additions will be tracked in the "
            "associated GitHub repository. The maintainers commit to keeping the "
            "dataset accessible for at least three years post-publication."
        ),
        "rai:dataSocialImpact": (
            "The benchmark aims to improve evaluation rigor for AI agents in "
            "scientific contexts. Risks include over-fitting agent training to "
            "the included capsules and misuse as a sole proxy for general "
            f"scientific competence. We recommend reporting CORE-bench v1.1 "
            f"{split_label} results alongside other agentic and scientific "
            "benchmarks."
        ),
        "rai:personalSensitiveInformation": (
            "None. All source capsules are publicly licensed Code Ocean "
            "artifacts; no human-subjects data, biometric data, health/medical "
            "data, or personally identifying information about end users is "
            "included. Standard author-attribution metadata (names, "
            "institutional affiliations) is preserved in the original "
            "capsules. None of the following sensitive categories are "
            "represented in the data records: gender, socio-economic status, "
            "geography of subjects, language demographics, age, culture, "
            "experience or seniority, health or medical data, or political "
            "or religious beliefs."
        ),
        "rai:hasSyntheticData": False,
        "prov:wasDerivedFrom": _build_derived_from(json.loads(manifest_path.read_text())),
        "prov:wasGeneratedBy": [
            {
                "prov:type": {"@id": "https://www.wikidata.org/wiki/Q4929239"},
                "prov:label": "Capsule selection and execution",
                "sc:description": (
                    "Authors of the benchmark identified candidate Code Ocean "
                    "capsules, executed each capsule three times in its declared "
                    "container environment, and recorded the resulting outputs "
                    "(figures, tables, printed numerics) for use as ground truth."
                ),
                "prov:atTime": f"{DATE_PUBLISHED}T00:00:00Z",
                "prov:wasAttributedTo": [
                    {
                        "@id": "https://example.org/corebench/agents/benchmark_authors",
                        "prov:label": "Benchmark authors",
                        "sc:description": (
                            "Graduate-level researchers in computer science and "
                            "computational sciences."
                        ),
                    }
                ],
            },
            {
                "prov:type": {"@id": "https://www.wikidata.org/wiki/Q109719325"},
                "prov:label": "Question and ground-truth annotation",
                "sc:description": (
                    "For each capsule, annotators authored natural-language "
                    "questions whose answers depend on the capsule's reproduced "
                    "outputs, and recorded the ground-truth answer derived from "
                    "the median or modal output across three independent reruns. "
                    "A second annotator cross-checked each item."
                ),
                "prov:atTime": f"{DATE_PUBLISHED}T00:00:00Z",
                "prov:wasAttributedTo": [
                    {
                        "@id": "https://example.org/corebench/agents/benchmark_authors",
                        "prov:label": "Benchmark authors",
                        "sc:description": "Graduate-level researchers (annotation + cross-check).",
                    }
                ],
            },
            {
                "prov:type": {"@id": "https://www.wikidata.org/wiki/Q5227332"},
                "prov:label": "Capsule preprocessing for distribution",
                "sc:description": (
                    "Each source capsule is repackaged as a single .tar.gz "
                    "archive named by Code Ocean capsule ID. The 'results/' "
                    "subdirectory of each capsule is scrubbed at evaluation time "
                    "to prevent ground-truth leakage to the agent under test."
                ),
                "prov:atTime": f"{DATE_PUBLISHED}T00:00:00Z",
                "prov:wasAttributedTo": [
                    {
                        "@id": "https://example.org/corebench/agents/benchmark_authors",
                        "prov:label": "Benchmark authors",
                        "sc:description": "Repackaging scripted; no model-generated content.",
                    }
                ],
            },
        ],
        "rai:dataBiases": (
            "Source capsules over-represent disciplines and labs with strong "
            "open-science / Code Ocean adoption (computer science, "
            "computational biology, computational social science). Geographic "
            "and institutional bias toward English-language, North American, "
            "and European research is likely present."
        ),
        "_capsule_url_template": capsules_template,
    }


def _augment_records(crois: dict, manifest_path: Path) -> None:
    """Embed the per-task records inline so the file is self-contained even if
    the manifest URL is not yet live. Validators that walk recordSet.data can
    verify field coverage without fetching the manifest."""
    tasks = json.loads(manifest_path.read_text())
    capsule_url = crois.pop("_capsule_url_template")
    materialized = []
    for t in tasks:
        materialized.append(
            {
                "tasks/capsule_id": t.get("capsule_id", ""),
                "tasks/capsule_title": t.get("capsule_title", ""),
                "tasks/field": t.get("field", ""),
                "tasks/language": t.get("language", ""),
                "tasks/task_prompt": t.get("task_prompt", ""),
                "tasks/expected_answer": json.dumps(t.get("results", []), ensure_ascii=False),
                "tasks/capsule_doi": t.get("capsule_doi", "") or "",
                "tasks/capsule_url": capsule_url.format(capsule_id=t.get("capsule_id", "")),
            }
        )
    # Croissant's data field on a RecordSet is optional; including it makes the
    # file self-validating without network access to the FileObject.
    crois["recordSet"][0]["data"] = materialized


def main() -> None:
    n_main = len(json.loads(MAINLINE_JSON.read_text()))
    n_ood = len(json.loads(OOD_JSON.read_text()))

    mainline = _build(
        name="core-bench-v1.1-mainline",
        description=(
            f"CORE-bench v1.1 mainline split: {n_main} scientific-reproducibility "
            "tasks derived from peer-reviewed Code Ocean compute capsules. Each "
            "task asks an AI agent to execute the capsule in a sandboxed Linux "
            "environment and answer questions whose ground-truth answers depend "
            "on the reproduced outputs. Disjoint from the OOD split."
        ),
        repo=HF_REPO_MAINLINE,
        manifest_path=MAINLINE_JSON,
        record_count=n_main,
        split_label="mainline",
    )
    _augment_records(mainline, MAINLINE_JSON)

    ood = _build(
        name="core-bench-v1.1-ood",
        description=(
            f"CORE-bench v1.1 out-of-distribution split: {n_ood} held-out "
            "scientific-reproducibility tasks. Drawn from disciplines and "
            "capsule shapes not represented in the mainline split, intended for "
            "measuring generalization of AI agents trained or tuned on the "
            "mainline tasks. Disjoint from the mainline split."
        ),
        repo=HF_REPO_OOD,
        manifest_path=OOD_JSON,
        record_count=n_ood,
        split_label="OOD",
    )
    _augment_records(ood, OOD_JSON)

    (HERE / "croissant_mainline.json").write_text(
        json.dumps(mainline, indent=2, ensure_ascii=False) + "\n"
    )
    (HERE / "croissant_ood.json").write_text(
        json.dumps(ood, indent=2, ensure_ascii=False) + "\n"
    )

    print(f"Wrote croissant_mainline.json ({n_main} tasks)")
    print(f"Wrote croissant_ood.json ({n_ood} tasks)")


if __name__ == "__main__":
    main()
