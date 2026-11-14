"""Build and publish the public WSDM research artifact as a static Hub Space."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.parse import quote
from urllib.request import Request, urlopen
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
REPO = "DQSWORDMAN/rag-decision-sufficiency"
GITHUB = "https://github.com/dqswordman/rag-decision-sufficiency"
FROZEN_VERSION = "v1.0.0"
MIRRORED_ROOTS = (
    "README.md", "LICENSE", "CITATION.cff", "DATA_PROVENANCE.md", "requirements.txt",
    "code", "data", "docs", "frozen", "reference",
)
CASE_NAMES = ("king-arthur", "ornans")
POLICIES = {"BO9": "Best measured (BO9)", "P1": "Additive (P1)", "P1-check": "P1 + one check"}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def public_bytes(url, max_bytes):
    headers = {"User-Agent": "wsdm-cpu-artifact-publisher"}
    with urlopen(Request(url, headers=headers), timeout=30) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"Public artifact exceeded its expected size: {url}")
    return data


def copy_public_files(destination, revision):
    # Export only committed files from the already-public GitHub repo.
    with tempfile.TemporaryFile() as archive:
        archive.write(git("archive", "--format=zip", revision, "--", *MIRRORED_ROOTS, "space"))
        archive.seek(0)
        with ZipFile(archive) as zipped:
            zipped.extractall(destination)
    (destination / "README.md").replace(destination / "GITHUB_README.md")
    for path in (destination / "space").iterdir():
        path.replace(destination / path.name)
    (destination / "space").rmdir()


def prepare_summary(destination):
    passed = subprocess.check_output(
        [sys.executable, str(destination / "code/reproduce.py"), "--verify-inputs"],
        text=True,
    )
    print(passed.strip(), flush=True)
    cases = {}
    for case in CASE_NAMES:
        cases[case] = json.loads(subprocess.check_output(
            [sys.executable, str(destination / "code/inspect_case.py"), "--case", case, "--json"]
        ))
    with (destination / "reference/results/selection_overall.csv").open(encoding="utf-8", newline="") as handle:
        archived = {
            row["method"]: row
            for row in csv.DictReader(handle)
            if row["budget"] == "3" and row["method"] in POLICIES
        }
    assert all(int(archived[method]["n"]) == 1440 for method in POLICIES)
    comparison = [
        {
            "name": label,
            "mean_regret": float(archived[method]["mean_regret"]),
            "mean_calls": float(archived[method]["mean_probes"]),
        }
        for method, label in POLICIES.items()
    ]
    write_json(destination / "summary.json", {"cases": cases, "comparison": comparison})


def prepare_release(destination, release_report):
    manifest = json.loads((destination / "docs/release-manifest.json").read_text(encoding="utf-8"))
    version = FROZEN_VERSION
    if manifest["artifact_version"] != version:
        raise ValueError("Update the Space copy and release link together with a new frozen version.")
    release_commit = git("rev-parse", f"refs/tags/{version}^{{commit}}").decode().strip()
    if release_commit != manifest["code"]["commit"]:
        raise ValueError("The release tag does not match the published source commit.")
    name = f"rag-decision-sufficiency-{version}.zip"
    archive = destination / name
    with archive.open("wb") as stream:
        subprocess.run(
            ["git", "archive", "--format=zip", f"--prefix=rag-decision-sufficiency-{version}/", release_commit],
            cwd=ROOT, stdout=stream, check=True,
        )
    if archive.stat().st_size > 100_000_000:
        raise ValueError("The frozen release ZIP exceeds 100,000,000 bytes.")
    with ZipFile(archive) as zipped:
        if zipped.testzip() or zipped.comment.decode("ascii") != release_commit:
            raise ValueError("The frozen release ZIP is corrupt or lacks its source commit.")

    expected_url = f"{GITHUB}/releases/download/{version}/reproduction_report.json"
    if manifest["verification"]["published_record"] != expected_url:
        raise ValueError("Unexpected source for the published verification.")
    if release_report is not None:
        report_content = release_report.read_bytes()
    else:
        report_content = public_bytes(expected_url, 100_000)
    report = json.loads(report_content)
    expected = manifest["verification"]
    if (
        len(report_content) > 100_000
        or report["status"] != "pass"
        or report["device"] != "CPU"
        or report["inputs_verified"] != expected["input_hashes_verified"]
        or len(report["reference_csvs_matched"]) != expected["reference_csvs_matched"]
        or len(report["tex_tables_matched"]) != expected["tex_tables_matched"]
    ):
        raise ValueError("The published CPU verification does not match the release manifest.")
    (destination / "reproduction_report.json").write_bytes(report_content)
    return {
        "version": version, "github_commit": release_commit, "zip": name,
        "bytes": archive.stat().st_size, "sha256": sha256(archive),
    }


def build(destination, release_report):
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError(f"Use an empty build directory: {destination}")
    source_commit = git("rev-parse", "HEAD").decode().strip()
    copy_public_files(destination, source_commit)
    prepare_summary(destination)
    release = prepare_release(destination, release_report)
    files = {
        path.relative_to(destination).as_posix(): sha256(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    write_json(destination / "SNAPSHOT.json", {
        "github_repository": GITHUB, "github_commit": source_commit,
        "frozen_release": release, "sha256": files,
    })
    print(f"Prepared {len(files) + 1} files from {source_commit[:12]}. Frozen ZIP: {release['bytes']} bytes.")
    return source_commit


def publish(destination, source_commit):
    from huggingface_hub import HfApi

    api = HfApi()
    upstream_head = git("ls-remote", "--heads", GITHUB + ".git", "refs/heads/main").decode().split()
    if not upstream_head:
        raise ValueError("Could not verify the public GitHub main branch.")
    if upstream_head[0] != source_commit:
        print("GitHub main advanced after checkout. Skipping the older snapshot.")
        return
    previous = api.space_info(REPO)
    if previous.private or previous.sdk != "static":
        raise ValueError("Expected a public Static Space.")
    commit = api.upload_folder(
        repo_id=REPO, repo_type="space", folder_path=destination,
        revision="main", parent_commit=previous.sha, delete_patterns=["*", "**/*"],
        ignore_patterns=[".gitattributes"],
        commit_message=f"Mirror GitHub {source_commit[:12]}",
    )
    expected = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
    actual = set(api.list_repo_files(REPO, repo_type="space", revision=commit.oid, token=False))
    if actual - {".gitattributes"} != expected:
        raise ValueError("The published file tree does not match the staged mirror.")
    base = f"https://huggingface.co/spaces/{REPO}/resolve/{commit.oid}"
    for filename in ("SNAPSHOT.json", "index.html", "summary.json"):
        expected_bytes = (destination / filename).read_bytes()
        if public_bytes(f"{base}/{quote(filename)}", len(expected_bytes)) != expected_bytes:
            raise ValueError(f"The public Space served unexpected bytes for {filename}.")
    print(commit.commit_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--release-report", type=Path, help="Reuse a local copy of the published verification.")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    source_commit = build(args.destination.resolve(), args.release_report)
    if args.publish:
        publish(args.destination.resolve(), source_commit)
