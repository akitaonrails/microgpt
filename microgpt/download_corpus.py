"""Download pinned, size-bounded source shards to the configured NAS dataset root."""
import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shutil

from huggingface_hub import HfApi, hf_hub_download

SOURCES = {
    "opencodeinstruct": ("nvidia/OpenCodeInstruct", "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d", ["data/train-*.parquet"]),
    "code-feedback": ("m-a-p/Code-Feedback", "f411b16a97c910ac9acf8b0d0948e340aa77cc34", ["Code-Feedback.jsonl"]),
    "opencoder-educational": ("OpenCoder-LLM/opc-sft-stage2", "7d28f40d579edd7c24402d17d0c7639f991e6f8d", ["educational_instruct/train-*.parquet"]),
    "reasoning-code": ("open-r1/Mixture-of-Thoughts", "e55fa28006c0d0ec60fb3547520f775dd42d02cd", ["code/train-*.parquet"]),
    "reasoning-math": ("open-r1/Mixture-of-Thoughts", "e55fa28006c0d0ec60fb3547520f775dd42d02cd", ["math/train-*.parquet"]),
    "cosmopedia": ("HuggingFaceTB/smollm-corpus", "3ba9d605774198c5868892d7a8deda78031a781f", ["cosmopedia-v2/train-*.parquet"]),
    "fineweb": ("HuggingFaceTB/smollm-corpus", "3ba9d605774198c5868892d7a8deda78031a781f", ["fineweb-edu-dedup/train-*.parquet"]),
    "smoltalk": ("HuggingFaceTB/smol-smoltalk", "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc", ["data/train-*.parquet"]),
    "opencoder": ("OpenCoder-LLM/opc-sft-stage1", "1bcab575f5e2d1c1fd6652720418524c27b3d58b", ["data/largescale_diverse_instruct-*.parquet"]),
    "magicoder": ("ise-uiuc/Magicoder-OSS-Instruct-75K", "5f839b1f368a76b161028bb9edff055db34022b2", ["data-oss_instruct-decontaminated.jsonl"]),
    "swesmith": ("SWE-bench/SWE-smith-trajectories", "08e109b4a59eaeebf80e4675cd125d42e7ac99a4", ["data/tool-*.parquet"]),
    "codeparrot": ("codeparrot/codeparrot-clean", "35a59fb025bc0a102f7d96eac09d145b896d487b", ["file-*.json.gz"]),
    "commitpack": ("bigcode/commitpackft", "fc56fe33c030c6daa414c2b112c932b8eed085e6", ["data/python/data.jsonl", "data/ruby/data.jsonl", "data/typescript/data.jsonl"]),
    "conversation": ("HuggingFaceTB/smoltalk2", "fc6cc2103c066455aade5d7fbb346039ae36ca5e", ["SFT/smoltalk_smollm3_everyday_conversations_no_think-*.parquet"]),
    "multilingual": ("HuggingFaceTB/smoltalk2", "fc6cc2103c066455aade5d7fbb346039ae36ca5e", ["SFT/smoltalk_multilingual_8languages_lang_5_no_think-*.parquet"]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--max-gb", type=float, default=2, help="Maximum total selected source shard bytes (decimal GB)")
    parser.add_argument("--max-files", type=int, default=4)
    args = parser.parse_args()
    root = Path(os.environ.get("MICROGPT_DATA_ROOT", "/mnt/gigachad/microgpt/dataset"))
    if not Path("/mnt/gigachad").is_mount() and root.is_relative_to("/mnt/gigachad"):
        parser.error("NAS not mounted; refusing to fill local disk")
    if min(args.max_gb, args.max_files) <= 0:
        parser.error("Limits must be positive")
    name, revision, patterns = SOURCES[args.source]
    info = HfApi().dataset_info(name, revision=revision, files_metadata=True)
    candidates = sorted((x for x in info.siblings if any(fnmatch.fnmatch(x.rfilename, p) for p in patterns)), key=lambda x: x.rfilename)
    selected = []
    size = 0
    for file in candidates:
        if file.size is None:
            raise ValueError("Missing file size")
        if len(selected) >= args.max_files or size + file.size > args.max_gb * 1e9:
            break
        selected.append(file)
        size += file.size
    if not selected:
        parser.error("No complete shard fits the budget")
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 2 * size + 1_000_000_000:
        parser.error("Insufficient space for downloads and temporary files")
    destination = root / "raw" / args.source
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": name, "revision": revision, "bytes": size, "files": []}
    # local_dir stores downloaded shards directly, without a duplicate hub cache.
    for file in selected:
        path = Path(hf_hub_download(name, file.rfilename, repo_type="dataset", revision=revision,
                                   local_dir=destination))
        with path.open("rb") as source:
            checksum = hashlib.file_digest(source, "sha256").hexdigest()
        if path.stat().st_size != file.size:
            raise ValueError("Download size mismatch")
        manifest["files"].append({"path": str(path.relative_to(destination)), "bytes": file.size, "sha256": checksum})
        print(f"Downloaded {args.source}: {file.rfilename} ({file.size / 1e6:.1f} MB)", flush=True)
    hf_hub_download(name, "README.md", repo_type="dataset", revision=revision, local_dir=destination)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
