#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path


def _require_huggingface_hub():
    try:
        from huggingface_hub import HfApi  # noqa: F401
    except Exception as e:  # pragma: no cover
        print("Missing dependency: huggingface_hub", file=sys.stderr)
        print("Install:", file=sys.stderr)
        print("  python3 -m pip install -U huggingface_hub", file=sys.stderr)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload a local folder to Hugging Face Hub (dataset repo).")
    parser.add_argument(
        "--repo-id",
        required=True,
        help='Dataset repo id, e.g. "yourname/business_card_dataset".',
    )
    parser.add_argument(
        "--folder",
        default="LLaMA-Factory/data/business_card_dataset",
        help="Local folder to upload (default: LLaMA-Factory/data/business_card_dataset).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private (if it does not exist yet).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN", ""),
        help="Hugging Face token (default: HF_TOKEN env var or cached login).",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload business_card_dataset",
        help="Commit message to use on the Hub.",
    )
    args = parser.parse_args()

    _require_huggingface_hub()
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"Folder not found: {folder}", file=sys.stderr)
        return 2

    # Token can be empty if user already ran `huggingface-cli login`.
    api = HfApi(token=args.token or None)

    # Helpful debug: show which account this token belongs to.
    try:
        who = api.whoami()
        # "name" is common; keep fallback generic across hub versions.
        username = who.get("name") if isinstance(who, dict) else str(who)
        if username:
            print(f"Authenticated as: {username}")
    except Exception:
        # We'll fail later with a clearer 401/403 when creating/uploading.
        pass

    # Create dataset repo if needed.
    try:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=bool(args.private),
            exist_ok=True,
        )
    except HfHubHTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status == 403:
            print(
                f"[WARN] Cannot create repo: {args.repo_id} (403 Forbidden). "
                "If the repo already exists, we'll try uploading anyway. "
                "Otherwise check: repo namespace (username/org) and token scope (Write).",
                file=sys.stderr,
            )
        else:
            raise

    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(folder),
        path_in_repo=".",
        commit_message=args.commit_message,
    )

    print(f"Uploaded: {args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
