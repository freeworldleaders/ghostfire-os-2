"""Provision the protected GhostFire approval-command owner token."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agents.approval_tokens import (
    AgentApprovalTokenStore,
    ApprovalTokenError,
    default_approval_token_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Provision or verify a CurrentUser DPAPI-protected "
            "agent approval owner token."
        )
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=default_approval_token_path(),
        help="Protected token file path.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Verify and reuse an existing valid token.",
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="Replace an existing token with a new token.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an existing token without changing it.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    selected_modes = sum(
        (
            bool(arguments.rotate),
            bool(arguments.verify),
        )
    )

    if selected_modes > 1:
        parser.error("--rotate and --verify are mutually exclusive")

    store = AgentApprovalTokenStore(arguments.path)

    try:
        if arguments.verify:
            payload = {
                **store.verify().as_dict(),
                "created": False,
                "rotated": False,
                "secret_exposed": False,
            }
        elif arguments.rotate:
            payload = store.rotate().as_dict()
        else:
            payload = store.provision(
                reuse_existing=arguments.reuse_existing,
            ).as_dict()
    except ApprovalTokenError as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "secret_exposed": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            {
                "status": "ok",
                **payload,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
