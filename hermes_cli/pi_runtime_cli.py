"""Managed Pi runtime installation and isolated authentication commands."""
from __future__ import annotations

import argparse


def _install(args: argparse.Namespace) -> int:
    from agent.pi_runtime import install_managed_pi

    paths = install_managed_pi(npm_bin=args.npm, force=bool(args.force))
    print(f"Installed and verified Pi 0.84.3 at {paths.root}")
    print("Next: hermes pi-runtime auth")
    return 0


def _auth(args: argparse.Namespace) -> int:
    from agent.pi_runtime import launch_managed_pi_auth

    return launch_managed_pi_auth(
        acknowledge_trusted_local=bool(args.acknowledge_trusted_local)
    )


def register_cli(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "pi-runtime",
        help="Install or authenticate the pinned experimental Pi runtime",
    )
    actions = parser.add_subparsers(dest="pi_runtime_command")
    install = actions.add_parser("install", help="Install and verify Pi 0.84.3 with npm ci")
    install.add_argument("--npm", default="npm", help="npm executable (default: npm)")
    install.add_argument("--force", action="store_true", help="Atomically replace an invalid existing install")
    install.set_defaults(func=_install)
    auth = actions.add_parser(
        "auth", help="Run the separate elevated trusted-local Pi auth flow"
    )
    auth.add_argument(
        "--acknowledge-trusted-local",
        action="store_true",
        required=True,
        help="Acknowledge that authentication is not protected by the normal Pi container",
    )
    auth.set_defaults(func=_auth)

    def dispatch(args: argparse.Namespace) -> int:
        if getattr(args, "pi_runtime_command", None) and args.func is not dispatch:
            return args.func(args)
        parser.print_help()
        return 0

    parser.set_defaults(func=dispatch)
    return parser
