# Security Policy

## Supported Versions

This project is currently maintained on the `main` branch.

## Reporting a Vulnerability

Please do not open a public issue for sensitive vulnerabilities.

Report security issues privately through one of these channels:

- GitHub Security Advisories for this repository
- a private maintainer contact channel you already use with the project owner

When reporting, include:

- a short description of the issue
- affected files, endpoints, or commands
- reproduction steps or a proof of concept
- impact and any suggested mitigation

## Scope

Please report issues such as:

- prompt injection that can bypass intended retrieval or refusal boundaries
- unintended data exposure from private corpora or local file paths
- insecure defaults around secrets, model endpoints, or file uploads
- unsafe document ingestion behavior that could execute untrusted input

## Secrets and Private Data

- Keep real secrets in local `.env` files only.
- Never commit API keys, personal literature PDFs, group internal documents, or raw audio recordings.
- Rotate any secret immediately if it was exposed in chat logs, screenshots, or commits.
