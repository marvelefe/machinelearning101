# Security Policy

## Supported Versions

This project is deployed continuously from the `main` branch. Only the latest
released version of the site and the latest commit on `main` are supported with
security updates.

| Version         | Supported          |
| --------------- | ------------------ |
| `main` (latest) | :white_check_mark: |
| Older commits   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
discussions, or pull requests.**

Instead, report them privately using one of the following methods:

1. **GitHub Private Vulnerability Reporting** — go to the
   [Security tab](https://github.com/marvelefe/machinelearning101/security) and
   click **"Report a vulnerability"**. This is the preferred method.
2. **Email** — send a description to **marvelcamoarmy@gmail.com**.

Please include the following where possible:

- A description of the vulnerability and its potential impact
- Step-by-step reproduction instructions (proof of concept)
- The affected URL, component, or file
- Any suggested fix or mitigation

## What to Expect

- **Acknowledgment** within 72 hours of your report.
- **Assessment** and an initial response (accepted / declined / needs more info)
  within 7 days.
- **Credit** in the release notes or advisory once a fix is published, unless you
  prefer to remain anonymous.

We ask that you give us a reasonable amount of time to address the issue before
any public disclosure (coordinated disclosure). We will keep you informed of our
progress throughout.

## Scope

This policy covers:

- The frontend web application (https://machinelearning101.com)
- The open-source code in this repository
- The optional authentication backend under `worker/` (Cloudflare Workers + D1)

The following are generally **out of scope**:

- Vulnerabilities in third-party dependencies (please report these upstream, but
  do let us know so we can update)
- Denial-of-service from extreme traffic volumes
- Findings that require an already-compromised device or account
- Missing security headers without a demonstrated impact

## Safe Harbor

We will not pursue legal action against researchers who make a good-faith effort
to comply with this policy, avoid privacy violations and service disruption, and
report findings promptly and privately.

Thank you for helping keep Machine Learning 101 and its users safe.
