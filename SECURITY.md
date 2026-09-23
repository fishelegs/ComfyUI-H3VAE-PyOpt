# Security Policy

## Supported versions

Security fixes are targeted at the current release line and the latest
`main` branch.

| Version | Security support |
| --- | --- |
| 0.2.x | Supported |
| main | Supported |
| 0.1.x and older | Not guaranteed |

Users should reproduce a security issue on the latest release or `main` when
possible before reporting it.

## Reporting a vulnerability

Please do **not** open a public GitHub issue for a vulnerability that could
expose credentials, execute unintended code, access unintended files, or
otherwise put users at risk.

If the repository's **Security** tab offers **Report a vulnerability**, use
GitHub private vulnerability reporting. If that option is unavailable, contact
the maintainer through the GitHub profile and request a private reporting
channel before sending exploit details.

Include, when relevant:

- affected version or commit
- operating system and Python/PyTorch/CUDA versions
- minimal reproduction steps
- expected versus observed behavior
- impact assessment
- whether the issue requires a malicious model, model-code directory, workflow,
  input file, or local configuration
- a proposed fix, if available

Do not include Registry tokens, API keys, model-service credentials, private
model assets, or unrelated local file contents in a report.

## Security boundaries

This project loads separately supplied MiniMax H3 / FL2VA Python model code and
model weights. Users are responsible for obtaining those components from
sources they trust. Loading untrusted Python model code is outside the security
boundary of this repository and can execute arbitrary Python with the user's
permissions.

The project also uses PyTorch, Triton, safetensors, and optional comfy-kitchen.
Vulnerabilities that originate entirely in those upstream projects should be
reported to the corresponding upstream security process; reports showing that
this repository uses an upstream component unsafely are in scope here.

Performance differences, numerical quality regressions, unsupported hardware,
and expected failures of experimental INT8 modes are normally bugs rather than
security vulnerabilities unless they create a concrete security impact.

## Secret handling

- Never commit `REGISTRY_ACCESS_TOKEN` or other credentials.
- Do not place secrets in workflows, benchmark JSON, issue bodies, logs, or
  example configuration.
- Treat any credential posted publicly as compromised and rotate it.
