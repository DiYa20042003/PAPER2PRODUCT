# Contributing

## Get started

1. Clone the repository and enter its root directory.
2. Follow the Python 3.10 and Node.js setup in [README.md](README.md).
3. Start the local application with `.\scripts\start.ps1` in PowerShell.
4. Stop it with `.\scripts\stop.ps1`.

## Before opening a pull request

Run the backend tests from the repository root:

```powershell
cd backend
..\.venv\Scripts\python.exe -m pytest tests -q
cd ..
```

Run the frontend checks:

```powershell
cd frontend
npm ci
npm run lint
npm run typecheck
npm run build
cd ..
```

The current Next.js version may stop a production build at its security
advisory gate. Retry once after acknowledging the notice; report the actual
result rather than treating a stopped build as a pass.

## Branches and pull requests

- Create branches using `feature/<name>` for features,
  `fix/<name>` for fixes, or `docs/<name>` for documentation.
- Open pull requests into `main` and describe scope, tests, and any known
  limitations. If the repository default branch has not yet been changed to
  `main`, use the current default base branch until maintainers change it.
- Keep each pull request focused and do not include unrelated generated or
  local runtime files.

## Data and secrets policy

Never commit `.env` or real keys, tokens, credentials, or passwords. Use the
tracked `.env.example` with placeholders for configuration.

Never edit the recorded metrics or alter them to make a run appear better.
Scientific results must come from the experiment and retain their actual
configuration and evidence.
