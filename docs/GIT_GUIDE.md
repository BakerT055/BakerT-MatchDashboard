# Putting BakerT MatchDashboard on GitHub

A step-by-step guide for Windows. Takes about 15 minutes. **Keep the repository PRIVATE** while you are building a product to sell.

## 0. One-time setup
1. **Install Git:** download from <https://git-scm.com/download/win> and accept the defaults. (This also installs *Git Credential Manager*, which handles GitHub login for you.)
2. **Create a GitHub account** at <https://github.com> if you don't have one.
3. Open **Command Prompt** (or *Git Bash*) and tell Git who you are – use the same email as your GitHub account:
   ```
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"
   ```

## 1. Prepare the folder
1. Extract the zip and rename/keep the folder as `BakerT-MatchDashboard`.
2. **Do not put your own data files in it before the first commit.** Your real results/fixtures may be licensed to you personally, and anything committed stays in Git history forever. The `.gitignore` already blocks everything in `data/` except the three sample files, plus `settings.json`, `.venv`, and any `.env`/secrets files.

## 2. Create the empty repository on GitHub
1. GitHub → **+** (top right) → **New repository**.
2. **Repository name:** `bakert-matchdashboard` (GitHub names can't contain spaces; the product name "BakerT MatchDashboard" lives in the README and app).
3. Set visibility to **Private**.
4. **Do NOT** tick "Add a README", ".gitignore" or "license" – the project already has them.
5. Click **Create repository**. Leave the page open; you need its URL.

## 3. First commit and push
In Command Prompt:
```
cd path\to\BakerT-MatchDashboard
git init -b main
git add .
git status
```
**Read the `git status` list before committing.** You should see only: `.github/`, `docs/`, `matchdash/`, `tests/`, `data/sample_history.csv`, `data/sample_fixtures.csv`, `data/team_aliases.json`, `.gitignore`, `README.md`, `LICENSE`, `CHANGELOG.md`, `ROADMAP.md`, `app.py`, `requirements.txt`, `run.bat`, `run.sh`. If you see `settings.json`, `.venv`, `*.pkl`, or any of your own data files, **stop** and tell me before continuing.

Then:
```
git commit -m "Initial commit: BakerT MatchDashboard v1.0.0"
git remote add origin https://github.com/YOUR-USERNAME/bakert-matchdashboard.git
git push -u origin main
```
The first push opens a browser window to sign in to GitHub. Refresh the repo page – your files are there. The **Actions** tab will run the tests automatically (a green tick means they pass).

Mark the first release:
```
git tag -a v1.0.0 -m "First working version"
git push origin v1.0.0
```

## 4. Everyday workflow
```
git status                       # what changed?
git add -A                       # stage everything (check status first!)
git commit -m "Add Elo feature"  # save a snapshot with a clear message
git push                         # upload to GitHub
```
Good habits:
- **Small commits, clear messages** ("Fix date parsing for UK dates", not "update").
- **Work on a branch for anything bigger than a quick fix** so `main` always works:
  ```
  git switch -c feature/elo-ratings
  ... work, commit ...
  git push -u origin feature/elo-ratings
  ```
  Then open a **Pull Request** on GitHub; the tests run on it; merge when green.
- **Run the tests before pushing:** `python tests/test_core.py`
- **Update `CHANGELOG.md`** and bump `__version__` in `matchdash/__init__.py` when you release (1.0.1 = bug fix, 1.1.0 = new feature, 2.0.0 = breaking change). Tag it: `git tag -a v1.1.0 -m "..."` then `git push origin v1.1.0`.

## 4b. Updating the repo with a new zip (e.g. v1.1.0)
1. Extract the new zip somewhere **else**.
2. Copy its contents **over** your existing `BakerT-MatchDashboard` folder (replace files; keep the hidden `.git` folder and your own `data/` files).
3. Review, commit, tag, push:
   ```
   git status
   git add -A
   git commit -m "Add ML engine and ML Lab (v1.1.0)"
   git push
   git tag -a v1.1.0 -m "ML engine"
   git push origin v1.1.0
   ```
`git status` should list only code/docs/tests – never `data/ml_bundle.pkl`, `settings.json` or your datasets (they are ignored).

## 5. Never commit
- **Secrets:** API keys, passwords, payment keys. Keep them in a `.env` file (already ignored) or in the host's environment settings. If one is ever pushed, treat it as leaked: revoke it and create a new one – deleting the file later does not remove it from history.
- **Your datasets and `settings.json`** (already ignored).
- Large files (hundreds of MB). Git is for code, not data.

## 6. Mistakes happen
| Problem | Fix |
|---|---|
| Staged a file by mistake (not yet committed) | `git restore --staged FILE` |
| Committed a file that should be ignored (not pushed yet) | `git rm --cached FILE`, add it to `.gitignore`, `git commit --amend` |
| Already pushed a secret or licensed data | Revoke the secret; for data, tell me – the history must be rewritten and the repo may need recreating |
| Want to undo the last commit but keep the work | `git reset --soft HEAD~1` |
| `git push` rejected | `git pull --rebase` then `git push` |

## 7. Backups and collaborators
- GitHub is your off-site backup once you push regularly.
- To add someone: repo → **Settings → Collaborators**. Give only the access needed.
- Turn on **two-factor authentication** on your GitHub account (Settings → Password and authentication).

## Prefer a point-and-click tool?
**GitHub Desktop** (<https://desktop.github.com>) does the same thing visually: *File → Add local repository* → *Publish repository* (tick **Keep this code private**). Still check the list of changed files before the first commit.
