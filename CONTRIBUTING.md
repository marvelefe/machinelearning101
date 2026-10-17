# Contributing to Machine Learning 101

First off, thank you for considering a contribution! 🎉 This project exists to
make machine learning approachable for everyone, and contributions of all kinds —
code, content, docs, design, and bug reports — are welcome and appreciated.

By participating, you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

- 🌐 Live site: **https://machinelearning101.com**
- 🐛 [Report a bug](https://github.com/marvelefe/machinelearning101/issues/new?template=bug_report.md)
- 💡 [Request a feature](https://github.com/marvelefe/machinelearning101/issues/new?template=feature_request.md)
- 🙋 [Good first issues](https://github.com/marvelefe/machinelearning101/labels/good%20first%20issue)
- 🤝 [Help wanted](https://github.com/marvelefe/machinelearning101/labels/help%20wanted)

---

## Ways to Contribute

You don't need to be an ML expert or a seasoned developer. Some of the most
valuable contributions are:

| Type | Examples |
| ---- | -------- |
| 🐛 **Bug fixes** | Broken visualizations, wrong math, mobile layout issues |
| ✨ **New modules** | Add an algorithm, dataset, or interactive demo |
| 📚 **Documentation** | Fix typos, clarify explanations, improve the guides in `docs/` |
| 🎨 **Design & UX** | Accessibility, responsive layout, clearer controls |
| ⚡ **Performance** | Bundle size, rendering, Web Worker offloading |
| 🧪 **Tests** | Increase coverage in `tests/` |
| 🌍 **Localization** | Help translate the learning content |

Not sure where to start? Browse issues labeled
[`good first issue`](https://github.com/marvelefe/machinelearning101/labels/good%20first%20issue)
or open a [discussion/question](https://github.com/marvelefe/machinelearning101/issues/new)
and we'll help you find something.

---

## Getting Started

### Prerequisites

- **Node.js** 22 or higher (see [`.nvmrc`](.nvmrc)) — we recommend
  [nvm](https://github.com/nvm-sh/nvm): `nvm use`
- **npm** 10 or higher
- **Git**
- Basic familiarity with Vue 3, TypeScript, and machine learning concepts is
  helpful but not required for docs/design work.

### Set Up Your Environment

1. **Fork** the repository on GitHub (use the **Fork** button in the top-right).

2. **Clone your fork** and enter the directory:

   ```bash
   git clone https://github.com/YOUR_USERNAME/machinelearning101.git
   cd machinelearning101
   ```

3. **Add the upstream remote** so you can stay in sync:

   ```bash
   git remote add upstream https://github.com/marvelefe/machinelearning101.git
   ```

4. **Install dependencies**:

   ```bash
   npm install
   ```

5. **Start the dev server**:

   ```bash
   npm run dev
   ```

6. **Verify** by opening `http://localhost:3000`.

---

## Development Workflow

### 1. Create a Branch

Always branch off an up-to-date `main`:

```bash
git checkout main
git pull upstream main
git checkout -b feat/your-feature-name
# or
git checkout -b fix/short-bug-description
# or
git checkout -b docs/what-you-are-documenting
```

Branch prefix convention: `feat/`, `fix/`, `docs/`, `chore/`, `refactor/`.

### 2. Make Your Changes

Please follow the project's coding standards (see [`AGENTS.md`](AGENTS.md)):

- **TypeScript** everywhere; no implicit `any`, and type your public APIs.
- **Vue 3 Composition API** with `<script setup>`.
- Keep functions small and single-purpose; prefer explicit over implicit.
- Follow the existing file/folder structure and naming conventions.
- Delete dead code — don't comment it out.
- Add JSDoc comments for public APIs.
- Ensure UI is **keyboard accessible** and **responsive** (desktop, tablet,
  mobile).

### 3. Test Your Changes

Run these before pushing — CI runs the same commands:

```bash
npm run test         # Unit tests (Vitest)
npm run type-check   # TypeScript type checking
npm run build        # Must succeed with no errors
```

Add or update tests for any new logic. One test file per module
(`tests/<module>.test.ts`).

### 4. Commit Your Changes

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git add .
git commit -m "feat: add k-nearest neighbors module"
git commit -m "fix: correct gradient descent step size"
git commit -m "docs: clarify feature scaling explanation"
```

Use the imperative mood and keep commits atomic — one logical change per commit.

### 5. Push and Open a Pull Request

```bash
git push origin feat/your-feature-name
```

Then open a Pull Request against `main`. The
[PR template](.github/PULL_REQUEST_TEMPLATE.md) will guide you. Link the issue
your PR addresses (e.g. `Closes #123`).

---

## Pull Request Checklist

Before requesting review, make sure:

- [ ] The change is focused and does one thing
- [ ] `npm run test` passes
- [ ] `npm run type-check` passes
- [ ] `npm run build` succeeds
- [ ] New/updated logic has tests
- [ ] Public APIs are typed and documented
- [ ] UI changes are responsive and keyboard accessible
- [ ] Relevant documentation is updated
- [ ] Commits follow Conventional Commits
- [ ] Branch is up to date with `main`

---

## Adding a New Learning Module

Modules are the heart of this project. Each module lives under
`content/modules/<domain>/<module>/` and contains three files:

- `config.json` — metadata, parameters, datasets, visualizations
- `content.md` — the learning content
- `quiz.json` — quiz questions and answers

Plus an algorithm implementation under `algorithms/` that extends
`BaseAlgorithm` (`types/algorithm.ts`) and implements `fit()`, `predict()`,
`getVisualizationData()`, and `getMetrics()`.

For a full walkthrough, see the
[Module Creation Guide](docs/MODULE_CREATION_GUIDE.md).

---

## Reporting Bugs & Requesting Features

Use the issue templates — they help us help you:

- [🐛 Bug report](https://github.com/marvelefe/machinelearning101/issues/new?template=bug_report.md)
- [💡 Feature request](https://github.com/marvelefe/machinelearning101/issues/new?template=feature_request.md)

For **security vulnerabilities**, please **do not** open a public issue — follow
our [Security Policy](SECURITY.md) instead.

---

## Project Documentation

| Document | Description |
| -------- | ----------- |
| [Developer Guide](docs/DEVELOPER_GUIDE.md) | Development setup and workflow |
| [Module Creation Guide](docs/MODULE_CREATION_GUIDE.md) | Add a new learning module |
| [State Management](docs/STATE_MANAGEMENT.md) | How Pinia stores are organized |
| [User Guide](docs/USER_GUIDE.md) | End-user documentation |
| [Deployment Guide](docs/DEPLOYMENT_GUIDE.md) | Deployment instructions |
| [Worker README](worker/README.md) | Optional auth backend |

---

## Recognition

All contributors are valued. Meaningful contributions will be listed in release
notes. Thank you for making Machine Learning 101 better! 🚀
