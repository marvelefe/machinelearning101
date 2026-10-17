# Directory & Awesome-List Submission Opportunities

Research date: **2026-10-17**

Project: **Machine Learning 101** — <https://machinelearning101.com> · repo: <https://github.com/marvelefe/machinelearning101>
Type: free, open-source (MIT), browser-based **interactive** ML learning platform for beginners (Nuxt 4 / Vue 3 / TypeScript).

This document lists curated GitHub directories ("awesome" lists, learning-resource indexes, ecosystem showcases)
where the project can legitimately be added via a pull request. Every repo below was verified live with the GitHub
API (star count, last push, merge activity) and its `README.md` / `CONTRIBUTING.md` were read to confirm the exact
insertion point and the required entry format.

> **Key strategic note.** Most *library/tool* awesome-lists gate on GitHub stars for the **project being submitted**
> (e.g. 50+, 1,000+). Machine Learning 101 is a **learning resource / website**, not a library, so the realistic wins
> are lists that index *educational resources and websites* — those have no star threshold. Those are ranked first.
>
> **Pre-req:** the repo reports no LICENSE to GitHub's licensee detector even though a valid `LICENSE` (MIT) file
> exists at the root. Before submitting to lists that link sources, double-check that GitHub shows the MIT badge
> (add the SPDX line only if needed). No candidate below requires the *submitted site* to carry an awesome badge.

---

## Summary table

| # | Repo | Stars | Active? | What to add / where | Confidence |
|---|------|-------|---------|---------------------|------------|
| 1 | [academic/awesome-datascience](#1-academicawesome-datascience) | ~30.2k | **Yes** — merges daily | `README.md` → `## Training Resources` → `### Tutorials` (append) | **High** |
| 2 | [nuxt/awesome](#2-nuxtawesome) | ~5.5k | **Yes** — last merge 2026-10-10 | `README.md` → `### Projects Using Nuxt` (append) | **High** |
| 3 | [hamid-moghadam/awesome-interactive-learning](#3-hamid-moghadamawesome-interactive-learning) | ~10 | **Yes** — merges 2026-10-12/16 | `README.md` → new `## Machine Learning & AI` (or `## Other Great Resources`) | **High** (low reach) |
| 4 | [vuejs/awesome-vue](#4-vuejsawesome-vue) | ~73.5k | **Yes** — merges 2026-10-01 | `README.md` → `## Projects Using Vue.js` → `### Apps/Websites` (append) | **Medium-High** |
| 5 | [gs00s/awesome-vue-3](#5-gs00sawesome-vue-3) | ~1.6k | Yes — last merge 2026-08-01 | `README.md` → `## Courses` (append) | **Medium-High** |
| 6 | [TirendazAcademy/Awesome-Data-Science-Resources](#6-tirendazacademyawesome-data-science-resources) | ~121 | Yes — pushed 2026-08-17 | `README.md` → `# 🚀 MACHINE LEARNING` → `### Courses` | **Medium-High** |
| 7 | [josephmisiti/awesome-machine-learning](#7-josephmisitiawesome-machine-learning) | ~74.6k | Yes — but PRs restricted | `courses.md` → `## Machine-Learning / Data Mining` | Medium |
| 8 | [ChristosChristofidis/awesome-deep-learning](#8-christoschristofidisawesome-deep-learning) | ~29k | Semi — last merge 2025-05 | `README.md` → `### Websites` (append) | Medium |
| 9 | [howie6879/mlhub123](#9-howie6879mlhub123) | ~1.1k | **Yes** — pushed 2026-10-16 | `README.md` → `## 资源` → `### 资源收集` | Medium (中文) |
| 10 | [guillaume-chevalier/Awesome-Deep-Learning-Resources](#10-guillaume-chevalierawesome-deep-learning-resources) | ~1.8k | Low — last push 2024-01 | `README.md` → `## Online Classes` (append) | Medium-Low |
| 11 | [ujjwalkarn/Machine-Learning-Tutorials](#11-ujjwalkarnmachine-learning-tutorials) | ~18.3k | Low — last push 2024-06 | `README.md` → `## Introduction` (append) | Medium-Low |
| 12 | [owainlewis/awesome-artificial-intelligence](#12-owainlewisawesome-artificial-intelligence) | ~16.7k | Yes — last merge 2026-08 | `README.md` → `## Learn` → `### Courses` | Medium-Low (selective) |
| 13 | [allmachinelearning/MachineLearning](#13-allmachinelearningmachinelearning) | ~3.9k | Low — last push 2024-04 | `README.md` → `## 课程与讲座` → `### 机器学习` | Low (中文, stale) |

**Clearly NO (do not submit):** see [Not recommended / rejected](#not-recommended--rejected).

---

## 1. academic/awesome-datascience

- **URL:** <https://github.com/academic/awesome-datascience> · **Stars:** ~30,198 · **Default branch:** `live`
- **Activity:** **Very active** — merged PRs #724/#725/#727 all on 2026-10-16. Actively maintained.
- **What it indexes:** A broad "awesome Data Science" list of tools, courses, tutorials, books, datasets, communities.
  Explicitly beginner-oriented ("How do you learn data science? By doing data science").
- **Fit:** **Excellent.** The `### Tutorials` subsection already contains interactive/visual ML learning resources
  (e.g. *StepByStepML — interactive calculator that visualizes the step-by-step math behind ML algorithms*).
  A free browser-based interactive platform fits the section's intent squarely.
- **Exact place:** `README.md` → `## Training Resources` → `### Tutorials`. Append at the **end** of the section
  (after `Train LLM From Scratch`). The section is not alphabetized; append-only is consistent with recent edits.
- **Contribution requirements (`CONTRIBUTING.md`):** consistent bullet format `- [Name](URL) - Short description.`;
  descriptions one sentence, title case, **no marketing language/superlatives**; prefer `https://`, no URL shorteners;
  add near similar items; complete the PR template (tip model, license type, **your relationship to the project**);
  link checks must pass. Code of Conduct applies.
- **Proposed entry:**
  ```markdown
  - [Machine Learning 101](https://machinelearning101.com) - Free browser-based platform that teaches core machine learning algorithms through hands-on, interactive visualizations with no setup.
  ```
- **Confidence:** **High.**

---

## 2. nuxt/awesome

- **URL:** <https://github.com/nuxt/awesome> (canonical; `nuxt-community/awesome-nuxt` redirects here) · **Stars:** ~5,537
- **Activity:** Active — `chore(showcase): remove faker.js` merged 2026-10-10; showcase entries are hand-curated.
- **What it indexes:** Everything Nuxt — official resources, modules, examples, articles, books, open-source projects,
  and a large curated **"Projects Using Nuxt"** website showcase.
- **Fit:** **Excellent.** ML101 is built with Nuxt 4. The showcase section lists production/education websites
  (Khan-style platforms, universities, products) with a one-line description.
- **Exact place:** `README.md` → `### Projects Using Nuxt`. Append to the **bottom** of the list (the section is a plain
  bullet list; new sites are added at the end).
- **Contribution requirements (`contributing.md`):** submission **must be Nuxt related**; search for duplicates first;
  **one suggestion per PR**; format exactly `[Title](link) - Description.`; additions go to the **bottom of the
  relevant category**; descriptions start with a capital and end with a period, no leading "A"/"An"; no trailing
  whitespace; PR title should state the package/site and why it belongs. New categories in a separate PR.
- **Proposed entry:**
  ```markdown
  - [Machine Learning 101](https://machinelearning101.com) - Free, interactive, browser-based machine learning learning platform for beginners.
  ```
- **Confidence:** **High.**

---

## 3. hamid-moghadam/awesome-interactive-learning

- **URL:** <https://github.com/hamid-moghadam/awesome-interactive-learning> · **Stars:** ~10
- **Activity:** **Very active for its age** — new repo, PRs #1 (2026-10-12) and #2 (2026-10-16) already merged.
  CC0-licensed, clearly welcoming contributions.
- **What it indexes:** *Only* interactive learning platforms, visualizations, labs and hands-on courses
  (Learn Git Branching, Seeing Theory, Frontend Mentor, AlgoViz, …). This is the most on-topic list found.
- **Fit:** **Perfect conceptually** — but reach is tiny (single-digit stars). Low risk, low reward.
- **Exact place:** `README.md`. Either append to `## Other Great Resources`, **or** (better) open a small companion PR
  adding a new `## Machine Learning & AI` section and its entry in `## Contents`.
- **Contribution requirements (`CONTRIBUTING.md`):** resource **must be interactive** (not static article/video);
  PR body uses format `- [Name](https://example.com/) – Short description (1 sentence)` (note the en dash `–`).
- **Proposed entry:**
  ```markdown
  - [Machine Learning 101](https://machinelearning101.com) – Free browser-based platform for learning ML algorithms by adjusting parameters and watching training run live.
  ```
- **Confidence:** **High** (that it is accepted); reach is **low**.

---

## 4. vuejs/awesome-vue

- **URL:** <https://github.com/vuejs/awesome-vue> · **Stars:** ~73,516
- **Activity:** **Very active** — multiple merges on 2026-10-01 (#4559, #4556, #4554).
- **What it indexes:** The flagship Vue.js list: projects, components, ecosystem, and a "Projects Using Vue.js" showcase
  split into `### Open Source`, `### Commercial Products`, `### Apps/Websites`, `### Interactive Experiences`.
- **Fit:** **Good.** Submit to the **website** showcase (not "Open Source", which requires the linked repo to be the
  entry and is stricter). Nuxt/Vue is used intensively and is detectable with vue-devtools, which satisfies the rule.
- **Exact place:** `README.md` → `## Projects Using Vue.js` → `### Apps/Websites`. **Append to the end** (order is
  first-come-first-served).
- **Contribution requirements (`.github/contributing.md` + `pull_request_template.md`):** put items in the right
  category; **always add to the end of the list**; site must load without errors/SSL issues and be reasonably fast;
  must use Vue intensively (detectable via vue-devtools, else attach a screenshot in the PR); must be **original and
  not too simple** (blogs/simple landing pages rejected); not a commercial product. Fill out the PR template checkboxes.
- **Proposed entry:**
  ```markdown
  - [Machine Learning 101](https://machinelearning101.com) - Interactive, browser-based machine learning learning platform for beginners built with Vue and Nuxt.
  ```
- **Confidence:** **Medium-High.**

---

## 5. gs00s/awesome-vue-3

- **URL:** <https://github.com/gs00s/awesome-vue-3> · **Stars:** ~1,646
- **Activity:** Yes — merges on 2026-08-01 (#150, #149, #147).
- **What it indexes:** Curated list of *Vue 3* resources, articles, packages, examples, tools, videos, books, courses.
- **Fit:** Good — a real Vue 3 + Nuxt project can be listed as a course/resource. The `## Courses` section holds
  learning platforms (Vue School, Vue Mastery, Frontend Masters).
- **Exact place:** `README.md` → `## Courses`. Append at the bottom. (Alternative: `## Examples` only if you point to
  the source repo; Courses is the better semantic fit for a learning platform.)
- **Contribution requirements (`CONTRIBUTING.md`):** state the **intent/problem** in the PR; quality bar (no spelling
  mistakes, reads well); must move the repo toward its vision; respect the Contributor Covenant. No strict formatting
  mandate — mirror the existing bullet style.
- **Proposed entry:**
  ```markdown
  - [Machine Learning 101](https://machinelearning101.com) - Interactive Vue 3 and Nuxt learning platform that teaches machine learning through in-browser visualizations.
  ```
- **Confidence:** **Medium-High.**

---

## 6. TirendazAcademy/Awesome-Data-Science-Resources

- **URL:** <https://github.com/TirendazAcademy/Awesome-Data-Science-Resources> · **Stars:** ~121
- **Activity:** Pushed 2026-08-17. Low-volume but current; no CONTRIBUTING.md (follow existing style).
- **What it indexes:** A tidy, sectioned data-science resource list: Data Science, **Machine Learning**, ML Engineering,
  Deep Learning, Generative AI, NLP, Data Engineering, SQL — each with `Free Books`, `Courses`, `Tools`, `Repos`, `Projects`.
- **Fit:** **Good.** The `# 🚀 MACHINE LEARNING` → `### Courses` block already lists third-party learning resources
  (Coursera, TutorialSearch, …). An interactive free platform fits.
- **Exact place:** `README.md` → `# 🚀 MACHINE LEARNING` → `### Courses`. Append at the bottom of that list.
- **Contribution requirements:** No `CONTRIBUTING.md`; match the surrounding bullets (`- [Name](url) - description`).
  Keep it factual and free.
- **Proposed entry:**
  ```markdown
  - [Machine Learning 101](https://machinelearning101.com) - Free interactive browser-based platform to learn machine learning algorithms by experimenting with them.
  ```
- **Confidence:** **Medium-High.**

---

## 7. josephmisiti/awesome-machine-learning

- **URL:** <https://github.com/josephmisiti/awesome-machine-learning> · **Stars:** ~74,597 · **Default branch:** `master`
- **Activity:** Repo pushed 2026-10-13, but the README carries an explicit notice: *"As of April 2026, too many PRs are
  being generated by LLMs… If you want to contribute… email me … with a link to your PR and I'll merge it."*
- **What it indexes:** The main list is **frameworks, libraries and software by language**. Learning material lives in
  separate files: `books.md`, `courses.md`, `blogs.md`, `events.md`, `meetups.md`.
- **Fit:** The interactive platform belongs **only** in `courses.md` ("free or paid online courses on machine learning…").
  `courses.md` already lists visual/interactive items (MLU-Explain, fast.ai) and even platform-style entries.
- **Exact place:** `courses.md` → `## Machine-Learning / Data Mining`, near the top (recent "System Designer"
  platform entries are at the top).
- **Contribution requirements:** No CONTRIBUTING.md, but the README notice **requires emailing the maintainer** to prove
  you are human, with a link to the PR. Entries use `* [Name](url) - description - free`. Expect friction.
- **Proposed entry:**
  ```markdown
  * [Machine Learning 101](https://machinelearning101.com) - Interactive, browser-based platform for learning core ML algorithms through hands-on visualizations - free
  ```
- **Confidence:** **Medium** (great reach; extra manual email gate + it is primarily a software list).

---

## 8. ChristosChristofidis/awesome-deep-learning

- **URL:** <https://github.com/ChristosChristofidis/awesome-deep-learning> · **Stars:** ~29,057
- **Activity:** Semi-active — last merge 2025-05-26 (#252). Occasional merges; the list is dated but still widely used.
- **What it indexes:** Deep-learning books, courses, videos, papers, tutorials, **websites**, datasets, tools,
  communities.
- **Fit:** Good — the `### Websites` section is literally external sites (deeplearning.net, Stanford pages…); ML101 is a
  website. `### Tutorials` is a reasonable alternative.
- **Exact place:** `README.md` → `### Websites`. Entries are a **numbered list** (`1. [name](url)`); append as the next
  number.
- **Contribution requirements:** `### Contributing` says simply to send a pull request (the link points to a fork's PR
  page). No linting, no format spec beyond the numbered list. CC0.
- **Proposed entry:**
  ```markdown
  11.  [Machine Learning 101](https://machinelearning101.com)
  ```
- **Confidence:** **Medium** (slow maintainer; dated README aesthetic).

---

## 9. howie6879/mlhub123

- **URL:** <https://github.com/howie6879/mlhub123> · **Stars:** ~1,147
- **Activity:** **Very active** — pushed 2026-10-16.
- **What it indexes:** A Chinese-language roundup of ML/DL **websites & resources** (news, tools, communities, blogs,
  courses, open-source books, projects).
- **Fit:** Good if you can supply a Chinese description; it indexes learning websites directly.
- **Exact place:** `README.md` → `## 资源` → `### 资源收集` (resource collection) or `### 课程学习` (courses).
- **Contribution requirements:** No `CONTRIBUTING.md`; entries are Chinese bullet lines; the site itself is bilingual,
  so a Chinese one-liner is expected. `https://machinelearning101.com` is fine as the URL.
- **Proposed entry:**
  ```markdown
  - [Machine Learning 101](https://machinelearning101.com) - 免费的浏览器端交互式机器学习学习平台，通过可视化实验帮助初学者理解算法。
  ```
- **Confidence:** **Medium** (language requirement; active maintainer).

---

## 10. guillaume-chevalier/Awesome-Deep-Learning-Resources

- **URL:** <https://github.com/guillaume-chevalier/Awesome-Deep-Learning-Resources> · **Stars:** ~1,821
- **Activity:** Low — last push 2024-01-18. Personal but fairly popular list.
- **What it indexes:** Trends, **Online Classes**, books, posts, practical resources, math theory, papers, videos, and a
  `Misc. Hubs & Links` section.
- **Fit:** Medium — the list is opinionated/personal; `## Online Classes` (which includes free courses and YouTube
  series) is the best home. `## Misc. Hubs & Links` is a fallback.
- **Exact place:** `README.md` → `## Online Classes`, append; or `## Misc. Hubs & Links`.
- **Contribution requirements:** No CONTRIBUTING.md; match `- [Name](url) - description.` bullets.
- **Proposed entry:**
  ```markdown
  - [Machine Learning 101](https://machinelearning101.com) - Free, browser-based interactive platform for building intuition about ML algorithms with zero setup.
  ```
- **Confidence:** **Medium-Low** (stale maintenance).

---

## 11. ujjwalkarn/Machine-Learning-Tutorials

- **URL:** <https://github.com/ujjwalkarn/Machine-Learning-Tutorials> · **Stars:** ~18,280
- **Activity:** Low — last push 2024-06-12. Large and well-known, but quiet.
- **What it indexes:** Beginner ML/DL tutorials, courses and articles, grouped by topic (`## Introduction`,
  `## Other Tutorials`, per-algorithm sections).
- **Fit:** Medium — `## Introduction` already lists beginner breakthroughs and "curated list" links; a free interactive
  beginner platform fits there or in `## Other Tutorials`.
- **Exact place:** `README.md` → `## Introduction`, append; or `## Other Tutorials` (end of file).
- **Contribution requirements:** No CONTRIBUTING.md. Entries are plain `- [Name](url)` bullets. Large list, informal
  review; CC0.
- **Proposed entry:**
  ```markdown
  - [Machine Learning 101](https://machinelearning101.com)
  ```
- **Confidence:** **Medium-Low** (maintainer inactivity; may not be reviewed).

---

## 12. owainlewis/awesome-artificial-intelligence

- **URL:** <https://github.com/owainlewis/awesome-artificial-intelligence> · **Stars:** ~16,680
- **Activity:** Yes — merges 2026-08-15. Now has automated validation (`scripts/validate_readme.py`) and a strict
  curation policy (`CURATION.md`).
- **What it indexes:** An intentionally **selective** guide organised around three pillars: AI foundations, building AI
  systems, agentic software engineering. Books, courses, papers.
- **Fit:** Medium-low. An interactive beginner ML platform *can* serve "AI foundations → high-quality courses and
  tutorials", but the policy explicitly warns against promotion-driven personal projects and general product
  directories. Submission must argue differentiation from the strongest existing entry.
- **Exact place:** `README.md` → `## Learn` → `### Courses`, appended.
- **Contribution requirements (`CONTRIBUTING.md` + `CURATION.md`):** read `CURATION.md`; must serve a pillar; provide
  evidence (primary source, maintenance status, **your affiliation**); format exactly
  `- [Resource name](https://example.com/): One factual sentence explaining its value to developers.`; run the three
  validation commands locally; complete the PR template. Maintainers may decline entries that pass checks but fail the
  scope/quality policy.
- **Proposed entry:**
  ```markdown
  - [Machine Learning 101](https://machinelearning101.com): A free, browser-based course that teaches core machine learning algorithms through interactive, parameter-driven visualizations.
  ```
- **Confidence:** **Medium-Low** (deliberately selective list).

---

## 13. allmachinelearning/MachineLearning

- **URL:** <https://github.com/allmachinelearning/MachineLearning> · **Stars:** ~3,865
- **Activity:** Low — last push 2024-04-01. Chinese-language ML resource index.
- **What it indexes:** Prerequisites, notes, **courses & talks** (`## 课程与讲座`) by ML/DL/RL, books, miscellaneous.
- **Fit:** Medium — an interactive beginner platform fits the courses section, but the repo is stale and mostly Chinese.
- **Exact place:** `README.md` → `## 课程与讲座` → `### 机器学习`.
- **Contribution requirements:** Has a `## 如何加入 How to contribute` section; no strict format; Chinese context
  preferred.
- **Proposed entry:**
  ```markdown
  - [Machine Learning 101](https://machinelearning101.com) - 免费的交互式机器学习学习平台，在浏览器中动手学习算法。
  ```
- **Confidence:** **Low** (stale + language barrier).

---

## Not recommended / rejected

These were checked and are **not viable** — do not spend a PR here.

| Repo | Reason |
|------|--------|
| [sindresorhus/awesome](https://github.com/sindresorhus/awesome) | README states **PRs are temporarily disabled**. It also indexes lists, not resources. |
| [sdmg15/Best-websites-a-programmer-should-visit](https://github.com/sdmg15/Best-websites-a-programmer-should-visit) | Repository is **archived**. |
| [Machine-Learning-Tokyo/Interactive_Tools](https://github.com/Machine-Learning-Tokyo/Interactive_Tools) | Topically ideal (interactive ML tools), but **last merged PR was 2020** — effectively unmaintained. |
| [EbookFoundation/free-programming-books](https://github.com/EbookFoundation/free-programming-books) | Organised **by programming language** for books/courses; a standalone platform has no correct section. |
| [practical-tutorials/project-based-learning](https://github.com/practical-tutorials/project-based-learning) | Requires a **project-based tutorial where the reader builds an artifact**. ML101 is a platform, not a tutorial series; a validator + link checker enforces this. |
| [hal9ai/awesome-dataviz](https://github.com/hal9ai/awesome-dataviz) | Only accepts **data-visualization libraries** (open-source repos, 50+ stars). ML101 is not a dataviz library. |
| [lukasmasuch/best-of-ml-python](https://github.com/lukasmasuch/best-of-ml-python) / [krzjoa/awesome-python-data-science](https://github.com/krzjoa/awesome-python-data-science) | Python **software/libraries** only. |
| [steven2358/awesome-generative-ai](https://github.com/steven2358/awesome-generative-ai) | Main list requires **1,000+ followers**; ML101 would be pushed to `DISCOVERIES.md`. |
| [natnew/Awesome-Data-Science](https://github.com/natnew/Awesome-Data-Science) | Requires the submitted repo to be **30+ days old and have 60+ stars** — not met. |
| [prakhar1989/awesome-courses](https://github.com/prakhar1989/awesome-courses) | **University courses** only; last push 2023. |
| [kmario23/deep-learning-drizzle](https://github.com/kmario23/deep-learning-drizzle) | Only accepts courses **with lecture videos**. |
| [dair-ai/ML-YouTube-Courses](https://github.com/dair-ai/ML-YouTube-Courses) | **YouTube-only**; last push 2024-01. |
| [yrgo/awesome-educational-games](https://github.com/yrgo/awesome-educational-games) | **Educational games** only. |
| [onmyway133/awesome-machine-learning](https://github.com/onmyway133/awesome-machine-learning) | Explicitly **CoreML/iOS** focused. |
| [TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials](https://github.com/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials) | Hosts its **own** lessons; not a directory of external resources. |
| [mercer/awesome-educate](https://github.com/mercer/awesome-educate) / [brandonhimpfen/awesome-ml](https://github.com/brandonhimpfen) style micro-lists | **Dead/low activity** (<60★, last push 2022) — not worth it. |

---

## How to submit

Submit **one PR per repo** (nearly every list requires this), appending to the bottom of the specified section.

1. **Fork & branch.** Branch like `docs/add-machine-learning-101`.
2. **Make the minimal edit** exactly in the file/section named above, matching the surrounding bullet style.
   Do not reorder or reword unrelated entries.
3. **Self-disclose authorship.** State in the PR body that you are the author/maintainer of Machine Learning 101
   (required by `academic/awesome-datascience` and `owainlewis/awesome-artificial-intelligence`; encouraged elsewhere).
4. **Use the project's canonical URL** `https://machinelearning101.com` (not the GitHub repo) unless the list
   explicitly asks for the source repo.
5. **Follow each list's mechanical rules:**
   - `academic/awesome-datascience`, `nuxt/awesome`, `vuejs/awesome-vue`: run the repo's link check / use their PR template.
   - `practical-tutorials/*` and `owainlewis/*`: run their local validators (`python3 scripts/...`) before opening the PR.
   - Watch for alphabetical-order requirements (most sections here are append-only; the dataviz list is not relevant).
6. **Order of attack (best ROI first):**
   1. `academic/awesome-datascience` (huge, active, perfect fit)
   2. `nuxt/awesome` (native Nuxt showcase)
   3. `vuejs/awesome-vue` (native Vue showcase)
   4. `TirendazAcademy/Awesome-Data-Science-Resources` and `gs00s/awesome-vue-3` (easy wins)
   5. `hamid-moghadam/awesome-interactive-learning` (trivial, on-topic)
   6. `josephmisiti/awesome-machine-learning` (`courses.md` + email the maintainer) and `ChristosChristofidis/awesome-deep-learning`
   7. Long shots: `owainlewis/...`, `howie6879/mlhub123`, `guillaume-chevalier/...`, `ujjwalkarn/...`, `allmachinelearning/...`
7. **Track progress** with the companion GitHub issue in this repo
   (`Submit project to curated awesome-lists and learning directories`).

### Suggested PR body (reusable)

> **Add [Machine Learning 101](https://machinelearning101.com) to `<section>`**
>
> Machine Learning 101 is a free, open-source (MIT), browser-based interactive learning platform for machine learning
> beginners. It runs entirely in the browser with zero setup and lets learners adjust parameters and watch algorithms
> (regression, classification, clustering, gradient descent, PCA, t-SNE, neural nets) train and update in real time.
> Built with Nuxt 4 / Vue 3 / TypeScript. I am the project maintainer.
