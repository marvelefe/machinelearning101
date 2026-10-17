<div align="center">

# Machine Learning 101

**Learn machine learning by doing — right in your browser.**

An interactive, open-source learning platform for beginners. Adjust parameters, watch algorithms run,
and build real intuition — with **zero setup required**.

[![Website](https://img.shields.io/badge/Website-machinelearning101.com-4f46e5?logo=googlechrome&logoColor=white)](https://machinelearning101.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/marvelefe/machinelearning101/actions/workflows/ci.yml/badge.svg)](https://github.com/marvelefe/machinelearning101/actions/workflows/ci.yml)
[![Nuxt](https://img.shields.io/badge/Nuxt-4.x-00DC82?logo=nuxt.js&logoColor=white)](https://nuxt.com)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

### 👉 **[Try it live at machinelearning101.com](https://machinelearning101.com)**

</div>

---

## 🌟 What is this?

Machine Learning 101 is a free, browser-based platform for building intuition about machine learning.
Instead of just reading theory, you **experiment**: tweak a learning rate and watch gradient descent
converge, drag points and see a decision boundary move, add clusters and watch K-means settle.

Everything runs client-side — no installs, no sign-up required, no backend needed. Your progress is
saved locally in your browser (with optional cloud sync if you sign in).

> **Zero setup · Visual-first · Learn by doing · Open source**

## ✨ Features

- 🎯 **Interactive learning** — Real-time parameter adjustments with immediate visual feedback
- 📊 **Rich visualizations** — Interactive 2D/3D charts powered by Chart.js, D3.js, and Plotly.js
- 🧪 **Hands-on experiments** — Train models, tune parameters, and inspect results instantly
- 📚 **Guided content** — Clear explanations, interactive quizzes, and progress tracking
- 🔒 **Privacy-focused** — Optional, opt-in analytics with full user control and GDPR compliance
- 📱 **Responsive** — Works on desktop, tablet, and mobile
- ⚡ **Zero setup** — No installations, no backend dependencies, no barriers to learning
- 🌐 **Fully static** — Deployable to any static host (Netlify, Vercel, GitHub Pages)
- 🔐 **Optional cloud sync** — Sign in to keep progress across devices (Cloudflare Workers + D1)

### Learning domains

| Domain | Topics |
| ------ | ------ |
| **Regression** | Linear regression, logistic regression |
| **Classification** | KNN, decision trees, SVM |
| **Clustering** | K-means, hierarchical clustering |
| **Preprocessing** | Feature scaling, encoding, imputation, feature selection |
| **Optimization** | Gradient descent, momentum, RMSProp, Adam |
| **Advanced** | PCA, t-SNE, neural networks, CNNs, hyperparameter tuning |

## 🚀 Getting Started

### Prerequisites

- **Node.js** 22+ (see [`.nvmrc`](.nvmrc) — we recommend [nvm](https://github.com/nvm-sh/nvm): `nvm use`)
- **npm** 10+

### Run Locally

```bash
# Clone the repository
git clone https://github.com/marvelefe/machinelearning101.git
cd machinelearning101

# Install dependencies
npm install

# Start the dev server
npm run dev
```

Open **http://localhost:3000** 🎉

### Useful Commands

| Command | Description |
| ------- | ----------- |
| `npm run dev` | Start the development server |
| `npm run build` | Build for production |
| `npm run preview` | Preview the production build locally |
| `npm run generate` | Generate the fully static site |
| `npm run test` | Run unit tests (Vitest) |
| `npm run type-check` | Run TypeScript type checking |

## 🏗️ Project Structure

```
.
├── algorithms/          # ML algorithm implementations (extend BaseAlgorithm)
│   ├── advanced/       #   Neural networks, PCA, t-SNE
│   ├── classification/ #   KNN, decision trees, SVM
│   ├── clustering/     #   K-means, hierarchical
│   ├── optimization/   #   Gradient descent variants
│   ├── preprocessing/  #   Scaling, encoding, imputation
│   └── regression/     #   Linear, logistic regression
├── components/          # Vue components
│   ├── layout/         #   Header, footer, navigation
│   ├── modules/        #   Module-specific UI
│   ├── shared/         #   Reusable components
│   └── visualizations/ #   Chart wrappers
├── composables/         # Vue composables (reusable logic)
├── content/            # Learning content (markdown + JSON)
│   └── modules/        #   Organized by domain
├── pages/              # Nuxt pages (auto-routing)
├── stores/             # Pinia state management
├── types/              # TypeScript definitions
├── utils/              # Utility functions
├── tests/              # Unit tests (Vitest)
├── worker/             # Optional Cloudflare Worker (auth backend)
└── service-workers/    # Web Workers (heavy computation)
```

## 🛠️ Tech Stack

- **Framework**: [Nuxt 4](https://nuxt.com) (Vue 3 + TypeScript, strict mode)
- **State**: [Pinia](https://pinia.vuejs.org) with persistence
- **Styling**: [Tailwind CSS](https://tailwindcss.com)
- **Visualizations**: [Chart.js](https://www.chartjs.org), [D3.js](https://d3js.org), [Plotly.js](https://plotly.com/javascript/)
- **Backend** (optional): Cloudflare Workers + D1
- **Testing**: [Vitest](https://vitest.dev)
- **Build**: Vite, with static site generation

## 📚 Documentation

| Guide | Description |
| ----- | ----------- |
| [User Guide](docs/USER_GUIDE.md) | How to use the platform |
| [Developer Guide](docs/DEVELOPER_GUIDE.md) | Development setup and workflow |
| [Module Creation Guide](docs/MODULE_CREATION_GUIDE.md) | Add a new learning module |
| [State Management](docs/STATE_MANAGEMENT.md) | How Pinia stores are organized |
| [Quiz Integration](docs/QUIZ_INTEGRATION_GUIDE.md) | Adding quizzes to modules |
| [Bookmark Integration](docs/BOOKMARK_INTEGRATION_GUIDE.md) | Bookmarking support |
| [Core Web Vitals](docs/CORE_WEB_VITALS.md) | Performance notes |
| [Deployment Guide](docs/DEPLOYMENT_GUIDE.md) | Deploying the site & backend |
| [Worker README](worker/README.md) | Optional auth backend |

## 🔐 Optional Authentication Backend

Signing in enables cross-device progress sync via Cloudflare Workers + D1 and Google OAuth.
This is entirely optional — the site works fully without it. See the
[Worker README](worker/README.md) and the [Deployment Guide](docs/DEPLOYMENT_GUIDE.md) to set it up.

## 🤝 Contributing

Contributions are warmly welcomed — code, content, docs, design, and bug reports alike! 💚

- 📖 Read the **[Contributing Guide](CONTRIBUTING.md)** to get started
- 🙋 Browse **[good first issues](https://github.com/marvelefe/machinelearning101/labels/good%20first%20issue)** — perfect for newcomers
- 🤝 Check out **[help wanted](https://github.com/marvelefe/machinelearning101/labels/help%20wanted)** issues for meatier tasks
- 💬 Ask questions in **[Discussions](https://github.com/marvelefe/machinelearning101/discussions)**
- 🐛 [Report a bug](https://github.com/marvelefe/machinelearning101/issues/new?template=bug_report.md) · 💡 [Request a feature](https://github.com/marvelefe/machinelearning101/issues/new?template=feature_request.md)

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md).
By participating you agree to abide by its terms.

## 🔒 Security

Found a vulnerability? Please report it privately — see our [Security Policy](SECURITY.md).
Do **not** open a public issue for security concerns.

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

## ⭐ Support the Project

If Machine Learning 101 helped you learn something, consider giving it a **star** ⭐ — it helps others
discover the project. And if you'd like to contribute, we'd love to have you!

<div align="center">

Made with ❤️ for learners everywhere.

**[machinelearning101.com](https://machinelearning101.com)**

</div>
