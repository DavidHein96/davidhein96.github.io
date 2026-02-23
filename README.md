# davidhein96.github.io

[![CI](https://github.com/davidhein96/davidhein96.github.io/actions/workflows/ci.yml/badge.svg)](https://github.com/davidhein96/davidhein96.github.io/actions/workflows/ci.yml)

Personal portfolio and technical blog — write-ups on research and engineering projects.

**[davidhein96.github.io](https://davidhein96.github.io)**

## Tech Stack

- [Astro](https://astro.build) — static site generator
- [Tailwind CSS](https://tailwindcss.com) — utility-first styling
- [MDX](https://mdxjs.com) — markdown with components for blog posts
- [Sharp](https://sharp.pixelplumbing.com) — image optimization

## Local Development

```bash
pnpm install
pnpm dev          # dev server at localhost:4321
pnpm check        # type checking
pnpm build        # production build → docs/
pnpm preview      # preview production build
```

## Project Structure

```
src/
├── content/posts/   # MDX blog posts (frontmatter-driven)
├── components/      # Astro components
├── layouts/         # Base and post layouts
├── pages/           # File-based routing
├── styles/          # Global CSS
└── assets/          # Static images (optimized at build)
docs/                # Built site (GitHub Pages)
```

## Deployment

The site is built to `docs/` and served via GitHub Pages from the `main` branch. CI runs type checking, Lighthouse audits, and link checking on every push.
