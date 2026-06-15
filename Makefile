.PHONY: install dev build preview check format format-check clean

install:
	pnpm install

dev:
	pnpm dev

build:
	pnpm build

preview: build
	pnpm preview

check:
	pnpm check

format:
	pnpm format

format-check:
	pnpm format:check

clean:
	rm -rf dist .astro node_modules/.vite
