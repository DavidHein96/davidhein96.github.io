import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import mdx from '@astrojs/mdx';

export default defineConfig({
  site: 'https://davidhein96.github.io',
  outDir: '../docs',
  integrations: [
    tailwind({ applyBaseStyles: false }),
    mdx(),
  ],
});
