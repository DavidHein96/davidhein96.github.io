import { defineCollection, z } from 'astro:content';

const posts = defineCollection({
  type: 'content',
  schema: ({ image }) =>
    z.object({
      title: z.string(),
      date: z.string(), // ISO format: YYYY-MM-DD — used for sorting
      categories: z.array(z.string()),
      description: z.string(),
      cover: image().optional(), // local image file, e.g. ./cover.jpg
      coverPosition: z.string().optional().default('center'), // CSS object-position, e.g. "top", "center 30%"
      draft: z.boolean().optional().default(false),
    }),
});

export const collections = { posts };
