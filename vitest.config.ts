import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['src/**/*.test.ts'],
  },
  // opentype.js CJS build conflicts with ESM — use the source version
  resolve: {
    alias: {
      'opentype.js': 'opentype.js/src/opentype.js',
    },
  },
});
