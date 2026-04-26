/// <reference types="vitest" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  test: {
    include: ['tests/**/*.test.ts'],
  },
  server: {
    fs: {
      allow: ['../..', '/tmp'],
    },
  },
});
