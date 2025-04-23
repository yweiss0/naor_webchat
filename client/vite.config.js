import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  build: {
    outDir: path.resolve(__dirname, 'static/dist/standalone'),
    lib: {
      name: "svelteWebComponents",
      entry: "src/main.js",
      formats: ["iife"],
      fileName: "swc"
    },
    emptyOutDir: true,
  },
  plugins: [svelte()],
});
