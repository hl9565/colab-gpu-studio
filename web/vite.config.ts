import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

declare const process: { env: { VITE_DEV_API_TARGET?: string } };

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 4173,
    proxy: { '/api': process.env.VITE_DEV_API_TARGET || 'http://127.0.0.1:8788' },
  },
});
