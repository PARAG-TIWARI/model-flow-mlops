import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    port: 3000,
    proxy: {
      '/health': 'http://127.0.0.1:8000',
      '/model': 'http://127.0.0.1:8000',
      '/predict': 'http://127.0.0.1:8000',
      '/metrics': 'http://127.0.0.1:8000',
      '/monitoring': 'http://127.0.0.1:8000',
      '/experiments': 'http://127.0.0.1:8000',
    }
  }
})
