/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        // Each token reads from a CSS variable (defined per-theme in index.css)
        // as an "R G B" triplet, so Tailwind's opacity modifiers (e.g. text-paper/85)
        // keep working across both the dark and light palettes.
        ink: 'rgb(var(--color-ink) / <alpha-value>)',
        surface: 'rgb(var(--color-surface) / <alpha-value>)',
        raised: 'rgb(var(--color-raised) / <alpha-value>)',
        line: 'rgb(var(--color-line) / <alpha-value>)',
        paper: 'rgb(var(--color-paper) / <alpha-value>)',
        muted: 'rgb(var(--color-muted) / <alpha-value>)',
        signal: 'rgb(var(--color-signal) / <alpha-value>)',
        signalDim: 'rgb(var(--color-signal-dim) / <alpha-value>)',
        amber: 'rgb(var(--color-amber) / <alpha-value>)',
        rust: 'rgb(var(--color-rust) / <alpha-value>)',
      },
      fontFamily: {
        display: ['"Fraunces"', 'serif'],
        body: ['"Inter"', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'monospace'],
      },
    },
  },
  plugins: [],
}
