import type { Config } from "tailwindcss"

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: 'var(--background)',
        'primary-text': 'var(--primary-text)',
        'secondary-text': 'var(--secondary-text)',
        accent: 'var(--accent)',
        'accent-secondary': 'var(--accent-secondary)',
        'card-surface': 'var(--card-surface)',
        'border-divider': 'var(--border-divider)',
        'scandium-dark': '#080808',
        'scandium-primary': '#2DD4BF',
        'scandium-secondary': '#22D3EE',
      },
      fontFamily: {
        serif: ['Playfair Display', 'Cormorant Garamond', 'serif'],
        sans: ['Inter', 'DM Sans', 'sans-serif'],
      },
      letterSpacing: {
        'wide-caps': '0.1em',
      },
      animation: {
        'marquee': 'marquee 40s linear infinite',
        'marquee-reverse': 'marquee-reverse 40s linear infinite',
        'marquee-slow': 'marquee 70s linear infinite',
      },
      keyframes: {
        marquee: {
          '0%':   { transform: 'translateX(0)' },
          '100%': { transform: 'translateX(-50%)' },
        },
        'marquee-reverse': {
          '0%':   { transform: 'translateX(-50%)' },
          '100%': { transform: 'translateX(0)' },
        },
      },
    },
  },
  plugins: [],
}

export default config
