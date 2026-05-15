'use client';

import { Button } from '@/components/ui/button';

const footerCols = [
  {
    label: 'COMPANY',
    links: ['About Us', 'Research', 'Team', 'Publications'],
    hrefs: ['#about', '#research', '#team', '#publications'],
  },
  {
    label: 'CAPABILITIES',
    links: ['Property Prediction', 'Stability Screening', 'Material Generation', 'Research API'],
    hrefs: ['#capabilities', '#capabilities', '#capabilities', '#capabilities'],
  },
];

export default function Footer() {
  return (
    <footer className="theme-dark bg-background text-primary-text relative border-t overflow-hidden" style={{ borderColor: 'var(--border-divider)' , background: 'var(--background)', color: 'var(--primary-text)' }}>
      {/* Watermark wordmark */}
      <div
        className="absolute bottom-0 left-0 right-0 flex items-end justify-center pointer-events-none select-none overflow-hidden"
        style={{ height: '160px' }}
        aria-hidden
      >
        <span
          className="font-serif font-bold whitespace-nowrap leading-none translate-y-[30%]"
          style={{
            fontSize: 'clamp(80px, 14vw, 180px)',
            color: 'var(--primary-text)',
            opacity: 0.04,
            letterSpacing: '-0.02em',
          }}
        >
          SCANDIUM LABS
        </span>
      </div>

      {/* Main footer content */}
      <div className="relative z-10 max-w-[1400px] mx-auto px-6 pt-20 pb-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-12 mb-16">
          {/* Col 1 & 2 – nav links */}
          {footerCols.map((col) => (
            <div key={col.label}>
              <p className="text-[11px] tracking-[0.12em] mb-6" style={{ color: 'var(--secondary-text)' }}>
                {col.label}
              </p>
              <ul className="flex flex-col gap-3">
                {col.links.map((link, i) => (
                  <li key={link}>
                    <a
                      href={col.hrefs[i]}
                      className="text-sm transition-colors duration-200 hover:opacity-60"
                      style={{ color: 'var(--primary-text)' }}
                    >
                      {link}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}

          {/* Col 3 – Reach out */}
          <div>
            <p className="text-[11px] tracking-[0.12em] mb-6" style={{ color: 'var(--secondary-text)' }}>
              REACH OUT
            </p>
            <a
              href="tel:+919876543210"
              className="block font-serif text-2xl mb-2 hover:opacity-70 transition-opacity"
              style={{ color: 'var(--primary-text)' }}
            >
              +91 98765 43210
            </a>
            <a
              href="mailto:contact@scandiumlabs.ai"
              className="block text-sm hover:opacity-70 transition-opacity"
              style={{ color: 'var(--secondary-text)' }}
            >
              contact@scandiumlabs.ai
            </a>
          </div>

          {/* Col 4 – CTA + Socials */}
          <div className="flex flex-col gap-6">
            <div>
              <p className="text-[11px] tracking-[0.12em] mb-6" style={{ color: 'var(--secondary-text)' }}>
                RESOURCES
              </p>
              <Button variant="outline" className="rounded-full px-6 py-5 hover:bg-[var(--card-surface)]" asChild>
                <a href="#">Download Pitch Deck →</a>
              </Button>
            </div>
            <div>
              <p className="text-[11px] tracking-[0.12em] mb-4" style={{ color: 'var(--secondary-text)' }}>
                SOCIALS
              </p>
              <div className="flex flex-wrap gap-3">
                {['LinkedIn', 'Twitter/X', 'GitHub', 'ResearchGate'].map((s) => (
                  <a
                    key={s}
                    href="#"
                    className="text-xs px-3 py-1.5 rounded-full border transition-all duration-200 hover:border-white/20 hover:text-white"
                    style={{ borderColor: 'var(--border-divider)', color: 'var(--secondary-text)' }}
                  >
                    {s}
                  </a>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Bottom bar */}
        <div
          className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-6 border-t text-center sm:text-left"
          style={{ borderColor: 'var(--border-divider)' }}
        >
          <span className="text-[11px] tracking-[0.1em]" style={{ color: 'var(--secondary-text)' }}>
            BUILDING THE PERIODIC TABLE OF THE FUTURE — FROM INDIA, FOR THE WORLD
          </span>
          <span className="text-[11px]" style={{ color: 'var(--secondary-text)' }}>
            © 2025 Scandium Labs. All Rights Reserved.
          </span>
          {/* Sc monogram */}
          <div className="flex items-center gap-2">
            <span
              className="w-7 h-7 rounded-sm flex items-center justify-center text-background text-xs font-bold font-serif"
              style={{ background: 'var(--accent)' }}
            >
              Sc
            </span>
            <span className="text-[11px] tracking-[0.1em]" style={{ color: 'var(--secondary-text)' }}>
              SCANDIUM LABS
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
}
