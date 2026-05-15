'use client';

import { motion } from 'framer-motion';
import { FadeUp, SectionLabel } from '@/components/ui/Animations';

const pubs = [
  {
    roman: 'I.',
    name: '"Physics-Constrained GNNs for Material Property Prediction"',
    venue: 'arXiv Preprint',
    year: '2024',
    img: 'https://images.unsplash.com/photo-1635070041078-e363dbe005cb?w=1200&q=60',
  },
  {
    roman: 'II.',
    name: 'Materials Project API Dataset — 154,000 Compounds Modelled',
    venue: 'Dataset',
    year: '2024',
    img: 'https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?w=1200&q=60',
  },
  {
    roman: 'III.',
    name: 'ICLR ML4Science Workshop — Accepted Abstract',
    venue: 'ICLR 2025',
    year: '2025',
    img: 'https://images.unsplash.com/photo-1628595351029-c2bf17511435?w=1200&q=60',
  },
  {
    roman: 'IV.',
    name: 'DST-SERB Grant — Computational Materials Research · ₹35L',
    venue: 'Government Grant',
    year: '2024',
    img: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=1200&q=60',
  },
  {
    roman: 'V.',
    name: 'DeepMind GNoME Follow-up — Referenced in Community Discussion',
    venue: 'Community Recognition',
    year: '2024',
    img: 'https://images.unsplash.com/photo-1555066931-4365d14bab8c?w=1200&q=60',
  },
  {
    roman: 'VI.',
    name: 'IIT Research Excellence — Computational Science Track',
    venue: 'Institutional Award',
    year: '2024',
    img: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=1200&q=60',
  },
];

export default function Publications() {
  return (
    <section id="publications" className="bg-background text-primary-text py-28 border-t" style={{ borderColor: 'var(--border-divider)', background: 'var(--background)', color: 'var(--primary-text)' }}>
      <div className="max-w-[1400px] mx-auto px-6 mb-16">
        <FadeUp>
          <SectionLabel label="PUBLICATIONS & RECOGNITION" />
          <h2
            className="font-serif leading-tight"
            style={{ fontSize: 'clamp(40px, 5vw, 64px)', color: 'var(--primary-text)' }}
          >
            Our Work, Recognised
          </h2>
        </FadeUp>
      </div>

      {/* Publication strips */}
      <div className="flex flex-col divide-y" style={{ borderTop: '1px solid var(--border-divider)', borderColor: 'var(--border-divider)' }}>
        {pubs.map((pub, i) => (
          <motion.div
            key={pub.roman}
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.2 }}
            transition={{ duration: 0.5, delay: i * 0.06 }}
            whileHover={{ translateY: -2 }}
            className="relative group overflow-hidden"
            style={{ borderColor: 'var(--border-divider)' }}
          >
            {/* Background texture */}
            <div
              className="absolute inset-0 transition-opacity duration-300 bg-cover bg-center opacity-0 group-hover:opacity-[0.07]"
              style={{ backgroundImage: `url(${pub.img})` }}
            />

            <div className="relative z-10 max-w-[1400px] mx-auto px-6 py-7 flex flex-col sm:flex-row sm:items-center gap-4 sm:gap-8">
              <span
                className="text-[11px] tracking-[0.1em] shrink-0 w-8"
                style={{ color: 'var(--secondary-text)' }}
              >
                {pub.roman}
              </span>

              <span
                className="font-serif flex-1 leading-snug"
                style={{ fontSize: 'clamp(18px, 2.2vw, 28px)', color: 'var(--primary-text)' }}
              >
                {pub.name}
              </span>

              <div className="flex items-center gap-4 shrink-0">
                <span
                  className="text-[11px] tracking-[0.08em] uppercase hidden sm:block"
                  style={{ color: 'var(--secondary-text)' }}
                >
                  {pub.venue}
                </span>
                <span
                  className="text-sm font-medium px-3 py-1 rounded-full border"
                  style={{
                    color: 'var(--accent)',
                    borderColor: 'rgba(200,240,96,0.3)',
                    background: 'rgba(200,240,96,0.06)',
                  }}
                >
                  {pub.year}
                </span>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
