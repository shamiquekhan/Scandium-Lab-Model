'use client';

import { motion } from 'framer-motion';
import AnimatedCounter from '@/components/ui/AnimatedCounter';
import { FadeUp, SectionLabel } from '@/components/ui/Animations';

const stats = [
  {
    roman: 'I.',
    label: 'Materials in Training',
    value: 154000,
    suffix: '+',
    img: 'https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?w=400&q=80',
  },
  {
    roman: 'II.',
    label: 'Faster Than DFT',
    value: 10000,
    suffix: '×',
    img: 'https://images.unsplash.com/photo-1635070041078-e363dbe005cb?w=400&q=80',
  },
  {
    roman: 'III.',
    label: 'Band Gap Accuracy (eV)',
    value: 0.12,
    suffix: '',
    img: 'https://images.unsplash.com/photo-1628595351029-c2bf17511435?w=400&q=80',
  },
  {
    roman: 'IV.',
    label: 'Properties Predicted',
    value: 6,
    suffix: '',
    img: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&q=80',
  },
];

export default function AboutStats() {
  return (
    <section id="about" className="theme-dark bg-background text-primary-text py-28" style={{ background: 'var(--background)', color: 'var(--primary-text)' }}>
      <div className="max-w-[1400px] mx-auto px-6">
        {/* Header */}
      <FadeUp className="mb-16">
        <SectionLabel label="ABOUT US" />
        <p
          className="text-base leading-relaxed max-w-2xl"
          style={{ color: 'var(--secondary-text)' }}
        >
          Founded on the principles of scientific rigour and computational first principles, Scandium Labs 
          is building the AI infrastructure for the next generation of materials discovery. We are a team 
          of computational physicists and ML researchers who believe that the models predicting tomorrow's 
          materials must be grounded in the physics of today. Our Physics-Informed Graph Networks don't 
          just learn from data — they respect the laws of nature.
        </p>
      </FadeUp>

      {/* Stat cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        {stats.map((stat, i) => (
          <FadeUp key={stat.roman} delay={i * 0.1}>
            <motion.div
              whileHover={{ y: -8 }}
              transition={{ duration: 0.2 }}
              className="relative overflow-hidden rounded-xl border p-6 flex flex-col justify-between min-h-[220px]"
              style={{
                background: 'var(--card-surface)',
                borderColor: 'var(--border-divider)',
              }}
            >
              {/* Background texture image */}
              <div
                className="absolute inset-0 z-0 bg-cover bg-center"
                style={{
                  backgroundImage: `url(${stat.img})`,
                  opacity: 0.08,
                }}
              />

              {/* Content */}
              <div className="relative z-10">
                <div className="flex items-center justify-between mb-4">
                  <span className="text-[11px] tracking-[0.1em]" style={{ color: 'var(--secondary-text)' }}>
                    {stat.roman}
                  </span>
                  <span className="text-[11px] tracking-[0.08em] uppercase" style={{ color: 'var(--secondary-text)' }}>
                    {stat.label}
                  </span>
                </div>
              </div>

              <div className="relative z-10">
                <div
                  className="font-serif font-light leading-none"
                  style={{ fontSize: 'clamp(48px, 5vw, 72px)', color: 'var(--primary-text)' }}
                >
                  <AnimatedCounter target={stat.value} suffix={stat.suffix} duration={1400} />
                </div>
                <div
                  className="mt-3 w-8 h-px"
                  style={{ background: 'var(--accent)' }}
                />
              </div>
            </motion.div>
          </FadeUp>
        ))}
      </div>
      </div>
    </section>
  );
}
