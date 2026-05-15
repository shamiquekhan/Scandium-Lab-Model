'use client';

import { motion } from 'framer-motion';
import { FadeUp, SectionLabel } from '@/components/ui/Animations';

const MARQUEE_TEXT = 'Featured Research — Physics-Constrained Intelligence — ';

const projects = [
  {
    id: 'I.',
    name: 'PINN-GNN for Band Gap Prediction',
    description:
      'Physics-informed loss constraints improve out-of-distribution accuracy by 34% over baseline CGCNN on Materials Project holdout sets.',
    domain: 'Computational Materials',
    status: 'Under Review — npj Computational Materials',
    img: 'https://images.unsplash.com/photo-1635070041078-e363dbe005cb?w=600&q=80',
    year: '2024',
  },
  {
    id: 'II.',
    name: 'Crystal Stability Screening at Scale',
    description:
      'Screening 500,000 candidate structures for thermodynamic stability using our constrained MEGNet variant in under 4 hours.',
    domain: 'High-throughput Discovery',
    status: 'arXiv Preprint · 2024',
    img: 'https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?w=600&q=80',
    year: '2024',
  },
  {
    id: 'III.',
    name: 'Perovskite Solar Absorber Discovery',
    description:
      'Identified 12 novel stable perovskite compositions with predicted band gaps in the 1.2–1.6 eV optimal solar range.',
    domain: 'Energy Materials',
    status: 'In Progress',
    img: 'https://images.unsplash.com/photo-1628595351029-c2bf17511435?w=600&q=80',
    year: '2025',
  },
];

export default function ResearchProjects() {
  return (
    <section id="research" className="theme-dark bg-background text-primary-text py-28 border-t" style={{ borderColor: 'var(--border-divider)', background: 'var(--background)', color: 'var(--primary-text)' }}>
      {/* Marquee header */}
      <div className="overflow-hidden mb-2 border-y py-4" style={{ borderColor: 'var(--border-divider)' }}>
        <div className="flex whitespace-nowrap animate-marquee-slow">
          {[MARQUEE_TEXT, MARQUEE_TEXT].map((t, i) => (
            <span
              key={i}
              className="font-serif pr-8"
              style={{ fontSize: 'clamp(32px, 4vw, 56px)', color: 'var(--primary-text)', opacity: 0.85 }}
            >
              {t}
            </span>
          ))}
        </div>
      </div>

      <div className="max-w-[1400px] mx-auto px-6 mt-20">
        {/* Section header */}
        <FadeUp className="mb-16">
          <SectionLabel label="OUR RESEARCH" />
          <h2
            className="font-serif leading-tight mb-4"
            style={{ fontSize: 'clamp(40px, 5vw, 64px)', color: 'var(--primary-text)' }}
          >
            Peer-Reviewed Work
          </h2>
          <p className="text-base max-w-xl" style={{ color: 'var(--secondary-text)' }}>
            Our published and preprint work forms the scientific foundation of the Scandium Labs platform. 
            Every model we deploy has been benchmarked against state-of-the-art baselines and validated 
            against held-out DFT data.
          </p>
        </FadeUp>

        {/* Cards */}
        <div className="grid md:grid-cols-3 gap-6">
          {projects.map((proj, i) => (
            <FadeUp key={proj.id} delay={i * 0.1}>
              <motion.div
                whileHover={{ scale: 1.02, translateY: -4 }}
                transition={{ duration: 0.2 }}
                className="rounded-xl border overflow-hidden flex flex-col group"
                style={{ background: 'var(--card-surface)', borderColor: 'var(--border-divider)' }}
              >
                {/* Image */}
                <div className="relative h-44 overflow-hidden">
                  <img
                    src={proj.img}
                    alt={proj.name}
                    className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                    style={{ filter: 'brightness(0.7) saturate(0.6)' }}
                  />
                  <div className="absolute top-4 left-4">
                    <span className="text-[11px] tracking-[0.1em] font-medium" style={{ color: 'var(--accent)' }}>
                      {proj.id}
                    </span>
                  </div>
                  <div className="absolute top-4 right-4">
                    <span
                      className="text-[10px] tracking-[0.08em] px-2 py-1 rounded-full border"
                      style={{
                        color: 'var(--secondary-text)',
                        borderColor: 'var(--border-divider)',
                        background: 'rgba(10,10,10,0.6)',
                      }}
                    >
                      {proj.year}
                    </span>
                  </div>
                </div>

                {/* Meta */}
                <div className="p-6 flex flex-col gap-3 flex-1">
                  <span className="text-[11px] tracking-[0.08em] uppercase" style={{ color: 'var(--accent)' }}>
                    {proj.domain}
                  </span>
                  <h3 className="font-serif text-lg leading-snug" style={{ color: 'var(--primary-text)' }}>
                    {proj.name}
                  </h3>
                  <p className="text-sm leading-relaxed flex-1" style={{ color: 'var(--secondary-text)' }}>
                    {proj.description}
                  </p>
                  <div
                    className="mt-auto pt-4 border-t text-[11px] tracking-[0.05em]"
                    style={{ borderColor: 'var(--border-divider)', color: 'var(--secondary-text)' }}
                  >
                    {proj.status}
                  </div>
                </div>
              </motion.div>
            </FadeUp>
          ))}
        </div>

        {/* CTA */}
        <FadeUp delay={0.3} className="mt-12 flex justify-center">
          <a
            href="#publications"
            className="inline-flex items-center gap-2 px-6 py-3 text-sm border rounded-full transition-all duration-200 hover:border-white/30 active:scale-[0.97]"
            style={{ borderColor: 'var(--border-divider)', color: 'var(--primary-text)' }}
          >
            View All Research →
          </a>
        </FadeUp>
      </div>
    </section>
  );
}
