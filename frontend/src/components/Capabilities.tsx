'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FadeUp, SectionLabel } from '@/components/ui/Animations';

const capabilities = [
  {
    id: 1,
    title: 'Property Prediction',
    icon: '⚛',
    subFeatures: ['I. Band Gap Prediction', 'II. Formation Energy', 'III. Elastic Moduli', 'IV. Thermal Conductivity'],
    img: 'https://images.unsplash.com/photo-1628595351029-c2bf17511435?w=800&q=80',
    desc: 'Predict key electronic and thermal properties of crystalline materials directly from atomic structure using physics-constrained GNNs.',
  },
  {
    id: 2,
    title: 'Stability Screening',
    icon: '🛡',
    subFeatures: ['I. Thermodynamic Stability', 'II. Phase Diagram Mapping', 'III. Synthesisability Scoring'],
    img: 'https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?w=800&q=80',
    desc: 'Screen millions of candidate structures for thermodynamic stability and synthesisability at DFT-quality accuracy.',
  },
  {
    id: 3,
    title: 'Novel Material Generation',
    icon: '✦',
    subFeatures: ['I. Inverse Design', 'II. Composition Optimisation', 'III. Out-of-distribution Extrapolation'],
    img: 'https://images.unsplash.com/photo-1635070041078-e363dbe005cb?w=800&q=80',
    desc: 'Generate entirely new material compositions with targeted properties using our constrained inverse-design pipeline.',
  },
  {
    id: 4,
    title: 'Research API & Integration',
    icon: '</> ',
    subFeatures: ['I. Python SDK', 'II. Materials Project Integration', 'III. REST API', 'IV. Batch Screening'],
    img: 'https://images.unsplash.com/photo-1555066931-4365d14bab8c?w=800&q=80',
    desc: 'Plug our prediction engine directly into your pymatgen / ASE workflows via a clean REST API and Python SDK.',
  },
];

export default function Capabilities() {
  const [activeId, setActiveId] = useState(1);
  const active = capabilities.find((c) => c.id === activeId)!;

  return (
    <section id="capabilities" className="theme-light-alt bg-background text-primary-text py-28 border-t relative overflow-hidden" style={{ borderColor: 'var(--border-divider)', background: 'var(--background)', color: 'var(--primary-text)' }}>


      <div className="max-w-[1400px] mx-auto px-6 relative z-10">
        {/* Header */}
        <FadeUp className="mb-16">
          <SectionLabel label="OUR CAPABILITIES" />
          <h2
            className="font-serif leading-tight mb-4"
            style={{ fontSize: 'clamp(40px, 5vw, 64px)', color: 'var(--primary-text)' }}
          >
            What We Build
          </h2>
          <p className="text-base max-w-xl" style={{ color: 'var(--secondary-text)' }}>
            From single-property prediction to full-scale batch screening, we offer the complete AI toolkit 
            for computational materials discovery. Every prediction is physics-constrained. Every output is trustworthy.
          </p>
        </FadeUp>

        {/* Two-column layout */}
        <div className="grid md:grid-cols-2 gap-8 items-start">
          {/* Accordion list */}
          <div className="flex flex-col divide-y" style={{ borderTop: '1px solid var(--border-divider)', borderColor: 'var(--border-divider)' }}>
            {capabilities.map((cap) => {
              const isOpen = activeId === cap.id;
              return (
                <div key={cap.id} style={{ borderColor: 'var(--border-divider)' }}>
                  <button
                    className="w-full flex items-center justify-between py-5 text-left group"
                    onClick={() => setActiveId(cap.id)}
                  >
                    <div className="flex items-center gap-4">
                      <span className="text-xl w-8">{cap.icon}</span>
                      <span
                        className="text-base font-medium transition-colors"
                        style={{ color: isOpen ? 'var(--accent)' : 'var(--primary-text)' }}
                      >
                        {cap.title}
                      </span>
                    </div>
                    <span
                      className="text-lg transition-transform duration-300"
                      style={{
                        color: isOpen ? 'var(--accent)' : 'var(--secondary-text)',
                        transform: isOpen ? 'rotate(45deg)' : 'rotate(0deg)',
                      }}
                    >
                      +
                    </span>
                  </button>

                  <AnimatePresence initial={false}>
                    {isOpen && (
                      <motion.div
                        key="content"
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.32, ease: [0.4, 0, 0.2, 1] }}
                        style={{ overflow: 'hidden' }}
                      >
                        <div className="pb-5 pl-12">
                          <p className="text-sm mb-4 leading-relaxed" style={{ color: 'var(--secondary-text)' }}>
                            {cap.desc}
                          </p>
                          <ul className="flex flex-col gap-2">
                            {cap.subFeatures.map((f) => (
                              <li key={f} className="text-[13px]" style={{ color: 'var(--secondary-text)' }}>
                                {f}
                              </li>
                            ))}
                          </ul>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              );
            })}
          </div>

          {/* Swapping image */}
          <div className="relative h-[420px] rounded-xl overflow-hidden sticky top-24">
            <AnimatePresence mode="wait">
              <motion.img
                key={active.id}
                src={active.img}
                alt={active.title}
                initial={{ opacity: 0, scale: 1.04 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.97 }}
                transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
                className="w-full h-full object-cover"
              />
            </AnimatePresence>
            <div
              className="absolute inset-0"
              style={{ background: 'linear-gradient(to top, rgba(10,10,10,0.7) 0%, transparent 60%)' }}
            />
            <div className="absolute bottom-5 left-5">
              <p className="text-xs tracking-[0.1em]" style={{ color: 'var(--accent)' }}>
                {active.title.toUpperCase()}
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
