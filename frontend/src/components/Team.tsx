'use client';

import { motion } from 'framer-motion';
import { FadeUp, SectionLabel } from '@/components/ui/Animations';

const team = [
  {
    initials: 'AK',
    name: 'Dr. Arjun Kumar',
    title: 'Founder & Chief Scientist',
    bio: 'Leads physics-constrained ML architecture and research direction. Formerly computational physics at IIT Bombay.',
    img: 'https://images.unsplash.com/photo-1633332755192-727a05c4013d?w=400&q=80',
  },
  {
    initials: 'PR',
    name: 'Priya Rajan',
    title: 'ML Research Lead',
    bio: 'Specialises in graph neural network architecture for crystal property prediction and inverse design.',
    img: 'https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=400&q=80',
  },
  {
    initials: 'SM',
    name: 'Siddharth Menon',
    title: 'Computational Chemistry',
    bio: 'DFT expert bridging quantum mechanics and ML training pipelines across diverse material classes.',
    img: 'https://images.unsplash.com/photo-1531427186611-ecfd6d936c79?w=400&q=80',
  },
  {
    initials: 'RV',
    name: 'Riya Varma',
    title: 'Software Engineering',
    bio: 'Builds the prediction API and data infrastructure on top of the Materials Project ecosystem.',
    img: 'https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=400&q=80',
  },
  {
    initials: 'KS',
    name: 'Karan Shah',
    title: 'Business & Partnerships',
    bio: 'Connects research outputs to battery, semiconductor, and pharma industry clients globally.',
    img: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&q=80',
  },
];

export default function Team() {
  return (
    <section id="team" className="theme-dark bg-background text-primary-text py-28 border-t" style={{ borderColor: 'var(--border-divider)', background: 'var(--background)', color: 'var(--primary-text)' }}>
      <div className="max-w-[1400px] mx-auto px-6">
        {/* Header */}
        <FadeUp className="mb-16">
          <SectionLabel label="OUR TEAM" />
          <h2
            className="font-serif leading-tight mb-4"
            style={{ fontSize: 'clamp(40px, 5vw, 64px)', color: 'var(--primary-text)' }}
          >
            The Minds Behind It
          </h2>
          <p className="text-base max-w-xl" style={{ color: 'var(--secondary-text)' }}>
            A tight-knit group of physicists, ML researchers, and computational chemists building
            at the frontier of AI for science.
          </p>
        </FadeUp>

        {/* Team grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-5">
          {team.map((member, i) => (
            <FadeUp key={member.name} delay={i * 0.08}>
              <motion.div
                whileHover={{ scale: 1.02 }}
                transition={{ duration: 0.2, ease: 'easeOut' }}
                className="rounded-xl border overflow-hidden flex flex-col"
                style={{ background: 'var(--card-surface)', borderColor: 'var(--border-divider)' }}
              >
                {/* Portrait */}
                <div className="relative h-52 overflow-hidden">
                  <img
                    src={member.img}
                    alt={member.name}
                    className="w-full h-full object-cover object-top transition-transform duration-500 hover:scale-105"
                    style={{ filter: 'grayscale(20%) brightness(0.85)' }}
                  />
                  <div
                    className="absolute inset-0"
                    style={{ background: 'linear-gradient(to top, rgba(17,17,17,0.8) 0%, transparent 55%)' }}
                  />
                </div>

                {/* Info */}
                <div className="p-5 flex flex-col gap-2 flex-1">
                  <div>
                    <p className="font-medium text-sm" style={{ color: 'var(--primary-text)' }}>
                      {member.name}
                    </p>
                    <p className="text-[11px] tracking-[0.05em] uppercase mt-0.5" style={{ color: 'var(--accent)' }}>
                      {member.title}
                    </p>
                  </div>
                  <p className="text-[12px] leading-relaxed mt-1" style={{ color: 'var(--secondary-text)' }}>
                    {member.bio}
                  </p>
                </div>
              </motion.div>
            </FadeUp>
          ))}
        </div>

        {/* Join CTA card */}
        <FadeUp delay={0.3} className="mt-12">
          <div
            className="rounded-2xl border p-10 md:p-14 flex flex-col md:flex-row items-start md:items-center justify-between gap-8"
            style={{ background: 'var(--card-surface)', borderColor: 'var(--border-divider)' }}
          >
            <div>
              <SectionLabel label="JOIN THE TEAM" />
              <h3
                className="font-serif leading-tight mb-3"
                style={{ fontSize: 'clamp(28px, 3.5vw, 48px)', color: 'var(--primary-text)' }}
              >
                Join the Scandium Labs<br />research team.
              </h3>
              <p className="text-sm max-w-lg" style={{ color: 'var(--secondary-text)' }}>
                We&apos;re looking for physicists, ML engineers, and materials scientists who want to
                work at the frontier of AI for science.
              </p>
            </div>
            <a
              href="#contact"
              className="shrink-0 inline-flex items-center gap-2 px-8 py-4 text-sm font-medium rounded-full transition-all duration-200 hover:opacity-90 active:scale-[0.97]"
              style={{ background: 'var(--accent)', color: '#0A0A0A' }}
            >
              Send Your CV →
            </a>
          </div>
        </FadeUp>
      </div>
    </section>
  );
}
