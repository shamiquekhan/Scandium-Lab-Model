'use client';

import { motion } from 'framer-motion';

interface Props {
  label: string;
  title?: string;
  children?: React.ReactNode;
}

const wordFadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.08, duration: 0.6, ease: [0.22, 1, 0.36, 1] },
  }),
};

export function SectionLabel({ label }: { label: string }) {
  return (
    <span
      className="text-[11px] tracking-[0.1em] font-medium block mb-4"
      style={{ color: 'var(--secondary-text)' }}
    >
      {label}
    </span>
  );
}

export function StaggerText({ text, className = '' }: { text: string; className?: string }) {
  const words = text.split(' ');
  return (
    <span className={`inline-flex flex-wrap gap-x-3 ${className}`}>
      {words.map((word, i) => (
        <motion.span
          key={i}
          custom={i}
          variants={wordFadeUp}
          initial="hidden"
          animate="visible"
          className="inline-block"
        >
          {word}
        </motion.span>
      ))}
    </span>
  );
}

export function FadeUp({
  children,
  delay = 0,
  className = '',
}: {
  children: React.ReactNode;
  delay?: number;
  className?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.2 }}
      transition={{ duration: 0.6, delay, ease: [0.22, 1, 0.36, 1] }}
      className={className}
    >
      {children}
    </motion.div>
  );
}
