'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { FadeUp, SectionLabel } from '@/components/ui/Animations';

const MARQUEE = 'Flexible Plans, Powerful Results! · Flexible Plans, Powerful Results! · Flexible Plans, Powerful Results! · ';

const plans = {
  monthly: [
    {
      name: 'Research Plan',
      price: '$800',
      period: '/month',
      description: 'For academic researchers and university labs.',
      highlighted: false,
      features: [
        'Access to prediction API (10,000 queries/month)',
        'Band gap, formation energy, elastic moduli endpoints',
        'Materials Project dataset integration',
        'Email support',
        '1 user seat',
      ],
      cta: 'Get Started',
    },
    {
      name: 'Enterprise Plan',
      price: '$4,500',
      period: '/month',
      description: 'For industry R&D teams in battery, semiconductor, and pharma sectors.',
      highlighted: true,
      features: [
        'Everything in Research Plan',
        'Unlimited API queries',
        'Custom model fine-tuning on proprietary datasets',
        'Dedicated Slack channel + priority support',
        'Up to 10 seats',
        'Quarterly research briefings',
      ],
      cta: 'Contact Sales',
    },
  ],
  annual: [
    {
      name: 'Research Plan',
      price: '$640',
      period: '/month',
      description: 'For academic researchers and university labs. Save 20% annually.',
      highlighted: false,
      features: [
        'Access to prediction API (10,000 queries/month)',
        'Band gap, formation energy, elastic moduli endpoints',
        'Materials Project dataset integration',
        'Email support',
        '1 user seat',
      ],
      cta: 'Get Started',
    },
    {
      name: 'Enterprise Plan',
      price: '$3,600',
      period: '/month',
      description: 'For industry R&D teams. Save 20% annually.',
      highlighted: true,
      features: [
        'Everything in Research Plan',
        'Unlimited API queries',
        'Custom model fine-tuning on proprietary datasets',
        'Dedicated Slack channel + priority support',
        'Up to 10 seats',
        'Quarterly research briefings',
      ],
      cta: 'Contact Sales',
    },
  ],
};

export default function Pricing() {
  const [billing, setBilling] = useState<'monthly' | 'annual'>('monthly');
  const activePlans = plans[billing];

  return (
    <section id="pricing" className="theme-light-alt bg-background text-primary-text py-28 border-t" style={{ borderColor: 'var(--border-divider)', background: 'var(--background)', color: 'var(--primary-text)' }}>
      {/* Marquee */}
      <div className="overflow-hidden border-b py-4 mb-20" style={{ borderColor: 'var(--border-divider)' }}>
        <div className="flex whitespace-nowrap animate-marquee-slow">
          {[MARQUEE, MARQUEE].map((t, i) => (
            <span key={i} className="font-serif pr-8" style={{ fontSize: 'clamp(28px, 3.5vw, 48px)', color: 'var(--primary-text)', opacity: 0.75 }}>
              {t}
            </span>
          ))}
        </div>
      </div>

      <div className="max-w-[1400px] mx-auto px-6">
        <FadeUp className="mb-12">
          <SectionLabel label="ACCESS PLANS" />
          <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-6">
            <div>
              <h2 className="font-serif leading-tight mb-4" style={{ fontSize: 'clamp(40px, 5vw, 64px)', color: 'var(--primary-text)' }}>
                Flexible Plans,<br />Powerful Results
              </h2>
              <p className="text-base max-w-xl" style={{ color: 'var(--secondary-text)' }}>
                Whether you're an academic researcher screening perovskites or an enterprise battery team 
                running million-structure campaigns, we have a plan that fits your computational needs and budget.
              </p>
            </div>

            {/* Toggle */}
            <div
              className="flex items-center gap-1 p-1 rounded-full border self-start sm:self-auto"
              style={{ borderColor: 'var(--border-divider)', background: 'var(--card-surface)' }}
            >
              {(['monthly', 'annual'] as const).map((b) => (
                <button
                  key={b}
                  onClick={() => setBilling(b)}
                  className="relative px-5 py-2 text-[12px] tracking-[0.08em] uppercase rounded-full transition-colors duration-200"
                  style={{ color: billing === b ? 'var(--card-surface)' : 'var(--primary-text)' }}
                >
                  {billing === b && (
                    <motion.div
                      layoutId="billing-pill"
                      className="absolute inset-0 rounded-full"
                      style={{ background: 'var(--accent)' }}
                      transition={{ type: 'spring', stiffness: 380, damping: 35 }}
                    />
                  )}
                  <span className="relative z-10">{b === 'monthly' ? 'Monthly' : 'Annual · Save 20%'}</span>
                </button>
              ))}
            </div>
          </div>
        </FadeUp>

        {/* Plan cards */}
        <div className="grid md:grid-cols-2 gap-6 max-w-4xl mx-auto">
            {activePlans.map((plan, i) => (
              <FadeUp key={`${billing}-${plan.name}`} delay={i * 0.1}>
                <div
                  className="rounded-2xl border p-8 flex flex-col h-full transition-all duration-300 relative overflow-hidden"
                  style={{
                    background: 'var(--card-surface)',
                    borderColor: plan.highlighted ? 'var(--primary-text)' : 'var(--border-divider)',
                    boxShadow: plan.highlighted ? '0 8px 30px rgba(0,0,0,0.08)' : 'none',
                  }}
                >
                  <div className="mb-6 relative z-10">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-[11px] tracking-[0.1em] uppercase font-semibold" style={{ color: plan.highlighted ? 'var(--primary-text)' : 'var(--secondary-text)' }}>
                        {plan.name}
                      </p>
                      {plan.highlighted && (
                        <span className="text-[10px] tracking-[0.08em] px-2 py-0.5 rounded-full" style={{ background: 'var(--primary-text)', color: 'var(--card-surface)' }}>
                          POPULAR
                        </span>
                      )}
                    </div>
                    <div className="flex items-end gap-1 mb-3">
                      <span className="font-serif" style={{ fontSize: 'clamp(36px, 4vw, 52px)', color: 'var(--primary-text)', lineHeight: 1 }}>
                        {plan.price}
                      </span>
                      <span className="text-sm pb-1" style={{ color: 'var(--secondary-text)' }}>{plan.period}</span>
                    </div>
                    <p className="text-sm" style={{ color: 'var(--secondary-text)' }}>{plan.description}</p>
                  </div>

                  <div className="border-t mb-6" style={{ borderColor: 'var(--border-divider)' }} />

                  <ul className="flex flex-col gap-3 flex-1 mb-8">
                    {plan.features.map((f, fi) => (
                      <li key={fi} className="flex items-start gap-3 text-sm relative z-10" style={{ color: 'var(--primary-text)' }}>
                        <span style={{ color: 'var(--primary-text)', marginTop: 1, fontWeight: 'bold' }}>✓</span>
                        {f}
                      </li>
                    ))}
                  </ul>

                  <a
                    href="#contact"
                    className="inline-flex items-center justify-center gap-2 px-6 py-3 text-sm font-medium rounded-full transition-all duration-200 hover:opacity-90 active:scale-[0.97] relative z-10"
                    style={
                      plan.highlighted
                        ? { background: 'var(--primary-text)', color: 'var(--card-surface)' }
                        : { border: '1px solid var(--border-divider)', color: 'var(--primary-text)' }
                    }
                  >
                    {plan.cta} →
                  </a>
                </div>
              </FadeUp>
            ))}
        </div>
      </div>
    </section>
  );
}
