'use client';

import { FadeUp, SectionLabel } from '@/components/ui/Animations';

const testimonials = [
  {
    quote: "Scandium Labs' band gap predictions matched our DFT calculations within 0.08 eV — at 10,000× the speed.",
    name: 'Prof. Ramesh Iyer',
    role: 'Materials Science, IISc Bangalore',
  },
  {
    quote: 'The physics constraints are what make us trust this tool. It doesn\'t produce unphysical results.',
    name: 'Dr. Meera Nair',
    role: 'Condensed Matter Physics, TIFR Mumbai',
  },
  {
    quote: 'We screened 50,000 battery cathode candidates in a weekend. That would have taken us 3 years with DFT.',
    name: 'Vikram Sinha',
    role: 'Materials Lead, QuantaBattery',
  },
  {
    quote: 'Their API integrated cleanly into our existing pymatgen workflow with almost zero friction.',
    name: 'Dr. Ananya Roy',
    role: 'Computational Chemist, CSIR-NCL',
  },
  {
    quote: 'The extrapolation accuracy on novel compositions is genuinely impressive — beyond what we expected.',
    name: 'Prof. David Chen',
    role: 'MIT Collaboration, DMSE',
  },
  {
    quote: 'First AI materials tool I\'ve seen that respects thermodynamic constraints by design, not as an afterthought.',
    name: 'Dr. Sunita Patel',
    role: 'Senior Researcher, IIT Delhi',
  },
];

const Row = ({ items, reverse }: { items: typeof testimonials; reverse?: boolean }) => {
  const doubled = [...items, ...items];
  return (
    <div className="overflow-hidden py-3">
      <div className={`flex gap-5 whitespace-nowrap ${reverse ? 'animate-marquee-reverse' : 'animate-marquee'}`}>
        {doubled.map((t, i) => (
          <div
            key={i}
            className="inline-flex flex-col gap-4 shrink-0 w-[340px] rounded-xl border p-6"
            style={{ background: 'var(--card-surface)', borderColor: 'var(--border-divider)', whiteSpace: 'normal' }}
          >
            <p className="text-sm font-medium leading-relaxed" style={{ color: 'var(--primary-text)' }}>
              &ldquo;{t.quote}&rdquo;
            </p>
            <div className="flex items-center gap-3 mt-auto">
              <div
                className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-semibold shrink-0"
                style={{ background: 'rgba(200,240,96,0.15)', color: 'var(--accent)' }}
              >
                {t.name.split(' ').map((n) => n[0]).join('').slice(0, 2)}
              </div>
              <div>
                <p className="text-xs font-medium" style={{ color: 'var(--primary-text)' }}>{t.name}</p>
                <p className="text-[11px]" style={{ color: 'var(--secondary-text)' }}>{t.role}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default function Testimonials() {
  const row1 = testimonials.slice(0, 3);
  const row2 = testimonials.slice(3, 6);

  return (
    <section id="testimonials" className="theme-dark bg-background text-primary-text py-28 border-t overflow-hidden" style={{ borderColor: 'var(--border-divider)', background: 'var(--background)', color: 'var(--primary-text)' }}>
      <div className="max-w-[1400px] mx-auto px-6 mb-14">
        <FadeUp>
          <SectionLabel label="OUR REVIEWS" />
          <h2 className="font-serif leading-tight" style={{ fontSize: 'clamp(40px, 5vw, 64px)', color: 'var(--primary-text)' }}>
            Trusted by Researchers<br />Worldwide
          </h2>
        </FadeUp>
      </div>

      <Row items={row1} />
      <Row items={row2} reverse />
    </section>
  );
}
