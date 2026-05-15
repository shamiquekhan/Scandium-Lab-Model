'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { FadeUp, SectionLabel } from '@/components/ui/Animations';
import { Button } from '@/components/ui/button';

const MARQUEE = 'Let\'s Connect · Let\'s Connect · Let\'s Connect · Let\'s Connect · Let\'s Connect · ';

export default function Contact() {
  const [form, setForm] = useState({ name: '', institution: '', email: '', message: '' });
  const [sent, setSent] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSent(true);
    setTimeout(() => setSent(false), 4000);
    setForm({ name: '', institution: '', email: '', message: '' });
  };

  const inputStyle = {
    background: 'var(--card-surface)',
    border: '1px solid var(--border-divider)',
    color: 'var(--primary-text)',
    borderRadius: '6px',
    fontSize: '14px',
  };

  const inputFocusClass =
    'w-full px-4 py-3 text-sm transition-all duration-200 focus:outline-none focus:border-accent placeholder:text-secondary-text';

  return (
    <section id="contact" className="bg-background text-primary-text py-28 border-t" style={{ borderColor: 'var(--border-divider)', background: 'var(--background)', color: 'var(--primary-text)' }}>
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
        <FadeUp className="mb-14">
          <SectionLabel label="LET'S CONNECT" />
          <h2 className="font-serif leading-tight mb-4" style={{ fontSize: 'clamp(40px, 5vw, 64px)', color: 'var(--primary-text)' }}>
            Start a Conversation
          </h2>
          <p className="text-base max-w-xl" style={{ color: 'var(--secondary-text)' }}>
            Whether you want to discuss a research collaboration, an enterprise pilot, a grant partnership, 
            or just want to understand what our models can do for your specific material class — our science team is here.
          </p>
        </FadeUp>

        <div className="grid md:grid-cols-2 gap-12 items-start">
          {/* Form */}
          <FadeUp>
            <form onSubmit={handleSubmit} className="flex flex-col gap-4">
              <div className="grid sm:grid-cols-2 gap-4">
                <div>
                  <label className="text-[11px] tracking-[0.08em] block mb-2" style={{ color: 'var(--secondary-text)' }}>NAME</label>
                  <input
                    type="text"
                    name="name"
                    required
                    value={form.name}
                    onChange={handleChange}
                    placeholder="Dr. Jane Smith"
                    className={inputFocusClass}
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label className="text-[11px] tracking-[0.08em] block mb-2" style={{ color: 'var(--secondary-text)' }}>INSTITUTION</label>
                  <input
                    type="text"
                    name="institution"
                    value={form.institution}
                    onChange={handleChange}
                    placeholder="IIT / Industry"
                    className={inputFocusClass}
                    style={inputStyle}
                  />
                </div>
              </div>

              <div>
                <label className="text-[11px] tracking-[0.08em] block mb-2" style={{ color: 'var(--secondary-text)' }}>EMAIL</label>
                <input
                  type="email"
                  name="email"
                  required
                  value={form.email}
                  onChange={handleChange}
                  placeholder="you@institution.ac.in"
                  className={inputFocusClass}
                  style={inputStyle}
                />
              </div>

              <div>
                <label className="text-[11px] tracking-[0.08em] block mb-2" style={{ color: 'var(--secondary-text)' }}>MESSAGE</label>
                <textarea
                  name="message"
                  required
                  rows={5}
                  value={form.message}
                  onChange={handleChange}
                  placeholder="Tell us about your research or project..."
                  className={inputFocusClass}
                  style={{ ...inputStyle, resize: 'none' }}
                />
              </div>

              <Button
                type="submit"
                disabled={sent}
                className="w-full py-6 text-sm font-semibold mt-4 transition-all duration-200 active:scale-[0.98]"
              >
                {sent ? '✓ Message Sent!' : 'Send Message →'}
              </Button>
            </form>
          </FadeUp>

          {/* Contact details card */}
          <FadeUp delay={0.15}>
            <div className="relative rounded-2xl border overflow-hidden p-8" style={{ background: 'var(--card-surface)', borderColor: 'var(--border-divider)' }}>
              {/* Decorative molecule bg */}
              <div
                className="absolute inset-0 bg-cover bg-center opacity-[0.04]"
                style={{ backgroundImage: 'url(https://images.unsplash.com/photo-1635070041078-e363dbe005cb?w=800&q=60)' }}
              />
              <div className="relative z-10 flex flex-col gap-8">
                <div>
                  <p className="text-[11px] tracking-[0.1em] mb-3" style={{ color: 'var(--secondary-text)' }}>EMAIL</p>
                  <a href="mailto:contact@scandiumlabs.ai" className="font-serif text-2xl hover:opacity-70 transition-opacity" style={{ color: 'var(--primary-text)' }}>
                    contact@scandiumlabs.ai
                  </a>
                </div>
                <div>
                  <p className="text-[11px] tracking-[0.1em] mb-3" style={{ color: 'var(--secondary-text)' }}>PHONE</p>
                  <a href="tel:+919876543210" className="font-serif text-2xl hover:opacity-70 transition-opacity" style={{ color: 'var(--primary-text)' }}>
                    +91 98765 43210
                  </a>
                </div>
                <div>
                  <p className="text-[11px] tracking-[0.1em] mb-3" style={{ color: 'var(--secondary-text)' }}>LOCATION</p>
                  <p className="text-base" style={{ color: 'var(--primary-text)' }}>IIT Research Ecosystem<br />India</p>
                </div>

                <div className="border-t pt-6" style={{ borderColor: 'var(--border-divider)' }}>
                  <p className="text-[11px] tracking-[0.1em] mb-4" style={{ color: 'var(--secondary-text)' }}>FIND US</p>
                  <div className="flex gap-4">
                    {['LinkedIn', 'Twitter/X', 'GitHub', 'ResearchGate'].map((s) => (
                      <a
                        key={s}
                        href="#"
                        className="text-xs px-3 py-2 rounded-md border transition-all duration-200 hover:border-white/20"
                        style={{ borderColor: 'var(--border-divider)', color: 'var(--secondary-text)' }}
                      >
                        {s}
                      </a>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </FadeUp>
        </div>
      </div>
    </section>
  );
}
