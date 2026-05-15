'use client';

import { useState, useEffect } from 'react';
import { ArrowRight } from 'lucide-react';
import { CoolBlobEffect } from '@/components/ui/cool-blob-effect';
import { ParticleTextEffect } from '@/components/ui/interactive-text-particle';

const BG_VIDEO = "https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260508_155101_f2540600-6fe9-433e-8e48-b3f4b72f0727.mp4";

export default function Hero() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    // Small delay ensures CSS transitions trigger properly after the initial render
    const t = setTimeout(() => setMounted(true), 50);
    return () => clearTimeout(t);
  }, []);

  const getTransition = (delayMs: number) => ({
    opacity: mounted ? 1 : 0,
    transform: mounted ? 'translateY(0)' : 'translateY(-8px)',
    transition: `opacity 0.4s cubic-bezier(0.23, 1, 0.32, 1) ${delayMs}ms, transform 0.4s cubic-bezier(0.23, 1, 0.32, 1) ${delayMs}ms`
  });

  return (
    <section id="hero" className="relative w-full h-screen overflow-hidden bg-black" style={{ fontFamily: 'Inter, sans-serif' }}>
      <CoolBlobEffect />

      {/* Top Heading: Absolutely positioned above the centered particle text */}
      <div className="absolute top-[28%] md:top-[32%] left-0 right-0 z-20 flex justify-center pointer-events-none">
        <h2 
          className="text-white/90 font-serif italic text-3xl md:text-4xl lg:text-5xl tracking-wide drop-shadow-xl"
          style={getTransition(50)}
        >
          Where physics meets
        </h2>
      </div>

      {/* Interactive Text Particle Overlay */}
      <div className="absolute inset-0 z-10 flex items-center justify-center pointer-events-auto mt-[4vh] md:mt-[2vh]">
        <ParticleTextEffect 
          text="PREDICTION" 
          colors={['ffffff', 'f3e8ff', '8A2BE2', 'E0B0FF']} 
          particleDensity={5}
        />
      </div>

      <div className="relative z-20 flex flex-col items-center text-center pt-[60vh] md:pt-[65vh] px-5 sm:px-8 pointer-events-none">

        <p
          className="mt-6 md:mt-8 text-white/95 text-lg md:text-2xl leading-relaxed max-w-lg md:max-w-2xl font-medium drop-shadow-lg"
          style={{ 
            fontFamily: "'Courier New', Courier, monospace", 
            letterSpacing: '0.02em',
            textShadow: '0 2px 10px rgba(0,0,0,0.8)',
            ...getTransition(130)
          }}
        >
          predict material properties from atomic structure
          <br className="hidden sm:block" />
          {' '}powered by physics-enforced graph neural networks
        </p>

        <button
          className="mt-7 md:mt-8 flex items-center gap-2.5 px-5 py-2.5 rounded-full text-black text-sm font-medium transition-all duration-300 hover:opacity-80 group pointer-events-auto"
          style={{  
            fontFamily: 'Inter, sans-serif', 
            backgroundColor: '#ffffff',
            ...getTransition(180)
          }}
        >
          Request Access
          <ArrowRight size={15} className="group-hover:translate-x-0.5 transition-transform duration-200" />
        </button>
      </div>
    </section>
  );
}
