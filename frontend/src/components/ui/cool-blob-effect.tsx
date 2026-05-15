"use client";

import { cn } from "@/lib/utils";

export const CoolBlobEffect = () => {
  return (
    <div className="absolute inset-0 z-0 overflow-hidden bg-black flex items-center justify-center">
      {/* SVG filter for perfectly smooth metaballs */}
      <svg className="hidden">
        <defs>
          <filter id="goo">
            <feGaussianBlur in="SourceGraphic" stdDeviation="30" result="blur" />
            <feColorMatrix in="blur" mode="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 30 -12" result="goo" />
          </filter>
        </defs>
      </svg>
      
      <div className="meta w-full h-full relative" style={{ filter: 'url(#goo)' }}>
        <div 
          className="ball absolute w-[300px] md:w-[500px] aspect-square rounded-full bg-[#8A2BE2] opacity-80 mix-blend-screen"
          style={{
            top: '20%', left: '20%',
            animation: 'rotate 10s infinite linear, move 8s infinite alternate ease-in-out, wobble 9s infinite alternate ease-in-out'
          }}
        />
        <div 
          className="ball absolute w-[250px] md:w-[400px] aspect-square rounded-full bg-[#f3e8ff] opacity-80 mix-blend-screen"
          style={{
            bottom: '10%', right: '15%',
            animation: 'rotate 12s infinite linear reverse, move 10s infinite alternate-reverse ease-in-out, wobble 11s infinite alternate-reverse ease-in-out'
          }}
        />
        <div 
          className="ball absolute w-[350px] md:w-[450px] aspect-square rounded-full bg-[#4B0082] opacity-80 mix-blend-screen"
          style={{
            top: '40%', left: '50%',
            transform: 'translateX(-50%)',
            animation: 'rotate 15s infinite linear, move 12s infinite alternate ease-in-out, wobble 13s infinite alternate ease-in-out'
          }}
        />
      </div>
    </div>
  );
};
