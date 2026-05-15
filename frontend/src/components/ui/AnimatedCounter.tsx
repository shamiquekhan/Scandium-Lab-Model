'use client';

import { useEffect, useRef, useState } from 'react';
import { useInView } from 'framer-motion';

interface Props {
  target: number;
  duration?: number;
  suffix?: string;
  className?: string;
}

export default function AnimatedCounter({ target, duration = 1200, suffix = '', className = '' }: Props) {
  const ref = useRef<HTMLSpanElement>(null);
  const isInView = useInView(ref, { once: true, amount: 0.5 });
  const [value, setValue] = useState(0);

  useEffect(() => {
    if (!isInView) return;
    let start = 0;
    const step = target / (duration / 16);
    const timer = setInterval(() => {
      start += step;
      if (start >= target) {
        setValue(target);
        clearInterval(timer);
      } else {
        setValue(Number.isInteger(target) ? Math.floor(start) : Number(start.toFixed(2)));
      }
    }, 16);
    return () => clearInterval(timer);
  }, [isInView, target, duration]);

  const formatted = Number.isInteger(target)
    ? value >= 1000 ? value.toLocaleString() : value.toString()
    : value.toFixed(2);

  return (
    <span ref={ref} className={className}>
      {formatted}{suffix}
    </span>
  );
}
