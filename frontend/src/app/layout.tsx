import type { Metadata } from "next";
import "./globals.css";
import { ThemeProvider } from '@/components/ThemeProvider';

export const metadata: Metadata = {
  title: "Scandium Labs | Physics Meets Prediction",
  description: "AI-accelerated materials discovery powered by physics-constrained graph neural networks",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className="antialiased bg-background text-primary-text font-sans selection:bg-accent selection:text-background overflow-x-hidden" style={{ background: 'var(--background)', color: 'var(--primary-text)' }}>
        {children}
      </body>
    </html>
  );
}
