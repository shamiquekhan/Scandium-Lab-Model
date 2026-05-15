import Navbar from '@/components/Navbar';
import Hero from '@/components/Hero';
import AboutStats from '@/components/AboutStats';
import Capabilities from '@/components/Capabilities';
import ResearchProjects from '@/components/ResearchProjects';
import Publications from '@/components/Publications';
import Team from '@/components/Team';
import Pricing from '@/components/Pricing';
import Testimonials from '@/components/Testimonials';
import Contact from '@/components/Contact';
import Footer from '@/components/Footer';

export default function Home() {
  return (
    <>
      <Navbar />
      <main>
        <Hero />
        <AboutStats />
        <Capabilities />
        <ResearchProjects />
        <Publications />
        <Team />
        <Pricing />
        <Testimonials />
        <Contact />
      </main>
      <Footer />
    </>
  );
}
