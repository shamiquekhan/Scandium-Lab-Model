# Scandium Labs Single-Page Website Implementation Plan

The objective is to build a full single-page website for Scandium Labs exactly matching the structure, aesthetic, animations, and component language of the AIXOR template. We will use Next.js, React, Tailwind CSS, and Framer Motion.

## Proposed Changes

We will build out the `src/app/` and `src/components/` structure.

### Project Setup
- Create `src/app/layout.tsx` for global HTML structure and importing `globals.css`.
- Create `src/app/page.tsx` as the main landing page orchestrating all sections.
- Create `src/components/ui/` for reusable elements (buttons, animated counters, cards).

### Components Structure
We will create individual React components for each section to maintain code clarity and reusability.

#### [NEW] `src/components/Navbar.tsx`
Sticky navigation with backdrop blur and mobile menu toggle.

#### [NEW] `src/components/Hero.tsx`
Full viewport section with looping video background, 96px headline, and a floating quote card on the right.

#### [NEW] `src/components/AboutStats.tsx`
Four numbered stat cards featuring animated counters (using Framer Motion) and image textures.

#### [NEW] `src/components/Capabilities.tsx`
Accordion list for services, updating the visible image on the right upon selection.

#### [NEW] `src/components/ResearchProjects.tsx`
Featured project cards and a looping "Featured Research Papers!" marquee header.

#### [NEW] `src/components/Publications.tsx`
Horizontal strips that lift and reveal a background texture on hover.

#### [NEW] `src/components/Team.tsx`
Grid of team portraits with hover scale effects and a "Join Our Research" CTA.

#### [NEW] `src/components/Pricing.tsx`
Monthly/Annual toggle switching between Research and Enterprise plans.

#### [NEW] `src/components/Testimonials.tsx`
Two rows of auto-scrolling marquee testimonials moving in opposite directions.

#### [NEW] `src/components/Contact.tsx`
Contact form and details grid.

#### [NEW] `src/components/Footer.tsx`
Four-column footer grid with a massive "SCANDIUM LABS" watermark text overlay at the bottom.

#### [NEW] `src/components/SectionHeading.tsx`
A reusable component for the standard section label (e.g., "ABOUT US" in 11px caps) and intro paragraph.

#### [NEW] `src/components/ui/AnimatedCounter.tsx`
A helper component to count numbers from 0 to a target value when scrolled into view using Framer Motion.

## Styling & Animations
- **Tailwind CSS**: Utilize the pre-configured design tokens in `tailwind.config.ts` (colors: background, primary-text, secondary-text, accent, accent-secondary).
- **Framer Motion**: Implement staggered fade-ups (translateY 24px -> 0, opacity 0 -> 1 over 600ms, threshold 0.8), hover state lifts (-2px Y translation), infinite marquees, and scale effects as specified in the prompt.
- **Fonts**: Use Playfair Display/Cormorant Garamond for headings and Inter/DM Sans for body and labels, matching the CSS setup.

## Verification Plan

### Manual Verification
1. Run `npm run dev` and open the local preview.
2. Verify visual fidelity:
   - Check if colors and typography match the AIXOR dark/lime aesthetic.
   - Ensure the hero video loops behind a 60% dark overlay.
3. Verify animations:
   - Check scroll-triggered fade-ups across all sections.
   - Confirm stat counters animate from zero when in view.
   - Validate accordion expansion and image swapping in Capabilities.
   - Test marquee loops (Research, Testimonials) and hover lifts on team/publication cards.
4. Verify responsiveness (desktop vs. mobile).
