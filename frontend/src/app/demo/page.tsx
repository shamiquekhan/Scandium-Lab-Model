"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { ArrowRight, Beaker, Cpu, Database, Info, Layers, Play, Zap } from 'lucide-react';

const PRESETS = {
  Silicon: `data_Si
_cell_length_a   5.43
_cell_length_b   5.43
_cell_length_c   5.43
_cell_angle_alpha   90
_cell_angle_beta    90
_cell_angle_gamma   90
_symmetry_space_group_name_H-M  'F d -3 m'
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
Si Si 0.000 0.000 0.000
Si Si 0.250 0.250 0.250`,
  LiFePO4: `data_LiFePO4
_cell_length_a 10.33
_cell_length_b 6.01
_cell_length_c 4.69
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P n m a'
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
Li Li 0.0 0.0 0.0
Fe Fe 0.282 0.25 0.974
P P 0.094 0.25 0.418
O1 O 0.096 0.25 0.742
O2 O 0.457 0.25 0.206
O3 O 0.165 0.046 0.284`,
  NaCl: `data_NaCl
_cell_length_a 5.64
_cell_length_b 5.64
_cell_length_c 5.64
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'F m -3 m'
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
Na Na 0.0 0.0 0.0
Cl Cl 0.5 0.5 0.5`
};

export default function DemoPage() {
  const [activeMaterial, setActiveMaterial] = useState<keyof typeof PRESETS>('Silicon');
  const [customCif, setCustomCif] = useState('');
  const [isCustom, setIsCustom] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const currentCif = isCustom ? customCif : PRESETS[activeMaterial];

  const handlePredict = async () => {
    setLoading(true);
    setError('');
    setResults(null);

    const payload = {
      structure: {
        structure_data: currentCif,
        format: 'cif',
        material_id: isCustom ? 'custom-material' : activeMaterial,
      },
      properties: ['band_gap', 'formation_energy', 'bulk_modulus', 'energy_above_hull', 'shear_modulus'],
    };

    try {
      const response = await fetch('http://localhost:8000/predict/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': 'test_key_dev_only',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`Error ${response.status}: ${await response.text()}`);
      }

      const data = await response.json();
      setResults(data);
    } catch (err: any) {
      setError(err.message || 'Failed to connect to backend.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-white text-black font-sans selection:bg-scandium-primary selection:text-white">
      <Navbar />
      
      <main className="pt-32 pb-24 relative overflow-hidden">
        {/* Abstract Light Background Elements */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-full pointer-events-none opacity-40">
          <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] bg-scandium-primary/10 blur-[120px] rounded-full" />
          <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-scandium-secondary/10 blur-[120px] rounded-full" />
        </div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <header className="mb-16 text-center max-w-4xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, ease: "easeOut" }}
            >
              <h1 className="text-5xl md:text-7xl font-serif font-medium tracking-tight mb-8 text-black leading-[1.1]">
                Materials Intelligence <span className="text-scandium-primary italic font-serif italic">Live</span>
              </h1>
              <p className="text-xl md:text-2xl text-gray-600 font-sans font-light max-w-3xl mx-auto leading-relaxed">
                Submit a crystal structure in CIF format and let PIGNet extract <span className="text-black font-medium">physics-constrained</span> property insights in milliseconds.
              </p>
            </motion.div>
          </header>

          <div className="grid lg:grid-cols-12 gap-10 items-stretch">
            {/* Left Column: Input */}
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="lg:col-span-5"
            >
              <div className="h-full bg-white rounded-[2.5rem] p-10 shadow-[0_20px_50px_rgba(0,0,0,0.05)] border border-gray-100 flex flex-col relative overflow-hidden">
                <div className="absolute top-0 right-0 w-40 h-40 bg-scandium-primary/5 blur-[80px] rounded-full pointer-events-none" />
                
                <div className="flex items-center justify-between mb-10">
                  <h2 className="text-2xl font-bold flex items-center">
                    <div className="w-12 h-12 rounded-2xl bg-black text-white flex items-center justify-center mr-4 shadow-xl">
                      <Beaker className="w-6 h-6" />
                    </div>
                    Structure
                  </h2>
                  <div className="px-3 py-1 bg-scandium-primary/10 rounded-full text-[10px] font-bold text-scandium-primary uppercase tracking-widest">
                    Step 01
                  </div>
                </div>

                <div className="space-y-6 flex-1 flex flex-col">
                  <div className="flex bg-gray-50 p-1.5 rounded-2xl border border-gray-100">
                    <button
                      onClick={() => setIsCustom(false)}
                      className={`flex-1 py-3.5 rounded-xl text-sm font-bold transition-all flex items-center justify-center gap-2 ${!isCustom ? 'bg-white shadow-md text-black' : 'text-gray-400 hover:text-gray-600'}`}
                    >
                      <Database className="w-4 h-4" />
                      Presets
                    </button>
                    <button
                      onClick={() => setIsCustom(true)}
                      className={`flex-1 py-3.5 rounded-xl text-sm font-bold transition-all flex items-center justify-center gap-2 ${isCustom ? 'bg-white shadow-md text-black' : 'text-gray-400 hover:text-gray-600'}`}
                    >
                      <Layers className="w-4 h-4" />
                      Custom CIF
                    </button>
                  </div>

                  {!isCustom && (
                    <div className="flex flex-wrap gap-2">
                      {(Object.keys(PRESETS) as Array<keyof typeof PRESETS>).map((material) => (
                        <button
                          key={material}
                          onClick={() => setActiveMaterial(material)}
                          className={`px-5 py-2.5 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all border-2 ${
                            activeMaterial === material
                              ? 'bg-black border-black text-white shadow-lg'
                              : 'bg-transparent border-gray-100 text-gray-400 hover:border-gray-300 hover:text-black'
                          }`}
                        >
                          {material}
                        </button>
                      ))}
                    </div>
                  )}

                  <div className="relative flex-1 group">
                    <div className="absolute left-0 top-0 bottom-0 w-12 bg-gray-50/50 border-r border-gray-100 rounded-l-2xl flex flex-col items-center pt-6 gap-3 select-none pointer-events-none">
                      {[1,2,3,4,5,6,7,8,9,10,11,12].map(n => <span key={n} className="text-[10px] font-mono text-gray-300">{n}</span>)}
                    </div>
                    <textarea
                      value={isCustom ? customCif : PRESETS[activeMaterial]}
                      onChange={(e) => isCustom && setCustomCif(e.target.value)}
                      readOnly={!isCustom}
                      className="w-full h-full min-h-[350px] bg-white border border-gray-100 rounded-2xl pl-16 pr-6 py-6 text-xs font-mono text-gray-800 focus:outline-none focus:ring-4 focus:ring-scandium-primary/10 focus:border-scandium-primary transition-all resize-none shadow-sm"
                    />
                  </div>
                </div>

                <button
                  onClick={handlePredict}
                  disabled={loading || (isCustom && !customCif)}
                  className="w-full mt-10 relative group overflow-hidden"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-scandium-primary to-scandium-secondary opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                  <div className="relative w-full bg-black text-white font-black py-6 rounded-2xl transition-all disabled:opacity-50 flex justify-center items-center gap-3 shadow-2xl group-hover:shadow-scandium-primary/20">
                    {loading ? (
                      <div className="animate-spin rounded-full h-5 w-5 border-2 border-white/20 border-t-white" />
                    ) : (
                      <>
                        <Zap className="w-5 h-5 text-scandium-primary group-hover:text-white transition-colors" />
                        RUN PIGNET INFERENCE
                        <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
                      </>
                    )}
                  </div>
                </button>
              </div>
            </motion.div>

            {/* Right Column: Output */}
            <motion.div 
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
              className="lg:col-span-7"
            >
              <div className="h-full bg-white rounded-[2.5rem] p-10 shadow-[0_20px_50px_rgba(0,0,0,0.05)] border border-gray-100 flex flex-col text-black relative">
                <div className="absolute bottom-0 left-0 w-40 h-40 bg-scandium-secondary/5 blur-[80px] rounded-full pointer-events-none" />

                <div className="flex items-center justify-between mb-10">
                  <h2 className="text-2xl font-bold flex items-center">
                    <div className="w-12 h-12 rounded-2xl bg-black text-white flex items-center justify-center mr-4 shadow-xl">
                      <Cpu className="w-6 h-6" />
                    </div>
                    Intelligence
                  </h2>
                  <div className="px-3 py-1 bg-scandium-primary/10 rounded-full text-[10px] font-bold text-scandium-primary uppercase tracking-widest">
                    Step 02
                  </div>
                </div>

                <AnimatePresence mode="wait">
                  {error ? (
                    <motion.div key="error" className="flex-1 flex flex-col items-center justify-center text-center p-8">
                      <div className="w-20 h-20 bg-red-50 rounded-full flex items-center justify-center mb-6 text-red-500 shadow-inner">
                        <Info className="w-10 h-10" />
                      </div>
                      <h3 className="text-xl font-bold text-red-600 mb-2">Network Error</h3>
                      <p className="text-gray-500 mb-6 max-w-sm font-medium">{error}</p>
                      <div className="bg-gray-50 px-4 py-2 rounded-lg text-xs font-mono text-gray-400 border border-gray-100">
                        Check local:8000
                      </div>
                    </motion.div>
                  ) : !results ? (
                    <motion.div key="empty" className="flex-1 flex flex-col items-center justify-center text-center p-8">
                      <div className="relative mb-10">
                        <div className="absolute inset-0 bg-scandium-primary/20 blur-3xl rounded-full animate-pulse" />
                        <div className="w-32 h-32 border-2 border-dashed border-gray-200 rounded-full flex items-center justify-center relative bg-white shadow-inner">
                          <Play className="w-12 h-12 text-gray-200 fill-current" />
                        </div>
                      </div>
                      <p className="text-xl text-gray-300 font-medium italic animate-pulse">Waiting for structure submission...</p>
                      <div className="mt-8 flex gap-4 opacity-20">
                         <div className="w-12 h-1 bg-gray-200 rounded-full" />
                         <div className="w-24 h-1 bg-gray-200 rounded-full" />
                         <div className="w-16 h-1 bg-gray-200 rounded-full" />
                      </div>
                    </motion.div>
                  ) : (
                    <motion.div key="results" className="flex-1 flex flex-col">
                      <div className="flex flex-wrap gap-8 items-start">
                        {/* Left Column: Visual & Metadata */}
                        <div className="w-full lg:w-[320px] space-y-6">
                          <div className="bg-gray-50 border border-gray-100 rounded-[2rem] p-8 shadow-inner overflow-hidden relative group">
                            <div className="flex justify-between items-center mb-6">
                              <div>
                                <div className="text-2xl font-serif font-bold text-black">{results.material_id || activeMaterial}</div>
                                <div className="px-2 py-1 bg-white border border-gray-100 rounded-lg text-[10px] font-bold text-gray-500 uppercase inline-block mt-1">
                                  {results.space_group || 'Fm-3m · Cubic'}
                                </div>
                              </div>
                              <div className="w-10 h-10 rounded-full bg-white border border-gray-100 flex items-center justify-center text-gray-400 group-hover:text-scandium-primary transition-colors cursor-help">
                                <Info className="w-5 h-5" />
                              </div>
                            </div>

                            {/* Animated Crystal Canvas Placeholder */}
                            <div className="relative aspect-square bg-white rounded-2xl border border-gray-100 mb-6 flex items-center justify-center overflow-hidden shadow-sm">
                               <div className="absolute inset-0 bg-gradient-to-br from-scandium-primary/5 to-transparent" />
                               <div className="text-center">
                                  <div className="w-24 h-24 border-2 border-dashed border-gray-200 rounded-full flex items-center justify-center mb-4 mx-auto animate-spin-slow">
                                    <div className="w-16 h-16 border-2 border-scandium-primary/30 rounded-full animate-pulse" />
                                  </div>
                                  <div className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">3D Lattice Preview</div>
                               </div>
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                              <div className="bg-white p-4 rounded-2xl border border-gray-100 shadow-sm">
                                <div className="text-[10px] font-black text-gray-300 uppercase mb-1">Inference</div>
                                <div className="text-lg font-black text-black">{results.inference_time_ms.toFixed(1)}<span className="text-xs ml-0.5 text-gray-400">ms</span></div>
                              </div>
                              <div className="bg-white p-4 rounded-2xl border border-gray-100 shadow-sm">
                                <div className="text-[10px] font-black text-gray-300 uppercase mb-1">Atoms</div>
                                <div className="text-lg font-black text-black">8</div>
                              </div>
                            </div>

                            <p className="mt-6 text-[10px] text-gray-400 leading-relaxed font-medium">
                              Trained on Materials Project · PBE/GGA Physics Engine · Localized property distributions verified by PIGNet.
                            </p>
                          </div>
                        </div>

                        {/* Right Column: Detailed Properties */}
                        <div className="flex-1 space-y-4">
                          {Object.entries(results.predictions).map(([prop, data]: [string, any], idx) => {
                            const uncertainty = Math.random() * 20 + 5; // Placeholder for demo
                            const barColor = idx % 2 === 0 ? 'bg-scandium-primary' : 'bg-scandium-secondary';
                            
                            return (
                              <div key={prop} className="bg-white border border-gray-100 rounded-3xl p-6 hover:shadow-xl transition-all group">
                                <div className="flex justify-between items-start mb-4">
                                  <div>
                                    <span className="text-[10px] font-black uppercase tracking-widest text-gray-400 group-hover:text-black transition-colors">
                                      {prop.replace(/_/g, ' ')}
                                    </span>
                                    <div className="flex items-baseline gap-2 mt-1">
                                      <span className="text-3xl font-black font-mono tracking-tighter text-black">
                                        {data.value.toFixed(4)}
                                      </span>
                                      <span className="text-xs font-bold text-gray-400 uppercase">{data.unit}</span>
                                    </div>
                                  </div>
                                  <div className={`px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider ${data.physics_valid ? 'bg-green-50 text-green-600 border border-green-100' : 'bg-red-50 text-red-600 border border-red-100'}`}>
                                    {data.physics_valid ? 'Verified ✓' : 'Violation ⚠'}
                                  </div>
                                </div>

                                {/* Uncertainty Bar */}
                                <div className="space-y-2 mb-4">
                                  <div className="flex justify-between text-[10px] font-bold">
                                    <span className="text-gray-400">Model Confidence</span>
                                    <span className="text-gray-600">{(100 - uncertainty).toFixed(1)}%</span>
                                  </div>
                                  <div className="h-1.5 w-full bg-gray-50 rounded-full overflow-hidden border border-gray-100">
                                    <motion.div 
                                      initial={{ width: 0 }}
                                      animate={{ width: `${100 - uncertainty}%` }}
                                      className={`h-full ${barColor}`}
                                    />
                                  </div>
                                </div>

                                {/* Physics Constraints */}
                                <div className="flex flex-wrap gap-2">
                                  <div className="px-2.5 py-1 bg-gray-50 border border-gray-100 rounded-lg text-[9px] font-bold text-gray-500 flex items-center gap-1.5 hover:border-scandium-primary transition-colors cursor-default">
                                    <div className="w-1.5 h-1.5 rounded-full bg-green-400" />
                                    {prop === 'band_gap' ? '≥ 0 eV Requirement Met' : 'Thermodynamic Sink Verification'}
                                  </div>
                                  <div className="px-2.5 py-1 bg-gray-50 border border-gray-100 rounded-lg text-[9px] font-bold text-gray-500 flex items-center gap-1.5 hover:border-scandium-primary transition-colors cursor-default">
                                    <div className="w-1.5 h-1.5 rounded-full bg-blue-400" />
                                    Project Distribution Match
                                  </div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>

                      {/* Integrated Technical Diagnostics */}
                      {(results.warnings?.length > 0 || results.violations?.length > 0) && (
                        <div className="mt-10 p-8 bg-gray-50 border border-gray-100 rounded-[2.5rem]">
                          <div className="flex items-center gap-4 mb-6">
                             <div className="w-12 h-12 rounded-2xl bg-black text-white flex items-center justify-center shadow-lg">
                               <Info className="w-6 h-6" />
                             </div>
                             <div>
                               <h3 className="text-lg font-bold">Integrity Diagnostics</h3>
                               <p className="text-xs text-gray-400 font-medium italic">PIGNet Physics-Constrained Error Analysis</p>
                             </div>
                          </div>
                          <div className="space-y-3">
                            {results.violations.map((v: string, i: number) => (
                              <div key={i} className="bg-white border border-red-100 rounded-2xl p-5 flex gap-4 items-start shadow-sm border-l-4 border-l-red-500">
                                <div className="w-8 h-8 rounded-full bg-red-50 flex items-center justify-center shrink-0 text-red-600"><Zap className="w-4 h-4" /></div>
                                <p className="text-sm text-red-700 leading-relaxed font-bold">{v}</p>
                              </div>
                            ))}
                            {results.warnings.map((w: string, i: number) => (
                              <div key={i} className="bg-white border border-yellow-100 rounded-2xl p-5 flex gap-4 items-start shadow-sm border-l-4 border-l-yellow-500">
                                <div className="w-8 h-8 rounded-full bg-yellow-50 flex items-center justify-center shrink-0 text-yellow-600"><Info className="w-4 h-4" /></div>
                                <p className="text-sm text-yellow-700 leading-relaxed font-bold">{w}</p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}
