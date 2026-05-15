"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';

const PRESETS = {
  Silicon: `data_Si
_symmetry_Int_Tables_number 1
_symmetry_space_group_name_H-M  'P 1'
_cell_length_a   5.43
_cell_length_b   5.43
_cell_length_c   5.43
_cell_angle_alpha   90
_cell_angle_beta    90
_cell_angle_gamma   90
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
Si Si 0.000 0.000 0.000
Si Si 0.250 0.250 0.250`,
  LiFePO4: `data_LiFePO4
_symmetry_Int_Tables_number 1
_symmetry_space_group_name_H-M  'P 1'
_cell_length_a 10.33
_cell_length_b 6.01
_cell_length_c 4.69
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
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
_symmetry_Int_Tables_number 1
_symmetry_space_group_name_H-M  'P 1'
_cell_length_a 5.64
_cell_length_b 5.64
_cell_length_c 5.64
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
Na Na 0.0 0.0 0.0
Cl Cl 0.5 0.5 0.5`
};

export default function LiveDemo() {
  const [activeMaterial, setActiveMaterial] = useState<keyof typeof PRESETS>('Silicon');
  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handlePredict = async () => {
    setLoading(true);
    setError('');
    setResults(null);

    const payload = {
      structure: {
        structure_data: PRESETS[activeMaterial],
        format: 'cif',
        material_id: activeMaterial,
      },
      properties: ['band_gap', 'formation_energy', 'bulk_modulus'],
    };

    try {
      const response = await fetch('http://127.0.0.1:8000/predict/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': 'test_key_dev_only', // Development test key
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
    <section className="py-24 bg-scandium-dark text-white relative border-t border-white/5">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-5xl font-bold tracking-tight mb-4">
            Try <span className="text-scandium-primary">PIGNet</span> Live
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Experience our physics-informed graph neural network in real-time. Predict material properties directly from crystal structure.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Input Panel */}
          <div className="bg-white/5 border border-white/10 rounded-2xl p-6 backdrop-blur-sm">
            <h3 className="text-xl font-semibold mb-6 flex items-center">
              <span className="bg-scandium-primary/20 text-scandium-primary px-3 py-1 rounded-full text-sm mr-3">1</span>
              Select Material Structure
            </h3>
            
            <div className="flex flex-wrap gap-3 mb-6">
              {(Object.keys(PRESETS) as Array<keyof typeof PRESETS>).map((material) => (
                <button
                  key={material}
                  onClick={() => setActiveMaterial(material)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                    activeMaterial === material
                      ? 'bg-scandium-primary text-black'
                      : 'bg-white/5 text-gray-300 hover:bg-white/10 hover:text-white'
                  }`}
                >
                  {material}
                </button>
              ))}
            </div>

            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-b from-transparent to-black/50 pointer-events-none rounded-xl" />
              <textarea
                readOnly
                value={PRESETS[activeMaterial]}
                className="w-full h-64 bg-black/50 border border-white/10 rounded-xl p-4 text-xs font-mono text-gray-400 focus:outline-none focus:border-scandium-primary/50 resize-none"
              />
            </div>

            <button
              onClick={handlePredict}
              disabled={loading}
              className="w-full mt-6 bg-gradient-to-r from-scandium-primary to-scandium-secondary hover:opacity-90 text-black font-semibold py-4 rounded-xl transition-all shadow-[0_0_20px_rgba(45,212,191,0.2)] hover:shadow-[0_0_30px_rgba(45,212,191,0.4)] disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center"
            >
              {loading ? (
                <div className="animate-spin rounded-full h-5 w-5 border-2 border-black/20 border-t-black" />
              ) : (
                'Run Inference'
              )}
            </button>
          </div>

          {/* Results Panel */}
          <div className="bg-white/5 border border-white/10 rounded-2xl p-6 backdrop-blur-sm flex flex-col">
            <h3 className="text-xl font-semibold mb-6 flex items-center">
              <span className="bg-scandium-primary/20 text-scandium-primary px-3 py-1 rounded-full text-sm mr-3">2</span>
              Prediction Results
            </h3>

            {error ? (
              <div className="bg-red-500/10 border border-red-500/20 text-red-400 p-4 rounded-xl">
                {error}
                <div className="mt-2 text-sm opacity-80">Make sure the FastAPI backend is running on localhost:8000.</div>
              </div>
            ) : !results ? (
              <div className="flex-1 flex flex-col items-center justify-center text-gray-500 min-h-[300px]">
                <div className="w-16 h-16 rounded-full border border-white/10 flex items-center justify-center mb-4">
                  <svg className="w-6 h-6 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <p>Select a material and run inference to see predictions</p>
              </div>
            ) : (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex-1 flex flex-col"
              >
                <div className="flex justify-between items-center mb-6 pb-6 border-b border-white/10">
                  <div>
                    <div className="text-sm text-gray-400 mb-1">Model Version</div>
                    <div className="font-mono text-scandium-primary">{results.model_version}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-gray-400 mb-1">Inference Time</div>
                    <div className="font-mono text-scandium-primary">{results.inference_time_ms.toFixed(1)} ms</div>
                  </div>
                </div>

                <div className="space-y-4 mb-8">
                  {Object.entries(results.predictions).map(([prop, data]: [string, any]) => (
                    <div key={prop} className="bg-black/30 border border-white/5 rounded-xl p-4 flex justify-between items-center">
                      <div className="flex items-center gap-3">
                        <div className={`w-2 h-2 rounded-full ${data.physics_valid ? 'bg-green-500 shadow-[0_0_10px_rgba(34,197,94,0.5)]' : 'bg-red-500'}`} />
                        <div className="capitalize text-gray-200">{prop.replace(/_/g, ' ')}</div>
                      </div>
                      <div className="text-right">
                        <span className="text-xl font-bold text-white mr-2">{data.value.toFixed(4)}</span>
                        <span className="text-sm text-gray-500">{data.unit}</span>
                      </div>
                    </div>
                  ))}
                </div>

                {results.warnings?.length > 0 && (
                  <div className="mt-auto bg-yellow-500/10 border border-yellow-500/20 rounded-xl p-4">
                    <div className="flex items-center gap-2 text-yellow-500 font-medium mb-2">
                      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                      </svg>
                      Physics Warnings
                    </div>
                    <ul className="list-disc list-inside text-sm text-yellow-500/80 space-y-1">
                      {results.warnings.map((w: string, i: number) => <li key={i}>{w}</li>)}
                    </ul>
                  </div>
                )}
                
                {results.violations?.length > 0 && (
                  <div className="mt-auto bg-red-500/10 border border-red-500/20 rounded-xl p-4">
                    <div className="flex items-center gap-2 text-red-500 font-medium mb-2">
                      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                      </svg>
                      Physics Violations
                    </div>
                    <ul className="list-disc list-inside text-sm text-red-400 space-y-1">
                      {results.violations.map((v: string, i: number) => <li key={i}>{v}</li>)}
                    </ul>
                  </div>
                )}

              </motion.div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
