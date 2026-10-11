'use strict';
/* The presenter's keynotes (round 20): what to say in a few words per beat, written by hand for a 15-minute talk.
   k: one to three keynotes. show: what is on the screen and needs no words (point at it, do not read it out).
   The full sentences stay in js/notes.js (key `n` in the World, and the video's narration). One entry per beat: the engine checks. */
const KEYS = [
  /* S01 */ [
    { k: ['Renormalon-subtracted NRQCD for heavy hadrons'], show: 'title, coauthors, arXiv number' },
    { k: ['An infrared obstruction, decades old', 'We remove it: a family of states becomes predictable'] },
  ],
  /* S02 */ [
    { k: ['All-heavy: the cleanest bound states of QCD', 'Four families to follow'], show: 'the four particles and their names' },
    { k: ['Mass far above momentum far above binding', 'Controlled expansions'], show: 'the three scales' },
    { k: ['Precision laboratory: EFT, lattice, experiment at 10 MeV'] },
    { k: ['Timely: lattice baryons at 10 to 20 MeV', 'LHC excess at the top threshold'], show: 'CMS and ATLAS chips' },
    { k: ['The obstacle: fixed-order scale dependence', 'As large as the signal'], show: 'the two numbers' },
  ],
  /* S03 */ [
    { k: ['QCD at the heavy-quark mass: integrate out hard modes'] },
    { k: ['NRQCD: integrate out soft modes, potentials remain'] },
    { k: ['pNRQCD: ordinary quantum mechanics'] },
    { k: ['Kinetic term plus pair potentials', 'Matched to QCD, not modelled'], show: 'the Hamiltonian' },
    { k: ['Singlet pair: colour factor four thirds'] },
    { k: ['Baryon: each pair gets half of it', 'NNLO, three-body forces at fixed order'] },
    { k: ['Static limit: no 1/m_Q, no spin', 'Spin: 5 to 10 percent of the binding'] },
  ],
  /* S04 */ [
    { k: ['Potential = series in alpha_s + a constant', 'Factorial growth: asymptotic, the renormalon'], show: 'the two formulas' },
    { k: ['Terms shrink, reach a minimum, blow up', 'Best case: an error of order Lambda_QCD'] },
    { k: ['Symptom: a factor 2 in scale moves the mass', 'LO band does not overlap the others'], show: '700 MeV' },
    { k: ['Origin: infrared', 'The smallest term, carried onto the potential'] },
    { k: ['A constant: the curve slides, same at every r'] },
    { k: ['The force does not change', 'One number is ambiguous, not the shape'] },
    { k: ['Borel plane: a branch point at one half'] },
  ],
  /* S05 */ [
    { k: ['Idea: isolate the leading renormalon, resum the growth'] },
    { k: ['Three pieces. First: the series we know'] },
    { k: ['Subtract the renormalon part of each term'] },
    { k: ['Second: the renormalon series, an infinite sum'] },
    { k: ['Borel-summed: a closed form'] },
    { k: ['Third: Lambda_R, one constant', 'Past the branch point: what the Borel sum cannot reach'] },
    { k: ['Lambda_R takes it in: free of the leading ambiguity', 'Absorbed in the quark mass, tuned on quarkonium', 'Baryons, tetraquarks: same tuned masses'] },
    { k: ['Normalization has converged: a stable subtraction'], show: '0.771, 0.772' },
    { k: ['Minimal: same inputs, series only reorganized', 'First use in a bound state'], show: 'the references' },
  ],
  /* S06 */ [
    { k: ['The catch: a wavefunction samples every r', 'Scale 1/r leaves the perturbative regime'] },
    { k: ['1/r falls below Lambda_QCD'] },
    { k: ['Smooth scale: saturates at 1 GeV', 'No kinks: stable Monte Carlo'], show: 'the formula' },
    { k: ['1 GeV: active where the states live', '2 GeV: back to fixed order'] },
    { k: ['Vary lambda by a factor of two', 'Flavours set by the lightest quark'] },
  ],
  /* S07 */ [
    { k: ['Variational Monte Carlo: an upper bound, a starting point'] },
    { k: ['Green’s function Monte Carlo: projects onto the ground state'] },
    { k: ['One trial wavefunction per channel'], show: 'the three forms' },
    { k: ['Statistical errors small', 'Time extent and step tracked as systematics'], show: 'the numbers' },
  ],
  /* S08 */ [
    { k: ['Fixed order: broad bands, LO apart'] },
    { k: ['MRS: narrower bands, orders overlap'] },
    { k: ['Tune charm and bottom on the spin-averaged 1S', 'Mass drift: percent level becomes under 1 percent'] },
    { k: ['Lambda_R ends up inside the tuned masses'] },
    { k: ['From here: predictions from two masses and the top mass'], show: 'the two values' },
  ],
  /* S09 */ [
    { k: ['Reading: statistics, then scale variation', 'Top row: states with a top quark'] },
    { k: ['Toponium: close to threshold calculations', 'Wide band, small shift between orders', 'Zero-width limit'] },
    { k: ['The two tuned masses meet: B_c'] },
    { k: ['Parameter-free', 'About 25 MeV below the spin average'], show: 'the three values' },
    { k: ['With a top quark: benchmarks, not observables'] },
  ],
  /* S10 */ [
    { k: ['Triply heavy baryons: ours, NNLO with MRS'], show: 'the four masses' },
    { k: ['Lattice: we sit 125 to 175 MeV below'], show: 'the lattice values' },
    { k: ['Fixed order sits closer', 'The MRS advantage is not the central value'] },
    { k: ['Scale band: 700 MeV becomes 10', 'That makes the comparison meaningful'] },
    { k: ['The gap is a pattern: it falls like 1/m_Q', 'What the static limit leaves out'] },
    { k: ['Top benchmarks, up to three top quarks'], show: 'the six masses' },
  ],
  /* S11 */ [
    { k: ['Ratio baryon over quarkonium: the renormalon partly cancels', 'Dashed: the lower bound, three halves'] },
    { k: ['MRS: all orders in one narrow band'], show: '1.50 to 1.51' },
    { k: ['Lattice a little higher, drifting down to us', 'The same gap again'] },
    { k: ['Fixed order agrees only through huge bands'] },
  ],
  /* S12 */ [
    { k: ['Tetraquarks: which ones bind?', 'Ratio above zero means bound'] },
    { k: ['It depends on the quark masses'] },
    { k: ['Binding needs a mass hierarchy', 'MRS tightens the window by a third', 'NLO only'], show: '0.085 against 0.125' },
    { k: ['Three bind: two top antiquarks at the centre', 'Two fall apart into mesons'], show: 'names and masses' },
    { k: ['Bound at every λ: every band clears zero'], show: 'the binding energies' },
  ],
  /* S13: Conclusions (round 24e: the old S13 and S14 in one panel) */ [
    { k: ['To conclude: first MRS inside a bound-state calculation'] },
    { k: ['What did it buy? The band: 700 MeV to 10'] },
    { k: ['Three more diagnostics, same direction'], show: 'the rows' },
    { k: ['Hallmarks of convergence, still asymptotic', 'Lambda bands test the scheme, not the truncation'] },
    { k: ['Payoff: the gap to the lattice is interpretable', 'Missing 1/m_Q physics, not noise'] },
    { k: ['A family of predictions from two tuned masses'], show: 'the particles: do not list them' },
    { k: ['Next: spin and 1/m_Q, other channels, top threshold', 'Thank you'] },
    { k: ['The whole talk at a glance'] },
  ],
  /* S14 */ [
    { k: ['Last stop: the references'] },
    { k: ['The paper is on the arXiv'], show: 'the list' },
  ],
];
/* when each panel should start, in seconds, for a 15-minute talk (the pace of the narrated video) */
const KEY_T = [0, 31, 84, 147, 227, 344, 399, 456, 514, 576, 643, 696, 769, 868];   /* round 24e: the old S14 (818) is part of S13 */
/* which particles each beat is about (round 20b): they move for as long as the talk is about them, the others stay still.
   One list per beat. Ids are the shelf's (SHELF.glyphs), plus 'pair' (the static pair), 'hero' (the title baryon), '*' (all).
   A particle also moves in any beat that writes its name or its number, whatever this table says. */
const LIVE = (() => {
  const _ = [], FAM = ['cc', 'bb', 'Occc', 'T4'], OM = ['Occc', 'Occb', 'Ocbb', 'Obbb'], QQ = ['cc', 'bb'], P = ['pair'];
  return [
    /* S01 */ [['hero'], ['hero', ...FAM]],
    /* S02 */ [FAM, FAM, FAM, ['Occc'], _],
    /* S03 */ [P, P, P, P, P, P, P],
    /* S04 */ [P, _, _, _, _, _, _],
    /* S05 */ [_, _, _, _, _, _, _, _, _],
    /* S06 */ [P, _, _, _, _],
    /* S07 */ [P, _, _, _],
    /* S08 */ [_, _, QQ, QQ, QQ],
    /* S09 */ [_, ['et'], [...QQ, 'bc'], ['bc'], ['tc', 'tb']],
    /* S10 */ [OM, OM, ['Obbb'], ['Obbb'], OM, ['Otcc', 'Otcb', 'Otbb', 'Ottc', 'Ottb', 'Ottt']],
    /* S11 */ [_, _, _, _],
    /* S12 */ [['T4'], ['T4'], _, ['bbtt', 'bctt', 'cctt', 'bbbb', 'ccbb'], ['bbtt', 'bctt', 'cctt']],
    /* S13 */ [_, _, _, _, _, ['*'], _, _],
    /* S14 */ [_, _],
  ];
})();
