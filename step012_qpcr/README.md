# Step 12: Förster transfer, computed from the geometry

Two hybridisation probes land on a DNA target a couple of nucleotides apart.
The donor hands its energy to the acceptor without ever emitting a photon, and
how much it hands over is recomputed **on every frame** from the separation the
scene's own geometry produces — not keyframed.

    make structure   rebuild Resources/scene.json (needs step008a_plasmid)
    make run         render renders/fret.gif
    make test        run the 25 tests

## A correction to the proposal, kept because it matters

This step was proposed around a **TaqMan hydrolysis probe** — one probe with a
reporter and a quencher, cut apart by the polymerase. Working the numbers first
ruled that out:

When the polymerase reaches a TaqMan probe, the probe is *hybridised*, and
hybridised DNA is stiff (persistence length ~50 nm, far longer than a 25-mer).
So reporter and quencher sit at opposite ends of a rigid 8.5 nm rod. Förster
transfer at 8.5 nm with R₀ near 5.5 nm is about **4%** — by FRET alone the
intact probe would already be ~96% *bright*. Real TaqMan probes are dark until
cut, so FRET is **not** what darkens them. The dominant mechanism is contact
quenching, a binding equilibrium with no distance law to compute.

Driving a TaqMan render off the r⁻⁶ law would have taught something false. The
LightCycler HybProbe format has no such problem: FRET between two adjacent
probes *is* the designed mechanism, and the separation is set by base-pair
geometry, which is measured.
