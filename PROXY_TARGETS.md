# Desired proxies

Working notes, not for `release/` yet. What the proxy-consistency term actually needs, what
we measured, and the fields worth acquiring — ranked by expected value per unit of effort.

## The corrected criterion

The proxy head is `proxy_head(loc_encoder(coords))`, and the location encoder receives only
longitude, latitude and calendar features (hour, day-of-year, day-of-week, year). It has no
meteorological input. The best that branch can represent is therefore

    E[ z | place, hour, day-of-year, year ]

which is the proxy's **space-time climatology**. However much state information a proxy's
anomalies carry, the term as written cannot transfer it.

So the screening statistic is not the one we first used:

| statistic | surface NO₂ | what it means |
|---|---|---|
| anomaly correlation with UFP, per point | **+0.533** | the proxy is informative in principle |
| **spatial climatology, Pearson over 7 monitors** | **+0.379** | what the proxy term can actually transfer |
| **spatial climatology, Spearman over 7 monitors** | **+0.218** | |

That gap is the whole story. Surface NO₂ is a genuinely informative field whose *spatial
pattern* is only weakly related to UFP's, so raising λ pulls the location representation
toward a weakly-correlated map. Measured, holding everything else fixed:

| λ | MATES bias | MATES rank r |
|---|---|---|
| 0 (term off) | 1.23× | **+0.029** |
| 0.5 | 1.26× | −0.004 |
| 1 | 1.23× | −0.153 |
| 2 | 1.19× | −0.233 |
| 5 | 1.22× | −0.386 |

Monotonic, over a range about four times the seed-to-seed noise.

**Screen every candidate below before building it:** rank-correlate the candidate's site-mean
value against site-mean UFP across all 13 sites available (7 monitors + 6 MATES). It takes
minutes and it is the property that decides whether the proxy helps or hurts. A candidate that
does not beat NO₂'s +0.218 is not worth acquiring.

## The proxies we want, in priority order

### 1. Dispersion-modelled particle-number field
The best match to the criterion, because its spatial climatology *is* the near-road ultrafine
gradient by construction.

- **What:** EPA's R-LINE line-source dispersion model (now inside AERMOD), run hourly over the
  basin to produce a gridded particle-number concentration field at 10–100 m.
- **Inputs, all public or already held:** road geometry (held), hourly traffic volumes (PeMS),
  size-resolved number emission factors (EMFAC for California, or MOVES), hourly meteorology
  (HRRR, held).
- **Why it should pass:** it models the same quantity as the target, and the sharp near-road
  structure that no field we currently have can express.
- **Known limitation:** primary traffic only, no nucleation. That is acceptable — primary is
  the component with spatial structure; the secondary component is regional and the model
  already reaches it through meteorology and solar geometry.
- **Cost:** weeks. Far cheaper than coupling microphysics into a chemical transport model.

### 2. Particle-number emission-density field
The cheap pilot for (1): build the emissions, skip the dispersion physics.

- **What:** EMFAC link-level emissions with size-resolved number factors, plus port (ship and
  rail) and LAX inventories, convolved with a simple distance-decay kernel.
- **Why:** tests whether a source-proximity climatology beats NO₂'s +0.218 before anyone
  commits to running a dispersion model.
- **Cost:** days. Do this one first.

### 3. Mobile-monitoring ultrafine climatology
Scientifically the ideal proxy; the constraint is access, not money.

- **What:** median particle number by road segment from the Los Angeles mobile campaigns
  (Westerdahl et al. 2005; Fruin et al. 2008; CARB Mobile Platform III; Harbor Communities
  Monitoring Study).
- **Why it cannot fail the criterion:** it is the target variable, measured at thousands of
  locations, and a spatial climatology by construction.
- **Blocker:** none of the sources we checked states that the underlying data is publicly
  downloadable. This is a request to investigators or CARB, and it is the same ask that
  underpins the observational proposal.

### 4. Telematics / probe-vehicle activity
- **What:** hourly vehicle-kilometres per link from StreetLight, INRIX or HERE; SCAG holds
  licences for this region.
- **Why:** traffic density is the dominant spatial driver of primary ultrafine particles, and
  unlike PeMS this covers **arterials**, not only freeway mainlines — which matters because
  several monitors are not freeway-adjacent.
- **Cost:** licensing, or a collaboration with SCAG.

## Also worth having, lower priority

- **Surface CO, NOₓ and black carbon** from the same AQS network. More stations, and black
  carbon is a more direct primary-combustion tracer than NO₂. Screen them the same way; BC in
  particular may have a better spatial climatology match than NO₂ does.
- **A published ultrafine land-use-regression surface for Los Angeles**, if one exists. That is
  a spatial climatology of the target by definition, and would be a drop-in proxy.

## Ruled out, with the measurement

Recorded so nobody re-tests them. All are anomaly correlations against UFP after removing
per-site hour × month climatology (`scripts/screen_proxies.py`):

| field | r | verdict |
|---|---|---|
| TEMPO NO₂ column | −0.009 | no state information about the target |
| TEMPO HCHO column | −0.079 | no |
| TEMPO HCHO/NO₂ ratio | −0.005 | no |
| TEMPO NO₂ / mixing depth | +0.086 | no |
| GOES AOD | −0.060 | no |
| AQS PM₂.₅ | +0.132 | mass carries little number signal, as expected |

For reference, HRRR mixing depth — already an observation feature, not a proxy — scores
−0.251, so every satellite column tested is less informative about UFP than a predictor the
model already contains.

## Result: the architecture fix was implemented and does not work

Implemented as `model.proxy_state_vars` (see `ufp_pcl/models/fusion.py`,
`ufp_pcl/data/dataset.py::_StateMixin`); the proxy head becomes
`proxy_head([e_loc, state_encoder(met)])` with seven HRRR fields, and the observation branch
is still never touched, so the gradient-routing asymmetry is preserved. The state pathway does
what it was meant to do mechanically — proxy loss falls from ~0.49 to ~0.39 — but it does not
help the model, and with the station proxy it clearly hurts.

Mean MATES rank correlation over 3 seeds:

| configuration | MATES bias | rank r |
|---|---|---|
| λ=0, proxy term off | 1.23× | **+0.029** |
| GOES AOD, λ=1 (original) | 1.37× | −0.275 |
| GOES AOD + state head | 1.30× | −0.239 |
| surface NO₂ stations, λ=1 | 1.19× | −0.217 |
| **surface NO₂ stations + state head** | 1.21× | **−0.460** |

A control worth recording: the baseline config at λ=0 and the station-proxy config at λ=0 give
*bit-identical* results per seed, confirming λ=0 genuinely disables the proxy path and that the
arms differ in nothing else.

## Why: no proxy we have has the right spatial climatology

Applying the screening statistic properly — across all 13 sites, not just the 7 training
monitors — rules out everything currently available:

| candidate | Pearson | Spearman | n |
|---|---|---|---|
| AQS surface NO₂ | −0.228 | **0.000** | 13 |
| GOES AOD | −0.516 | **−0.363** | 13 |
| TEMPO NO₂ column | −0.338 | −0.297 | 13 |

**Not one is positive.** Since the proxy term can only transfer a spatial climatology, and no
available proxy's climatology is even weakly correlated with UFP's, the term has nothing
useful to transfer — which is exactly what the λ sweep shows.

### A methodological warning worth keeping

On the 7 training monitors alone, GOES AOD scores Pearson +0.723 / Spearman **+0.929** — it
looks like an excellent proxy. Adding the 6 independent MATES sites takes it to −0.516 /
**−0.363**. The sign flips completely. The seven monitors are all near-road or urban and share
a siting bias, so any proxy screened on them alone will appear to work. **Screen on the
independent sites or the answer is not merely noisy, it is inverted.** This is the same
representativeness problem that forces the 1.42× correction, now biting the proxy diagnosis.

## CORRECTION: the proxy term IS earning its place — on the map, not on the points

Everything above evaluates at 13 points. The proxy term exists to shape the *field*, so it
was audited there too (`scripts/proxy_map_audit.py`): dynamic range and physical structure of
the rendered surface, effective rank of the location embedding, and whether the leading modes
still identify with physical drivers.

| arm (3 seeds) | eff rank | \|PC1~coast\| | \|r\| solar | map sd |
|---|---|---|---|---|
| **GOES AOD, λ=1** | 6.86 ± 1.46 | **0.698 ± 0.072** | **0.724 ± 0.110** | 5973 |
| no proxy, λ=0 | **4.26 ± 0.76** | 0.020 ± 0.015 | 0.173 ± 0.014 | 4831 |
| surface NO₂ stations, λ=1 | 8.69 ± 3.88 | 0.047 ± 0.015 | 0.404 ± 0.017 | 5012 |
| surface NO₂ + state head | 7.84 ± 0.44 | 0.070 ± 0.046 | 0.323 ± 0.025 | 5194 |

PCA eigenvector signs are arbitrary, so the magnitude of the mode correlation is the
meaningful quantity; an early single-seed reading of this table misinterpreted a sign flip as
instability.

The map does **not** collapse to a flat field at λ=0 — spatial standard deviation falls only
about 19% and the correlation with road density is unchanged or better. What collapses is the
latent structure, and the separation is unanimous across all nine runs:

- **Only GOES AOD recovers the coastal–inland mode**: 0.698 against 0.02–0.07 for every other
  arm, a 10–35× gap with non-overlapping spreads.
- Only GOES AOD strongly recovers the solar/photochemical mode: 0.724 against 0.17–0.40.
- Switching the proxy term off collapses effective rank: 4.26 ± 0.76 against 6.86 ± 1.46,
  also non-overlapping.

Two further observations the replicates exposed. The station-proxy arm is **unstable** —
effective rank ranges 4.75 to 12.50, a spread four times any other arm's. And the state head
tightens dimensionality more than anything else tested (± 0.44) without ever recovering
physical alignment: it does something real and reproducible, just not the thing needed.

**So λ=0 is the wrong recommendation, and the earlier one in this document is withdrawn.**
Switching the proxy term off buys point metrics at 13 sites by discarding the interpretable
structure the entire analysis rests on — the coastal–inland gradient and the diurnal
photochemical cycle both disappear from the embedding.

Note also that surface NO₂ keeps the embedding high-dimensional (effective rank 8.81) but
*not* physically aligned: PC1 ~ coast is −0.064 and solar 0.386. Rich and meaningless is not
better than compact and meaningful.

### What this changes about the criterion

AOD's value was never its anomaly correlation, which is genuinely useless (−0.060). It is
that AOD's **climatology carries the basin's coastal–inland organisation** — the large-scale
structure the location encoder needs in order to be physically interpretable. Both screening
statistics used above measure something else: the anomaly test measures state information,
and the 13-site rank test measures agreement with site-mean UFP. Neither measures whether a
proxy imparts usable large-scale spatial organisation, which is what the proxy term actually
transfers and what turns out to matter.

A candidate proxy should therefore be screened on all three:
1. anomaly correlation with UFP (state information — necessary only if the head is
   state-conditioned, which on this evidence is not worth doing);
2. 13-site rank correlation of site means (does it agree with the target where we can check);
3. **whether training with it preserves effective rank and physically identifiable leading
   modes** — run `scripts/proxy_map_audit.py`. On current evidence this is the decisive one,
   and it is the only one GOES AOD passes.

## LOSO does not separate these arms

Run as a matched control rather than against a historical run:

| arm | LOSO test R² |
|---|---|
| GOES AOD, λ=1 | −0.248 ± 0.18 |
| surface NO₂ gridded | −0.250 ± 0.22 |
| surface NO₂ stations | −0.320 ± 0.18 |

All three overlap. LOSO is blind to the difference the map audit resolves cleanly, so an
earlier note in this document comparing the new arms against a −0.195 figure from a
differently-configured run is withdrawn.

That completes the picture of why λ was never tunable on monitor error: **monitor error is the
wrong instrument, not λ the wrong knob.** Every point-based test we have is either capped
(blocked_time, +0.016 spatial ceiling), insensitive (LOSO, all arms within noise), or actively
misleading (MATES point metrics prefer switching the proxy term off). The quantity the proxy
term controls is the physical organisation of the location embedding, and only
`scripts/proxy_map_audit.py` measures it.

## The alternative to acquiring anything

The criterion above is a property of the *architecture*, not of nature. Feeding meteorological
state to the proxy head —

    proxy_head([ loc_encoder(coords), state_encoder(met) ])

— was the obvious cheap fix. It is implemented and it does not work (above). The state pathway
absorbs the anomaly as designed, but that leaves the location branch targeting the proxy's
spatial climatology more precisely, and every available proxy's spatial climatology is wrong.
Making the proxy term work better makes the damage worse.

**So the list above is not optional.** The constraint is not the architecture and not the
anomaly correlation: it is that we possess no field whose spatial pattern resembles the
spatial pattern of ultrafine particle number. Items 1–3 are all attempts to construct or
obtain exactly that, and each should be screened on the 13-site statistic before anyone
trains on it. Until one of them clears zero, **λ=0 is the right setting** — the model is
better with the proxy term switched off.
