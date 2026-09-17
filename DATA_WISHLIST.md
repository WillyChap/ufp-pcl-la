# What this model is missing

Everything below is a data gap we ran into and could not close, ordered by how much we
think closing it would buy. Each entry states the evidence from this work, not a general
wish — if a number is quoted it came out of a run in `outputs/`, and where an item rests
on judgement rather than measurement it says so.

The short version: **this model is time-rich and space-poor.** It reproduces the hour-to-hour
and day-to-day variation at monitors well (test R² 0.70, diurnal correlation +0.98 at every
site) and cannot tell two unmonitored places apart (LOSO R² −0.20). Almost every item below
is about buying spatial degrees of freedom.

---

## 1. Ultrafine particle measurements at genuine background locations

**The gap.** All seven concurrent monitors sit at populated or near-road locations. There is
no low-concentration end to the training set.

**Evidence.** The network runs **1.42×** above an independent background sample (14,602 vs
10,299 cm⁻³; independently measured range 1.36–1.52×), so a model fitted on it predicts that
elevated level everywhere and needs an external correction to produce a usable map. Worse,
between background locations the model does not discriminate at all: rank correlation
against six independent sites is *negative* for every land-surface predictor we tested, and a
bias-corrected constant beats every model there (RMSE ratio 1.19× against 1.48–1.51×).

**What it would buy.** The single largest improvement available. It would replace the
representativeness correction — currently a scalar divided out in post-processing — with
something the model learns, and it would give the spatial background field its first real
validation. Right now we can only say the background field is *not* validated.

**Where it might come from.** Month-long campaigns with a handful of condensation particle
counters at deliberately unrepresentative sites: inland desert, a mountain site above the
basin, a marine-influenced coastal site, a low-traffic residential interior. Precision
matters less than siting — even a modest instrument at a genuinely clean site is worth more
than another monitor downtown.

## 2. More distinct monitor locations, at almost any cost to per-site quality

**The gap.** Seven concurrent sites is not enough distinct geography to fit a spatial model.

**Evidence.** **34 of 38 predictors take three or fewer distinct values across the seven
monitors.** The error decomposition is **4.5% between-site, 95.5% within-site** — nearly all
the variance the model can see is temporal. On a blocked-time split, adding the entire
land-surface embedding moves skill by **+0.016**, which is the ceiling any spatial predictor
can reach against this network, and it invalidated two experiment designs before we measured
it. In the climatology, exactly **one** term survived leave-one-out selection out of all
candidates (`log road_w_0p3km`, LOO R² 0.22 across 13 sites once historical sites are
folded in).

**What it would buy.** This is the binding constraint on the whole spatial problem. Twenty to
thirty sites spanning real contrast would change what class of model is fittable — at seven,
any flexible spatial model is fitting noise, which is exactly what we spent the overfitting
work fighting.

**Where it might come from.** Mobile monitoring is the highest-leverage answer: a
vehicle-mounted counter driving repeated routes yields thousands of distinct locations
instead of seven. Campaigns of exactly this design have been run in this basin —
Westerdahl et al. (2005) drove an instrumented platform over Los Angeles freeways and
residential streets measuring particle number and size distribution, and Fruin et al. (2008)
extended it to on-road concentrations and their predictors across the region. CARB's mobile
monitoring programme has also covered the Los Angeles air basin (Mobile Platform III) and
the port-adjacent communities of West and Downtown Los Angeles (Harbor Communities
Monitoring Study).

**Caveat we could not resolve.** These campaigns are *published*, which means the
measurements exist; none of the pages we checked states that the underlying data is
publicly downloadable. Obtaining it would likely mean contacting the investigators or CARB
directly rather than downloading a file. Treat the citations as evidence the measurement is
feasible and has precedent here, not as a download link.

Low-cost sensor networks are the second-best route — the per-site quality penalty is easily
worth the spatial coverage at this point.

## 3. Near-road gradient transects at more than one freeway

**The gap.** Near-road enhancement is the one spatial signal in this model with any support,
and it rests on a single station.

**Evidence.** Road density at 0.3 km correlates **+0.895** with site mean concentration —
but **+0.637** with one site withheld, and **+0.045** against the historical MATES sites.
That is not a stable predictor; it is one station carrying a regression. Separately, **72% of
the domain has road density outside the range the network spans**, so most of any published
map is extrapolation (`honest_surface.py` marks it).

**What it would buy.** Either a defensible near-road term or an honest retraction of it.
Given that this is the only spatial structure the model claims, resolving it is worth more
than another year of record at existing sites.

**Where it might come from.** Paired near-road / 300 m-back measurements at three or four
freeways of different traffic composition — critically including differing **heavy-duty
truck** fractions, since diesel and gasoline ultrafine emissions differ enough that a single
road-density number should not stand in for both.

This design has a direct precedent in the same basin. Zhu et al. (2002) measured particle
number and size distribution along distance transects from two Los Angeles freeways chosen
for contrasting fleet composition: I-405, where under 5% of vehicles are heavy-duty diesel,
against I-710, where more than 25% are. Concentrations within roughly 50 m downwind reached
up to 30× the background found further away. Zhu et al. (2004) added seasonal repeats. That
is the measurement that would settle whether our road-density term is real or is one
station's regression — and it also shows the truck-fraction contrast we would need is
achievable with two well-chosen sites rather than a large network.

## 4. Size-resolved number distributions, not just total particle number

**The gap.** The target is integrated total number concentration. That single number sums two
physically distinct populations — fresh combustion particles and photochemically nucleated
ones — with no way to separate them.

**Evidence.** The model's pathway apportionment says the secondary contribution peaks near
midday (**0.31** of attributed gradient at 12:00) and falls to **0.17** by 16:00, while the
primary contribution rises from **0.21** to **0.30** across the same window. That is the
right physical shape, and there is nothing in the data to check it against — it is inferred
from integrated gradients over predictor groups, not measured.

**What it would buy.** Size-resolved data would come from the same instrument pairing the
Los Angeles transect studies already used — a condensation particle counter for total number
alongside a scanning mobility particle sizer covering roughly 6–220 nm (Zhu et al. 2002), so
items 3 and 4 can be closed by one deployment rather than two. Size-resolved data (or even a
few coarse size bins) would
turn the most physically interesting result in this work from an inference into a
measurement, and would let the two pathways be fitted as separate targets rather than
disentangled after the fact. The nucleation mode below ~30 nm and the traffic mode near
60–100 nm respond to different drivers; one model predicting their sum is doing strictly
harder work than two models predicting each.

## 5. Co-located chemical speciation

**The gap.** No chemical measurement supports the source attribution.

**Evidence.** The same as item 4: the primary/secondary split is entirely model-internal.
The spatial leading mode is coastal-inland (r = +0.69 with distance from coast) and the
leading temporal mode is solar (r = +0.81 with solar elevation, −0.02 with clock hour) —
consistent with photochemistry, but consistent is not confirmed.

**What it would buy.** Black carbon is the cheapest and most direct: it is a clean primary
combustion tracer, so co-located BC would validate the primary pathway almost immediately.
For the secondary pathway, gas-phase SO₂ and ammonia plus a marker of condensable organics
would constrain nucleation. Without these the apportionment stays a plausible story.

## 6. A proxy whose *spatial pattern* actually responds to atmospheric state

**The gap.** The proxy-consistency term has less to work with than its design assumes. Both
candidate proxies carry a spatial pattern that is almost entirely climatological.

**Evidence.** With the domain-mean hourly signal removed, a pure climatology of place, hour
and day-of-year explains R² = **0.620** of the GOES AOD spatial pattern; adding the full
meteorological regime state moves it by **−0.008** (−2% of the residual). For TEMPO NO₂ the
same test gives 0.602 and **+0.014** (4%). In other words the proxy tells the model where the
map is high, but essentially nothing about how today's map differs from the average map. This
is the most likely explanation for the finding that λ is not tunable on monitor error: the
proxy term is regularising toward a static field.

**What it would buy.** This is the gap most directly relevant to the proxy-loss method
itself. A proxy with genuine state-dependent spatial structure would let the consistency term
carry dynamical information instead of a climatology, which is what the formulation was
built to exploit.

**Where it might come from.** TEMPO NO₂ at native resolution rather than regridded is worth
testing first — our regridding may itself be averaging away the state dependence, and it
costs nothing to check.

Beyond that, a chemical transport model field would be state-dependent by construction,
which is exactly the property the proxy term needs. We looked for an existing product and
the situation is worth stating precisely, because the resolution and the relevant physics
both exist but have never been combined over this basin:

| What exists | Resolution | Period | Carries particle number? |
|---|---|---|---|
| LA Basin CMAQ v5.3.2 (Wang et al. 2024) | **1 km**, LA Basin | April 2020 only | No — mass and speciation only |
| South Coast CMAQ v5.4, 2022 Scoping Plan | 4 km / 1 km nested | scenario runs | No |
| CONUS air quality reanalysis (Kumar et al. 2025) | 12 km, hourly | 2005–2018 | No — PM₁/PM₂.₅/PM₁₀, O₃, NO₂, CO |
| CMAQ-APM (Mao et al. 2025) | 12 km, CONUS | Jun–Jul & Nov–Dec 2013 | **Yes** — size-resolved, CN10 and CCN |
| WRF-Chem + APM (Yu & Luo 2011) | Eastern US | — | **Yes** |

Three things follow. First, the one public high-resolution chemistry archive for this exact
domain (Wang et al., 1 km, output deposited at Stanford Digital Repository
`10.25740/qc346hv0119`) covers a single month in 2020 and is mass-only. Second, the one
public *reanalysis* is 12 km, mass-only, and ends in 2018 — before our target period. Third,
and most important: **standard CMAQ cannot supply what we need even at 1 km.** Its default
aerosol scheme carries three lognormal modes, which Mao et al. state explicitly is inadequate
for simulating ultrafine number; number concentration requires a size-resolved microphysics
module (APM, or a sectional scheme such as TOMAS) with an explicit nucleation mechanism —
in APM's case ternary H₂SO₄–H₂O–NH₃ ion-mediated nucleation. That configuration has been run
over the eastern US and over CONUS at 12 km, and as far as we can find, never over Los
Angeles at high resolution.

So the honest form of this wish is not "someone should download the LA chemical reanalysis."
It is: **the LA high-resolution domain and the number-capable microphysics have never been
run together, and doing so is a modelling project rather than a download.**

One encouragement and one caution about that project. The encouragement: a CTM proxy does not
need to be unbiased to be useful here. The proxy-consistency term constrains spatial
*pattern* and its response to state, not absolute level, and our finding is that the
satellite proxies carry almost no state-dependent pattern at all — so even a substantially
biased number field would supply information the current proxies do not. The caution: CMAQ-APM's
reported normalised mean bias for CN10 spans −6% to +192% across evaluation sites with
correlations of 0.59–0.66 at the better ones, so the field would come with real uncertainty,
and the coastal-subsidence meteorology of this basin is not among the regimes it has been
evaluated in.

Note also why a mass-only field cannot be substituted, however fine its grid: ultrafine
particles contribute negligibly to PM₂.₅ mass, so a well-validated mass field carries almost
no information about number. This is the same reason the problem is hard in the first place.

### 6a. What makes a proxy useful, and the one already sitting on disk

Before commissioning any new field it is worth stating what the proxy term actually requires,
because the criterion is testable in an afternoon and we had not applied it. A proxy is useful
if it passes four tests:

1. **Anomaly test (necessary).** Remove the same per-site hour-of-day × month climatology from
   both the proxy and the target. Whatever correlation survives is information a climatology
   term cannot already supply. If this is near zero, the proxy term is an expensive way to
   regularise toward a map we fit directly anyway.
2. **Non-redundancy test.** The correlation must survive controlling for predictors the model
   already has. A proxy that only tracks dilution duplicates the mixing-depth feature.
3. **Spatial reach test.** The correlation must persist at the distance over which you intend
   to extrapolate. A proxy that works only where a label already exists is useless.
4. **Density test.** The proxy must be observed at more distinct locations than the label.
   This is the one that matters most here, and it does *not* require a gridded field.

Applying these to everything on hand (`scripts/screen_proxies.py`,
`scripts/screen_proxy_distance.py`) gives a result that reframes this item. After climatology
removal, 76% of the UFP standard deviation remains — so the anomaly is where the variance is —
and the correlations against it are:

| candidate | r | r² | note |
|---|---|---|---|
| TEMPO NO₂ column | −0.009 | 0.000 | currently used as the proxy |
| TEMPO HCHO column | −0.079 | 0.006 | present in the master file, never used |
| TEMPO HCHO/NO₂ ratio | −0.005 | 0.000 | |
| TEMPO NO₂ / mixing depth | +0.086 | 0.007 | column → approximate surface concentration |
| GOES AOD | −0.060 | 0.004 | currently used as the proxy |
| AQS PM₂.₅, nearest station | +0.132 | 0.017 | mass, as expected, carries little number signal |
| HRRR mixing depth *(reference — already a model feature)* | −0.251 | 0.063 | |
| **AQS surface NO₂, nearest station** | **+0.533** | **0.285** | **21 stations** |

**Every satellite column fails the anomaly test outright** (|r| ≤ 0.09, i.e. below the
reference meteorological feature the model already contains). Surface NO₂ from the regulatory
network passes it by a wide margin. This is the cleanest available explanation for why λ was
never tunable on monitor error: the proxy term has been constraining the representation with a
field that carries essentially no state information about the target.

Surface NO₂ passes the other three tests as well:

- **Non-redundancy.** Partial correlation controlling for HRRR mixing depth is **+0.494**,
  against +0.533 unconditioned. Almost none of the skill is dilution the model already sees.
- **Spatial reach.** Four of the six overlapping UFP monitors happen to be co-located with an
  NO₂ station, so the pooled figure is flattered and should not be quoted alone. The
  distance-decay is the honest number: r = +0.533 at 2.7 km mean separation, +0.366 at 9.7 km,
  +0.364 at 13.4 km, +0.314 at 21 km, **+0.312 at 30 km**. The two genuinely non-co-located
  monitors give +0.470 at 4.6 km and +0.338 at 8.8 km. The signal is regional, not an artefact
  of sampling the same air.
- **Density.** 21 NO₂ stations against 7 UFP monitors — three times the spatial degrees of
  freedom, which is the binding constraint identified in item 2.

**These ideas were tested. They do not work — see the note below before acting on them.**

The obvious moves were: swap the proxy from satellite column to surface NO₂ (the data is
already downloaded); treat NO₂ as a multi-task target at its own 21 stations rather than
gridding it; and add a state pathway to the proxy head so the anomaly correlation becomes
usable. All three were implemented and evaluated over three seeds each.

### Outcome: the screening statistic above is the wrong one

None of them helped, and the reason revises the criterion. The proxy head is
`proxy_head(loc_encoder(coords))`, and the location encoder receives only coordinates and
calendar features — no meteorology. It can therefore only ever represent
`E[z | place, hour, day-of-year]`, the proxy's space-time **climatology**. However
informative a proxy's anomalies are, the term as written cannot transfer them.

What the proxy term actually supplies is large-scale spatial *organisation* to the location
embedding, and that is invisible to every point metric. Measured over three seeds
(`scripts/proxy_map_audit.py`):

| arm | effective rank | \|PC1 ~ coastal distance\| | \|r\| solar |
|---|---|---|---|
| **GOES AOD, λ=1 (the released model)** | 6.86 ± 1.46 | **0.698 ± 0.072** | **0.724 ± 0.110** |
| proxy term off (λ=0) | 4.26 ± 0.76 | 0.020 ± 0.015 | 0.173 ± 0.014 |
| surface NO₂ at 21 stations | 8.69 ± 3.88 | 0.047 ± 0.015 | 0.404 ± 0.017 |
| surface NO₂ + state-conditioned head | 7.84 ± 0.44 | 0.070 ± 0.046 | 0.323 ± 0.025 |

Only GOES AOD recovers the coastal–inland and solar/photochemical modes — a 10–35× gap on the
spatial mode, with non-overlapping spreads across all nine runs. Switching the proxy term off
collapses the embedding's effective rank. Surface NO₂ keeps the latent space
high-dimensional but not physically aligned.

Point metrics point the other way and should not be trusted here: at λ=0 the MATES rank
correlation is its best value anywhere (+0.029 against −0.275), and LOSO cannot separate the
arms at all (−0.248 ± 0.18, −0.250 ± 0.22, −0.320 ± 0.18). This is why λ was never tunable on
monitor error — **monitor error is the wrong instrument, not λ the wrong knob.**

**So the released configuration stands: GOES AOD at λ = 1.** Its anomaly correlation with UFP
is −0.060 and its 13-site climatology rank is −0.363; it is nonetheless the only proxy tested
that preserves a physically interpretable model. A candidate proxy must be screened on
`scripts/proxy_map_audit.py`, not on correlation with the target.

`PROXY_TARGETS.md` documents the full investigation and the proxies worth acquiring next.

One genuinely free finding survives: `VCD_HCHO` sits unused in the TEMPO master file. It fails
the anomaly test (r = −0.079), so it is not a proxy candidate — worth knowing, one run to
establish.

## 7. Hourly traffic volume instead of annual-average counts

**The gap.** Traffic enters as AADT — an annual average daily figure — so the traffic field is
static in time while the thing it is supposed to explain has a strong diurnal cycle.

**Evidence.** The model reproduces the diurnal cycle very well (r = +0.98 at every site, peak
hour matched), but it must do so through meteorology and solar geometry, because the traffic
predictor cannot vary by hour. The morning concentration peak is almost certainly a traffic
peak the model is currently explaining with the wrong variable.

**Where it comes from.** The Caltrans Performance Measurement System, **PeMS**
(`https://pems.dot.ca.gov`), is the archive of California's freeway loop-detector network and
is the hourly counterpart to the annual counts we already use. District 7 covers Los Angeles
and Ventura counties with roughly 4,200 detectors at about 1,300 locations, reporting
**5-minute** averaged flow (vehicles/hour) and speed, with more than ten years retained for
historical analysis. It also carries vehicle classification, which would supply the
heavy-duty fraction that item 3 needs and that road density cannot express.

Two practical notes. An account must be applied for and is approved in one to two business
days; we could not confirm from the site whether registration carries a fee, though the
dataset is used routinely in published research. More importantly, PeMS instruments the
**state highway network only** — freeway mainlines. Arterial volumes are not in it and would
have to come from the city separately (LADOT's signal-system counts), which matters because
several monitors are not freeway-adjacent.

**What it would buy.** Real time dependence in the primary pathway. Of everything on this
list this is the cheapest to obtain relative to what it would clarify, and it would sharpen
item 4's apportionment as a side effect.

## 8. Boundary-layer height that is actually resolved

**The gap.** Dilution depth comes from reanalysis and forecast-model fields at 3–30 km
resolution, which is coarse relative to the coastal and terrain-driven structure of this
basin.

**Evidence.** Meteorology is the largest attributed pathway at almost every hour (0.45–0.52 of
attributed gradient), so the model is leaning heavily on fields whose vertical structure is
parameterised rather than observed. The latent space has an effective rank of 6.6, and the
leading spatial mode is coastal-inland — a boundary-layer signature.

**What it would buy.** Ceilometer or lidar mixing-height observations would let this be
checked rather than assumed, and would separate genuine emission differences from dilution
differences — a distinction the current model cannot make.

## 9. Historical and current measurements that overlap in time

**The gap.** The historical monitoring sites we used are not concurrent with the target
period, so they can support the spatial climatology but cannot serve as held-out validation.

**Evidence.** Folding them in raises the climatology to 13 sites and gives LOO R² 0.22, which
is why they are in the final model. But because they do not overlap in time, an independent
test at those locations conflates spatial error with multi-year trend, and we could not use
them to settle the LOSO ranking — that had to be done separately, and it overturned the
ranking the LOSO protocol alone produced.

**What it would buy.** Even one year of overlap at a few historical locations would convert
them from climatology anchors into genuine spatial validation sites, which is the single
weakest link in the evaluation.

## 10. Source terms for the port complex and the airport

**The gap.** Two of the largest ultrafine sources in this basin have no representation in the
predictor set.

**Evidence.** Judgement as far as *our* network goes — we have no monitor close enough to
either source to demonstrate the omission directly. But ship, locomotive and aircraft
emissions are not proportional to road density, the only combustion-related spatial predictor
in the model, so their signature cannot currently be expressed at all. The external evidence
that this matters is strong: Hudda et al. (2014) mapped the plume downwind of LAX with an
instrumented vehicle and found particle number concentrations elevated roughly **4-fold at
10 km downwind** — a feature spanning a large fraction of our domain, which our predictor set
has no way to represent. That study is also a second demonstration that the mobile approach
in item 2 works in this basin.

**What it would buy.** Little for station metrics; potentially a lot for the map, in the
part of the domain most likely to matter for exposure work.

---

## What would *not* help much

Worth stating, because two of these looked promising and measurably were not:

- **More years at the existing seven sites.** The error is 95.5% within-site; the temporal
  problem is already well constrained and more of the same record does not touch the spatial
  one.
- **Higher-capacity models.** We located the overfitting in the location branch and had to
  add decoupled weight decay, coordinate jitter and sign constraints to suppress it. With 34
  of 38 predictors taking ≤3 distinct values, extra capacity buys memorisation.
- **Finer land-surface embeddings alone.** Resolution genuinely matters — the embedding must
  be exported at 30 m, since at 200 m the near-road signal is averaged away entirely
  (r = +0.76 against −0.05). But even at 30 m the whole embedding is worth +0.016 on a
  blocked-time split, because seven sites cannot constrain a rich spatial field no matter how
  good it is. Resolution is necessary and nowhere near sufficient.


---

## Sources for the items above

Measurement precedents in this basin, and the traffic archive:

- Westerdahl, Fruin, Sax, Fine, Sioutas (2005). *Mobile platform measurements of ultrafine
  particles and associated pollutant concentrations on freeways and residential streets in
  Los Angeles.* Atmospheric Environment.
  https://www.sciencedirect.com/science/article/abs/pii/S1352231005002232
- Fruin, Westerdahl, Sax, Sioutas, Fine (2008). *Measurements and predictors of on-road
  ultrafine particle concentrations and associated pollutants in Los Angeles.* Atmospheric
  Environment. https://www.sciencedirect.com/science/article/abs/pii/S135223100700859X
- Zhu, Hinds, Kim, Shen, Sioutas (2002). *Study of ultrafine particles near a major highway
  with heavy-duty diesel traffic.* Atmospheric Environment (I-710 vs I-405 transects).
  https://www.sciencedirect.com/science/article/abs/pii/S1352231002003540
- Zhu, Hinds et al. (2004). *Seasonal trends of concentration and size distribution of
  ultrafine particles near major highways in Los Angeles.* Aerosol Science and Technology.
  https://www.tandfonline.com/doi/abs/10.1080/02786820390229156
- Hudda, Gould, Hartin, Larson, Fruin (2014). *Emissions from an international airport
  increase particle number concentrations 4-fold at 10 km downwind.* Environmental Science &
  Technology. https://doi.org/10.1021/es5001566
- California Air Resources Board, *Mobile Monitoring Research Studies* (Mobile Platform III,
  Los Angeles air basin; Harbor Communities Monitoring Study).
  https://ww2.arb.ca.gov/resources/documents/mobile-monitoring-research-studies
- Caltrans Performance Measurement System (PeMS). https://pems.dot.ca.gov

Chemical transport modelling over this domain, and particle-number capability (item 6):

- Wang et al. (2024). *An updated modeling framework to simulate Los Angeles air quality –
  Part 1.* Atmos. Chem. Phys. 24, 2345. CMAQ v5.3.2 at 1 km over the LA Basin, April 2020.
  https://acp.copernicus.org/articles/24/2345/2024/ — output at
  https://doi.org/10.25740/qc346hv0119
- Kumar, Bhardwaj, He, et al. (2025). *A long-term high-resolution air quality reanalysis
  with a public-facing air quality dashboard over the Contiguous United States (CONUS).*
  Earth Syst. Sci. Data 17, 1807–1834. 12 km hourly, 2005–2018.
  https://doi.org/10.5194/essd-17-1807-2025 — data at https://doi.org/10.5065/cfya-4g50
- Mao et al. (2025). *Improved simulation of particle number concentrations over the US:
  integrating a size-resolved advanced particle microphysics model into CMAQ.* J. Geophys.
  Res. Atmos. https://doi.org/10.1029/2025JD044021
- Yu & Luo (2011). *Simulation of particle formation and number concentration over the
  Eastern United States with the WRF-Chem + APM model.* Atmos. Chem. Phys. 11, 11521.
  https://acp.copernicus.org/articles/11/11521/2011/
- High-resolution South Coast CMAQ for California's 2022 Scoping Plan, Environ. Sci.
  Technol. (2025). https://pubs.acs.org/doi/10.1021/acs.est.5c13824

Everything else in this document comes from runs in `outputs/` in this repository.

---

## Proposal ideas

Two proposals that would close most of the list above. They are deliberately complementary:
the campaign produces the evaluation data the reanalysis cannot validate without, and the
reanalysis produces the spatial prior the campaign's sparse points cannot interpolate
between. Each is fundable alone; declaring the other as a companion strengthens both.

The useful asset in writing either is that the motivating evidence is already measured. Most
proposals argue that a gap *might* matter; the numbers in this document show that it does.

### A. Modelling and data assimilation

**A particle-number reanalysis for the Los Angeles Basin: size-resolved aerosol microphysics
under hourly geostationary constraint.**

*Premise.* The high-resolution domain and the number-capable microphysics have never been run
together (item 6). Wang et al. reach 1 km over the basin but mass-only, for one month; Mao et
al. produce particle number but at 12 km CONUS, for two months of 2013. No hourly gridded
ultrafine *number* field exists for this airshed.

*Work.* CMAQ+APM (or WRF-Chem+APM) at 4 km over the South Coast Air Basin with a 1 km inner
nest, run for a full year, assimilating TEMPO NO₂ at hourly cadence together with surface
monitors. The deliverable is a public, size-resolved, hourly number-concentration reanalysis.

*Novelty, in ascending order of interest.* First number-resolved chemical transport
simulation at neighbourhood scale over Los Angeles; hourly assimilation of geostationary
composition retrievals, which is still an open problem; and — the one to lead with — use of
the resulting field as a physically consistent proxy for machine-learned exposure surfaces.

*Preliminary evidence from this work.* With the domain-mean signal removed, a pure
climatology of place, hour and day-of-year explains R² = 0.620 of the GOES AOD spatial
pattern, and adding the full meteorological regime state changes it by **−0.008**. For TEMPO
NO₂ the same test gives 0.602 and **+0.014**. The satellite fields currently used to constrain
statistical air-quality models therefore carry almost no information about how a given day's
map departs from the average map. That is a quantitative statement of why a dynamical proxy
is needed, and it is a result rather than an expectation.

*Likely homes.* NASA ROSES — Atmospheric Composition Modeling and Analysis, or the Health and
Air Quality applications element if the exposure framing is preferred; a TEMPO science team
call is the most natural fit given the assimilation component.

*Risk to state plainly.* There is no ultrafine number evaluation dataset over Los Angeles to
validate the product against — which is what proposal B produces. Name the circularity in the
text rather than leaving a reviewer to find it.

### B. Observational campaign

**Model-informed sampling design: a background-anchored mobile campaign to constrain
ultrafine particle spatial structure in Los Angeles.**

*Premise.* The contribution is not additional measurement but the use of a model's diagnosed
failure modes to decide where measurement should go. Siting is chosen to maximise contrast in
the predictors that current models cannot distinguish, rather than to maximise population
coverage.

*Design, in three tiers, each closing a numbered gap above.*

1. Four to six fixed condensation particle counters for twelve months at deliberately
   unrepresentative locations — inland desert, a site above the basin, marine-influenced
   coastal, low-traffic residential interior (item 1).
2. Mobile mapping with a condensation particle counter and scanning mobility particle sizer,
   on routes stratified by road density and land cover, with repeat visits across hours and
   seasons so that spatial and temporal variation separate (item 2).
3. Paired near-road transects at three or four freeways contrasted by heavy-duty fraction,
   following the I-405 / I-710 design (item 3).

Black carbon co-located throughout is inexpensive and converts the primary/secondary split
from a model inference into a measurement (items 4 and 5).

*Preliminary evidence from this work.* Thirty-four of thirty-eight predictors take three or
fewer distinct values across the seven available monitors. Error decomposes as 4.5%
between-site and 95.5% within-site. The entire land-surface embedding is worth **+0.016** on a
blocked-time split. The network runs **1.42×** above an independent background sample. And the
single spatial signal the model claims rests on one station: road density at 0.3 km correlates
+0.895 with site means, +0.637 with one site withheld, and **+0.045** against the historical
sites.

*Likely homes.* The Health Effects Institute is the closest fit, given both the health
relevance of ultrafine particles and a history of funding measurement. NSF Atmospheric
Chemistry suits the process-science framing; CARB or SCAQMD suit the regional and
environmental-justice framing.

*Risk to state plainly.* Campaigns are read as descriptive. The sampling-design argument —
selecting locations to maximise predictor contrast, in the spirit of optimal experimental
design — is what makes it methodological rather than observational.

### Sequencing

If only one can go forward first, B reduces risk for A and not the reverse: the reanalysis
needs number observations to be evaluated against, while the campaign is self-contained. The
weakness common to both is that neither closes the link from exposure surface to health
outcome, which should be addressed explicitly in any submission to a health-focused sponsor.

Programme names above reflect the funding landscape as of writing and should be checked
against current solicitations before scoping.
