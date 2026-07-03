# chemeng — Computational Chemical Engineering, herbouwd van nul

Volledige Python-herbouw van de kern van het archief van P. Vlasschaert
(petervlaschemeng.weebly.com, ondersteund door prof. dr. D. Constales, UGent).
Het origineel (TrueBasic + Maxima, 2015-2024) is van de grond af opnieuw
geïmplementeerd in NumPy/SciPy/sympy — inclusief de verificatielaag die het
origineel miste. Publicatie met instemming en volmacht van de auteur.

## Verificatie: `python3 verify.py` → 33/33 PASS

Elke module wordt getest tegen een analytische oplossing of onafhankelijke
referentie (geen zelfbevestiging):

| Module | Referentie |
|---|---|
| TDMA | np.linalg.solve op willekeurig diag.-dominant stelsel |
| Newton + symbolische Jacobiaan | residu machineprecisie |
| SOR 2D-Laplace | exacte oplossing u=y |
| Transiënte geleiding (plaat) | Fourier-reeksoplossing (Δ<0,05 K op 400 K) |
| Cilinder/bol (r=0 via L'Hôpital+ghost node) | regulariteit, symmetrie, fysische ordening bol<cil<plaat |
| VdW / RK / PR | P(V-wortels)≡P + ideale-gaslimiet Z→1 |
| Bubble-T + azeotroop (Margules-2) | ethanol/water: x_az=0,937 · 78,3 °C (lit. ±0,894 · 78,2 °C) |
| Rachford-Rice flash | massabalans per component < 1e-12 |
| McCabe-Thiele | Rmin-constructie + trapstelling |
| Smoker (1938) | ≡ Fenske bij totale reflux (6,4269 = 6,4269) |
| Rayleigh batch | massabalans + verrijkingslogica |
| Buisreactor Danckwerts-BC | gesloten Wehner-Wilhelm-oplossing (Δ<1e-4 bij Pe=50) |
| FVM CDS/upwind/hybrid/power-law | exacte oplossing + roosterconvergentie |
| Symbolische schema-afleiding | reproduceert aW=D+F/2, aE=D−F/2 (Versteeg & Malalasekera) |
| **Bubble Point-kolom (Wang-Henke)** | massabalans <2e-8 kmol/h · Σx=ΣKx=1 op elke trap · T monotoon 80,8→118,5 °C · benzeen→top/xyleen→bodem · N↑ ⇒ zuiverheid↑ |

## Structuur

    chemeng/
      numerics.py      TDMA · Newton met sympy-Jacobiaan · SOR
      heat.py          1D geleiding · transiënt plaat/cilinder/bol (impliciet)
      eos.py           CubicEOS → VanDerWaals / RedlichKwong / PengRobinson
      vle.py           Antoine · Margules-2 · bubble-T · Txy · azeotroopzoeker
      flash.py         Rachford-Rice (brentq binnen asymptootgrenzen)
      distillation.py  McCabe-Thiele · Smoker · Rayleigh
      rigorous.py      Bubble Point multicomponent-kolom (Wang-Henke, CMO)
      reactor.py       axiale dispersie + Danckwerts-BC's + analytische ref.
      fvm.py           4 convectie-diffusieschema's + symbolische afleiding
    verify.py          volledige verificatiesuite (33 tests)
    examples/demo.py   vier demonstratiefiguren → demo.png
    LICENSE            MIT-voorstel met dubbele attributie (nog te bevestigen)

## De signatuur van het origineel, behouden

De kenmerkende werkwijze van het archief — **eerst symbolisch afleiden, dan
numeriek rekenen** (origineel: Maxima → TrueBasic) — is bewaard:
`newton_symbolic()` genereert de Jacobiaan symbolisch met sympy en
`fvm.symbolic_scheme()` leidt de FVM-coëfficiënten symbolisch af vóór gebruik.

## Bewust behouden fysica uit het origineel

- Danckwerts-randvoorwaarden (gemengd aan inlaat, nulgradiënt aan uitlaat)
- r=0-singulariteitsbehandeling via L'Hôpital + ghost node
- fysische wortelselectie bij kubische EOS (V > b)
- MESH-formulering met tridiagonale componentbalansen (Wang-Henke)

## Roadmap

Thiele-Geddes / θ-methode · energiebalans-update van V in de BP-kolom
(niet-CMO) · activiteitsmodellen in de kolom-K's · residue curves ·
SIMPLE 1D/2D · FEM-assemblage · batch-dynamiek met enthalpie.
De fundamenten (TDMA, flash, Newton-sym, VLE, BP-raamwerk) liggen er.

Afhankelijkheden: numpy · scipy · sympy · matplotlib
