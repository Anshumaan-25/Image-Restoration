# Local screening — completed

All three approved arms completed 1,000 updates each with zero skips: A 18.56, B 19.14, C 18.74 minutes; combined 56.44. Selected steps A=1000, B/C=900. Development: eight images, 16 phases, fixed center 128-pixel crops, Q20, project RGB float metrics.

| Observation | Mean per-image worst-phase PSNR |
|---|---:|
| JPEG input | 29.230958 dB |
| A random-phase pixel-only model | 29.239696 dB |
| B stratified pixel-only model | 29.256013 dB |
| C stratified plus consistency | 29.241428 dB |

Models remain near input quality. C trails B by 0.014584 dB on selected development data; do not expand the consistency study automatically. Protected test remains unscored. [Complete report](results.md), [summary](studies/mps_screening_v1/summary.json), [audit](final_audit.md) and [proposed next diagnosis](../docs/architecture/research_checkpoint.md) supersede the earlier partial progress observations.
