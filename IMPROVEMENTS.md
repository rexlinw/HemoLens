# HemoLens — possible improvements

A backlog of enhancements you can pick from by goal (accuracy, trust, production, research). Nothing here is required for the app to run.

---

## Accuracy & science

1. **True paired labels** — Train nail/palm with per-person lab Hb (same draw as the photo), not class-based synthetic Hb. This is the main gap between training metrics and real-world use.
2. **Calibration** — Post-hoc calibration (e.g. isotonic / Platt) on a held-out **mobile-capture** set so g/dL aligns better with labs.
3. **Device & lighting normalization** — Augment with phone-like noise, white balance, JPEG; or add simple color constancy in preprocessing.
4. **Population shift** — More cohorts (skin tone, age, geography, cameras) and report **per-subgroup** error, not one global R²/MAE.
5. **Uncertainty** — Quantile regression, ensembles, or conformal intervals so the UI can show a likely range instead of a single number.
6. **Conjunctiva-specific capture** — Match training: lower palpebral conjunctiva framing, not generic “eye” selfies.

---

## Product & UX (mobile)

7. **On-screen guides** — Overlays or skeleton UI for eye / nail / palm so users match the training distribution.
8. **Live quality meter** — Blur, exposure, face/palm-in-frame before “Analyze” (server-side validation already exists; mirror hints in the client).
9. **Retake flow** — If `invalid_image`, show which modality failed and how to fix it (modality-specific messaging in the app).
10. **History & trends** — Local history with date; optional export for a clinician (with clear “not diagnostic” copy).
11. **Accessibility** — Larger tap targets, VoiceOver labels, high-contrast mode.
12. **Localization** — Copy in local languages for WHO ranges and disclaimers.

---

## Backend & API

13. **Versioning** — Expose `api_version` and `model_version` in `/health` so clients can detect mismatches.
14. **Structured errors** — Stable error codes (e.g. `EYE_NOT_DETECTED`, `BLURRY`, `MULTIMODAL_UNAVAILABLE`) for the app to branch on.
15. **Rate limiting & abuse protection** — Especially important on free hosting tiers.
16. **Optional auth** — API keys or user accounts for accountability or paid tiers.
17. **Async / larger uploads** — If you add video or very high-res: size limits and clear 413/400 responses.
18. **Observability** — Log aggregation and simple metrics (latency, validation failure rate, modality mix).

---

## ML engineering

19. **Reproducible runs** — Pin dependencies, fixed seeds, saved training config YAML next to each artifact (`.pkl`).
20. **Evaluation harness** — One script path: train → evaluate → write `multimodal_config.json` plus tables by Hb band.
21. **Model serving** — If models grow: ONNX or smaller architectures for edge/on-device inference.
22. **Safe rollouts** — Feature flags or shadow predictions before swapping production weights.

---

## Trust, compliance, clinical framing

23. **Clear labeling** — “Screening estimate only,” not diagnosis; cite limitations (lighting, not a lab substitute).
24. **Consent & data retention** — Policy on image logging; default to no image storage unless explicitly needed.
25. **Regulatory path** — If targeting clinical workflows, plan documented validation beyond offline ML scores.

---

## Quick reference: fast vs deep work

| Faster wins | Larger payoff, more effort |
|-------------|------------------------------|
| API version + error codes | Paired nail/palm–Hb datasets |
| Capture overlays + live blur hints | Calibration on real phone uploads |
| Better `invalid_image` UX | Subgroup metrics + uncertainty in UI |

---

*Last updated from project discussion; extend or reprioritize as your goals shift.*
