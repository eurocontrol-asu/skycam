# Evaluate daily dirt detection

The pale spot reported on 2024-05-10 is a known qualitative failure case. No
manually reviewed spot mask has been supplied for this change, so no recall,
precision or improvement is claimed. The detection thresholds and segmentation
algorithm remain unchanged.

1. Annotate a frozen set of days from one camera at native image coordinates.
   Separate installation, daily dirt/drops, clear sky and unknown/ambiguous
   pixels. Include faint and bright spots, thin objects, clean days, haze/fog,
   different exposures and seasons. Record camera, calibration, dates, image
   dimensions, annotation revision and reviewer. A spot outline must come from
   reviewed images, not from the computed mask being scored.
2. Fix and version the installation artifact and its reference-day selection.
   Keep the same artifact, geometry, analysis domain and parameters throughout
   a comparison. Include this identity in every result. Changing five reference
   days to ten changes the baseline and cannot alone demonstrate better dirt
   detection.
3. Score installation and daily static separately. Inside the valid annotated
   domain, call skystatic.evaluate.agreement with classes == 2 for daily static.
   Report pixel precision, recall and IoU, both strict and at a declared pixel
   tolerance. Exclude class 3 and unknown annotation regions; resize class labels
   only with nearest neighbours. Report counts and undefined ratios explicitly.
4. Small spots can disappear in an aggregate pixel score. Also report each
   annotated connected component's detection and missed area, stratified by
   contrast and size; retain overlays for all false positives and misses. State
   the overlap rule before scoring. Include per-day results rather than only
   pooled totals.
5. Report skyline-visible and skyline-missing days separately. The latter carry
   a conservative quality flag; do not silently exclude them from overall
   coverage or treat their daily-static mask as equally validated.
6. Keep exploratory cases (including the reported May 10 spot) distinct from
   fresh evaluation days. Freeze labels and the evaluation protocol before
   tuning. Compare a fixed baseline and candidate on the same inputs, then
   evaluate on untouched days without further threshold selection. Do not
   rerun or tune an unrelated frozen research campaign as part of package tests.

Store images, labels, manifests, masks and result tables under the existing
working/durable data roots, never in the code repository. Each campaign should
have a README beginning with its storage status, its dataset/annotation hash,
installation artifact ID, code commit, command, environment lock and metrics.
The package's synthetic regression tests validate software behavior; they do
not establish sensitivity to real-camera dirt.
