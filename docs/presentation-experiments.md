# Presentation experiments

Felix's decisions, updated 2026-11-13. This temporary branch prepares the Jetson timing
studies. Keep the statistics and reports before removing the branch after use.

| Study | Status | Scope / next step |
| --- | --- | --- |
| PCG/BDSV histograms | Updated Jetson horizon sweep ready | **B1 only** at N16/32/64/128/256/512. Plain PCG/BDSV, **PCG cap1,000**, relaxed limit barriers, ADMM off, 20 Hz updates and 500 Hz commands. SQP1, four-second figure-eight, friction mismatch, direct torque without PD/feedforward, unshifted retained warm starts, three repeats. 36 episodes / 2,880 solves planned, with scalar stage timings and batch-size checks. |
| Thesis horizon × batch matrix | Prepared for Jetson | Same six N values × B1/2/4/8/16, one SQP iteration, two repeats of 300 measured updates. Current stack, historical timing/seeding/weighting protocol. |
| Falling-cylinder B1/B16 demonstration | Discuss | Recover the historical runner and choose a paired protocol with measured solve latency charged to simulation. |
| Collision ADMM off/on | Discuss | Agree on a matched scene, clearance metrics and outcome criteria. |
| Feedback and latency compensation | Discuss | Compare plan sampling, PD, prediction, and prediction plus PD at 0/20 ms delay. |
| Cost-menu and horizon ablations | Optional; discuss | Choose a specific slide claim and match horizon-dependent weights. |
| Historical physical-arm footage | Locate if needed | Match footage to original logs. |
| New physical recordings | Separate follow-up | Agree on robot qualification first; these timing studies are simulation only. |
| Online hyperparameters, human-arm tracking, force estimation | Deferred proposals | Decide scope separately. |

Both timing studies save only scalar solve statistics, aggregate quality,
configuration and provenance. They save no trajectories, tick histories, plans
or replays. Jetson execution remains for Felix. The earlier N32/SQP5 GTX 1070
pilot predates the final scope; the new N128/SQP1 desktop comparison is a separate
cohort. The updated temporary branch prepares the full B1 horizon sweep and
retains stage timings to distinguish solver work from runtime timing bands.
The SQP1 matrix is unchanged and has not been run here.

The matrix uses bare PCG without ADMM and 30 ms knots; the histogram study
compares plain PCG/BDSV with limit barriers and 10 ms knots. Both use the current
canonical model, grasp frame and fitted armature. Increasing N increases the
prediction horizon. Treat the matrix as a new cohort rather than pooling it
with the old model/source measurements.
