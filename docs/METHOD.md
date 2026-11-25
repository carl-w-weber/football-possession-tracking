# Method

How the model works, how it was validated, what was tried and dropped, and where it falls
short. The overview is in the [README](../README.md).

## 1. Problem and metric

Each of the 199,319 frames of the tactical video gets a label: `H` (home in possession), `A`
(away in possession) or `D` (dead ball). Frames outside the two halves are not graded.

The ranking metric, **spellF1**, treats each maximal run of `H` or `A` as one possession.
Predicted possessions are matched one-to-one against true ones by temporal IoU, and the F1 of
that matching is averaged over IoU thresholds 0.3–0.7. This drives most design decisions:

- **Fragmentation is expensive.** One spurious flip splits a possession into two, which costs
  a miss *and* a false positive.
- **Timing matters.** Detecting every possession 1 s late scores about 0.65.
- **Dead balls split possessions.** `H D H` is two possessions, so stoppage detection is as
  important as team assignment.

## 2. Data

- **Tracking (ATD):** per frame, x/y in normalised pitch coordinates plus speed for 46 track
  slots (team known, identity not), and the ball. The ball is missing on 41% of graded frames,
  almost always in runs longer than 3 s.
- **Used:** tracking and its metadata (team of each slot, attack direction).
- **Not used as model input:** the videos, homography, broadcast feed and Smart Tagging.
  Smart Tagging was used only in a label-free diagnostic (`possession/diagnostics.py`).

## 3. Features (`features.py`, `dead_model.py`)

All distances are in metres and speeds in m/s.

- Ball position, linearly interpolated across gaps of at most 1 s.
- Nearest home and nearest away player distance to the ball.
- Ball speed from smoothed position differences.
- **Tempo:** mean speed of all tracked players, rolled over 0.5 / 1 / 3 / 6 s, plus its change
  over the previous and next 2 and 4 s. Players decelerate together at a whistle.
- Max and 90th-percentile player speed, and player spread.
- Ball visibility over 1 / 3 / 8 s, and time since and until the ball was seen.
- Distance from the last-seen and next-seen ball position to the touch lines and goal lines.
  Throw-ins and goal kicks restart on a line.
- How far the ball is outside the pitch.

## 4. Dead-ball detection

A `HistGradientBoostingClassifier` (300 trees, fixed seed) predicts *dead* per frame. It is
trained on the external matches only (section 7). Its log-odds, weighted and offset (`w_dead`,
`dead_bias`), plus a hard rule (ball tracked more than 0.75 m outside the pitch means dead) form
the `D` evidence.

A 3-state Viterbi decoder runs per half: entering or leaving `D` costs `pen_dead` and an
H↔A switch costs `pen_switch`. Because Viterbi sees the whole half, boundaries land where the
evidence actually changes, not half a smoothing window later. On held-out matches the median
offset of decoded stoppage boundaries is 0.0–0.04 s.

## 5. Team assignment: the controlled-touch rule (`touches.py`)

Inside each live interval:

1. **Contacts.** A contact is a run of frames with the ball within `touch_r` (0.6 m) of a
   player, credited to the nearest player's team.
2. **Restart.** The first contact after a stoppage sets the possessor.
3. **Change of possession.** An opponent's contact counts only if it is **controlled**:
   - held for at least `min_ctrl_s` (0.4 s), or
   - followed within `next_s` (3 s) by a contact from the same team (a reception or pass), or
   - **it changes the ball's course**: direction by more than `angle_deg` (35°) or speed by more
     than `speed_ratio` (50%) between the 5 frames before and after (an interception or
     clearance). Balls slower than 1 m/s are excluded.
4. **No evidence, no change.** A ball in flight, an untouched ball passing a player, or a
   missing ball keeps the previous team.

Error analysis drove step 3c. With only the duration and teammate tests, about 90% of the
missed turnovers on the validation matches were a brief opponent touch that redirected the
ball. Adding the course-change test raised held-out spellF1 from 0.61 to 0.68.

## 6. Parameters

All parameters live in `PARAMS` in `possession/model.py`. The structural ones are physical
quantities: metres, seconds, m/s. The rest were chosen by leave-one-game-out search on the
external matches (`tune.py`). Settings within about 0.02 spellF1 of each other were treated as
equivalent, and the more conservative one was kept rather than chasing two-match noise.

## 7. Validation without labels on the competition match

**External matches.** Metrica Sports' public sample games 1 and 2: about 286,000 frames of
tracking with human-annotated events. These are other matches, which the competition rules
explicitly allow for training and validation. Frame labels come from the events with fixed
rules:
- *dead* runs from the end of the event before each set piece until the set piece is taken,
  and covers the time before each kick-off;
- otherwise the possessor is the team of the latest on-ball event (pass, recovery, set piece,
  shot, ball lost, ball out).

This gives about 3.5 possessions per minute, the same rate as the competition's public window.

**Protocol.** Everything learned is trained on one match and scored on the other, using the
competition's own scorer (`evaluation/possession_metrics.py`).

**Noise simulation** (`EXT_NOISE=1`). The sample tracking is cleaner than the competition's
automatic tracking: 22 players are always tracked there, against 20.5 on average in the
competition match. The degraded copy drops 7% of player-frames in 4–16 s segments and adds
5 cm of ball jitter. Every improvement in the table below also holds on the degraded copy.

**Distribution checks.** Ball missingness covers 41% of graded frames in both datasets, and
player tempo during missing-ball stretches is almost identical (median 1.04 vs 1.05 m/s).
That supports transferring the "missing ball" semantics.

## 8. Results

| Step | clean spellF1 | noisy spellF1 | noisy macroF1 |
|---|---|---|---|
| Published baseline: nearest player + majority vote (±2 s) | 0.411 | 0.342 | 0.712 |
| Viterbi over H / A / D, hand-set evidence | 0.498 | 0.468 | 0.763 |
| + controlled-touch rule (1.2 m) | 0.574 | 0.556 | 0.777 |
| + dead-ball classifier | 0.611 | 0.583 | 0.831 |
| + course-change touch detection (0.8 m) | 0.683 | 0.646 | 0.852 |
| **Final** (0.6 m) | **0.702** | **0.660** | **0.851** |

The validation matches have 371 and 353 true possessions. The final model finds 305 and 282.
It still under-segments, mostly on possessions shorter than 2 s.

**Competition match, public 5 minutes** (17 possessions; pipeline check only, never used to
tune): spellF1 0.480 and macroF1 0.690. The published baseline scores 0.326 and 0.621.

## 9. Tried and dropped

- **Self-supervised team-shape model.** A classifier trained on frames where the ball is
  visible, with the decoder's own labels as targets and ball-free features as input (team
  centroids, spreads, forward motion). It was meant to fill the missing-ball stretches. It
  lowered held-out spellF1 (0.50 → 0.41): most missing-ball stretches are stoppages, and it
  pushed them toward `H`/`A`.
- **Cumulative-contact window** for control under pressure: no measurable gain.
- **Dead-ball classifier without ball-visibility features**, to reduce domain shift: −0.017 on
  validation, so kept the features.
- **Smart Tagging agreement as a selection signal:** it rewards long phases, so it prefers
  under-segmented outputs. Used only as a gross sanity check.

## 10. Limitations

- **Domain shift in dead-ball detection.** In the sample matches the ball is almost never
  tracked during a stoppage. In the competition match it often stays tracked through fouls and
  free kicks, which the classifier under-detects. That is the main reason this model scored
  below our team's hand-tuned model on the competition's public window (macroF1 0.690 vs
  0.844). More training matches with automatic tracking would address it directly.
- **Short possessions.** Spells under 2 s (an interception immediately lost) are about 25% of
  possessions in the sample matches and are mostly missed.
- **Systematic lead.** Transitions come about 0.4 s before the public labels, consistent with
  the reference crediting possession slightly after first contact. It is not corrected, because
  the only evidence for it is the public window.
