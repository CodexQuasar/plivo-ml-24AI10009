# NOTES

The model uses causal prosodic cues from the last 1.5 seconds before each pause: pause timing, energy level and slope, distributional energy statistics, voiced fraction, pitch level and slope, and the duration/shape of trailing voiced speech.  

It still struggles on pauses where the speaker sounds complete but continues after a long hesitation, and on cases where the turn-ending intonation is subtle, but the tuned ensemble is much better calibrated than the first baseline.  

Hindi currently scores better than English with this tuned feature set, which suggests the pitch-and-energy cues are useful but not yet fully language-robust.  

If I had one more day, I would add richer turn-level context features, better calibration, and a small validation sweep over ensemble weights. 