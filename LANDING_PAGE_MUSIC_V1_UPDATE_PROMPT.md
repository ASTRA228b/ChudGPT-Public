# Landing-page update prompt

Update the ChudGPT landing site to describe the completed ChudGPT-Public-Music V1 "Intelligent Slop" quality pass.

Keep the existing visual style and navigation. Update the Music V1 card and `/model-guide` entry with these verified facts:

- Music V1 remains the same custom 20,999,184-parameter decoder-only model family; it was not replaced by an external model.
- Its training corpus was rebuilt to 3,600 unique conversations and 4,320 assistant outputs.
- Title variety increased from 80 to 506 unique titles; style variety increased from 464 to 1,252 unique styles.
- Five stock lyric lines that previously appeared 1,076 times each were removed from the dominant pattern. The deployed runtime now uses four-token no-repeat decoding, repetition-aware multi-candidate ranking, and recent-session title/style/line diversity checks.
- Music V1 keeps its deliberately strange, funny "slop" personality, but the update focuses on more varied subjects, structures, titles, styles, and better-connected weirdness.
- Be honest that this is still a small experimental model and may produce awkward or inaccurate lyrics.
- Do not claim the rejected retrained checkpoint was deployed. Testing showed lower validation loss but worse human-readable coherence, so the proven checkpoint was retained with the improved generation pipeline.
- Link the Music experience to https://chudgpt-public.vercel.app/music and the Public project to https://github.com/ASTRA228b/ChudGPT-Public.

Also fix the mobile landing-page layout so the existing **Learn about the developer** link/card is visible on phone-sized screens. It must not be hidden by desktop-only breakpoints, clipped below a fixed-height section, covered by navigation, or removed from the mobile menu. Give it a clear tap target of at least 44px, preserve the current destination, and test at 320px, 375px, 390px, and 430px viewport widths. Do not remove or rename the desktop version.

Add a concise dated changelog item for this release. Do not invent benchmark claims beyond the facts above.
