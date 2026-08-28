# Landing-page update prompt

Update the ChudGPT landing page's ChudGPT-Public-Music V1 card and model guide with this accurate release note:

> Music V1 relevance and repetition update: improved topic-aware candidate selection, repaired malformed Title/Style metadata, validated lyric section order, removed a recurring style suffix, and added private log-derived suppression for overused lyric lines. Music requests now include source-aware diagnostics for the web client and Discord bot. All 295 Public tests and 83 Discord tests pass. The model is still a small experimental 21M-parameter checkpoint: structure and repetition handling improved, but grammar and long-song coherence remain imperfect.

Link the detailed report in the Public repository at `reports/MUSIC_V1_RELEVANCE_UPDATE.md`. Do not claim that a new checkpoint was deployed: a 600-step CUDA correction candidate was evaluated and rejected because its generated English was worse. Keep the existing Music V1 name, URL, and experimental/slop positioning.
