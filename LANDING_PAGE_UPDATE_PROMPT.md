# Landing-page update prompt

Update only the ChudGPT-Public card and release notes on the ChudGPT landing page.

Public v10 keeps the 20,999,184-parameter architecture and adds a balanced 12,000-conversation response-only tuning set, neural four-candidate selection, improved short-prompt coverage, and a small transparent identity/project-facts layer. Normal answers still come from the neural model; only exact ChudGPT identity and family metadata may be corrected by controlled assistance.

Use honest wording: this is an incremental experimental improvement, not a frontier model. The unseen mixed evaluation improved from 7/20 raw to 8/20 raw and reached 12/20 in production with identity assistance. Validation loss was 1.6730 after 800 CUDA/AMP tuning steps. Strict general reasoning, math, factual accuracy, coding, and long-context behavior remain weak, and important answers must be verified.

Keep the existing Public website/API and desktop-download links. Do not change cards for Buggy, Plus, Pro, Code, Ultimate, Mega, or archived checkpoints.
