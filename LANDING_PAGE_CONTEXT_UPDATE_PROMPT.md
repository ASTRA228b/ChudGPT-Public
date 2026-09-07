Update the ChudGPT landing page and model guide with this verified Public V20 maintenance update. Keep the existing design and mobile support.

Describe these changes accurately:
- Improved context handling for oversized requests. The model now retains the beginning and ending of the newest request when the full text cannot fit, instead of silently dropping the request. Older conversation turns still expire when the context fills.
- Improved supervised training preparation: reserve space for the complete answer, including its end token. Examples whose answer cannot fit are excluded rather than teaching truncated answers.
- These are context and training-pipeline improvements, not a newly trained checkpoint or a demonstrated general-intelligence increase.
- Public V20 is still the same experimental 20,999,184-parameter model. Its generated replies can be funny, awkward, or incorrect. Existing math, identity, and greeting systems remain unchanged. No new fallback answers were added.

Keep all existing API URLs, model IDs, download links, and Discord commands unless separately verified changes require an update. Do not invent releases, benchmark scores, larger context windows, unlimited memory, or new model capabilities.

Replace superseded update cards and old duplicate update logs with a concise current entry. Preserve required policy history and current documentation. Ensure the update and Learn about the developer link are visible on mobile as well as desktop. Check navigation and layout after editing.
