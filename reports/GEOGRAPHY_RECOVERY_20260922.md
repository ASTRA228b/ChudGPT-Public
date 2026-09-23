# Geography serving update — September 22, 2026

Public now answers explicit questions about all 50 U.S. state capitals and 123 country capitals from a local fact table. The country table also supports continent/location questions. Basic world questions cover continents, oceans, the equator, the prime meridian, mountains, deserts, country area, and U.S. state area.

The exact requested example, `What is the capital of Australia`, returns `The capital of Australia is Canberra.` Follow-ups such as `What about Canada?` work immediately after a capital question. State abbreviations, possessive wording, reverse lookups, multiple named capitals, and a full state-capitals list are supported.

Georgia is explicitly disambiguated between the U.S. state and country. South Africa, Bolivia, Switzerland, the Netherlands, Sri Lanka, and Malaysia retain their special capital/government-seat distinctions. Broader essays, historical questions, unknown places, and population questions are not replaced with unrelated static facts. This is bounded geography coverage, not a comprehensive world encyclopedia or live news system.

## Sources

- [U.S. capitals reference table](https://en.wikipedia.org/wiki/List_of_capitals_in_the_United_States)
- [Country capitals and capital-seat distinctions](https://en.wikipedia.org/wiki/List_of_national_capitals)
- [Australia's National Capital Authority: naming Canberra](https://www.nca.gov.au/education/canberras-history/siting-and-naming-canberra)
- [South African Government: three capitals](https://www.gov.za/south-africa-glance)
- [NOAA: five named oceans](https://oceanservice.noaa.gov/facts/howmanyoceans.html)
- [NOAA: largest and deepest ocean](https://oceanservice.noaa.gov/facts/biggestocean.html)

The local tables contain factual place-name mappings and independently written explanations, not copied source prose. Sources were consulted during this update; serving makes no web requests. Recheck facts when capitals or geographic conventions change.

## Delivery

The implementation is `public_geography.py`, called by `PublicModelService.chat`. The selected recovery-v5 checkpoint and its weights remain unchanged. The LGBTQIA+ and emoji recovery remains enabled. Restart the Public API process to load the updated Python modules.

Regression tests are in `tests/test_public_geography.py`; checkpoint-backed smoke results are recorded in `geography_recovery_20260922.json`.
