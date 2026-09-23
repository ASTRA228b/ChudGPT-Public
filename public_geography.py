"""Local geography facts with narrow intent matching; no network at serving time.

Sources and scope are recorded in reports/GEOGRAPHY_RECOVERY_20260922.md.
These facts assist the selected checkpoint; they do not change its weights.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from chudlm.emoji_awareness import strip_emoji_context


# State, USPS abbreviation, capital. Keep all 50 states in alphabetical order.
STATE_ROWS = """Alabama|AL|Montgomery
Alaska|AK|Juneau
Arizona|AZ|Phoenix
Arkansas|AR|Little Rock
California|CA|Sacramento
Colorado|CO|Denver
Connecticut|CT|Hartford
Delaware|DE|Dover
Florida|FL|Tallahassee
Georgia|GA|Atlanta
Hawaii|HI|Honolulu
Idaho|ID|Boise
Illinois|IL|Springfield
Indiana|IN|Indianapolis
Iowa|IA|Des Moines
Kansas|KS|Topeka
Kentucky|KY|Frankfort
Louisiana|LA|Baton Rouge
Maine|ME|Augusta
Maryland|MD|Annapolis
Massachusetts|MA|Boston
Michigan|MI|Lansing
Minnesota|MN|Saint Paul
Mississippi|MS|Jackson
Missouri|MO|Jefferson City
Montana|MT|Helena
Nebraska|NE|Lincoln
Nevada|NV|Carson City
New Hampshire|NH|Concord
New Jersey|NJ|Trenton
New Mexico|NM|Santa Fe
New York|NY|Albany
North Carolina|NC|Raleigh
North Dakota|ND|Bismarck
Ohio|OH|Columbus
Oklahoma|OK|Oklahoma City
Oregon|OR|Salem
Pennsylvania|PA|Harrisburg
Rhode Island|RI|Providence
South Carolina|SC|Columbia
South Dakota|SD|Pierre
Tennessee|TN|Nashville
Texas|TX|Austin
Utah|UT|Salt Lake City
Vermont|VT|Montpelier
Virginia|VA|Richmond
Washington|WA|Olympia
West Virginia|WV|Charleston
Wisconsin|WI|Madison
Wyoming|WY|Cheyenne"""
STATES = {name.casefold(): (name, abbreviation, capital)
          for name, abbreviation, capital in (row.split("|") for row in STATE_ROWS.splitlines())}
STATE_ALIASES = {abbreviation.casefold(): key for key, (_, abbreviation, _) in STATES.items()}

# Country, capital, continent/region. Special capital arrangements are below.
COUNTRY_ROWS = """United States|Washington, D.C.|North America
Canada|Ottawa|North America
Mexico|Mexico City|North America
Guatemala|Guatemala City|North America
Belize|Belmopan|North America
Honduras|Tegucigalpa|North America
El Salvador|San Salvador|North America
Nicaragua|Managua|North America
Costa Rica|San José|North America
Panama|Panama City|North America
Cuba|Havana|North America
Jamaica|Kingston|North America
Haiti|Port-au-Prince|North America
Dominican Republic|Santo Domingo|North America
Bahamas|Nassau|North America
Barbados|Bridgetown|North America
Trinidad and Tobago|Port of Spain|North America
Brazil|Brasília|South America
Argentina|Buenos Aires|South America
Chile|Santiago|South America
Peru|Lima|South America
Colombia|Bogotá|South America
Venezuela|Caracas|South America
Ecuador|Quito|South America
Bolivia|Sucre|South America
Paraguay|Asunción|South America
Uruguay|Montevideo|South America
Guyana|Georgetown|South America
Suriname|Paramaribo|South America
United Kingdom|London|Europe
Ireland|Dublin|Europe
France|Paris|Europe
Germany|Berlin|Europe
Spain|Madrid|Europe
Portugal|Lisbon|Europe
Italy|Rome|Europe
Netherlands|Amsterdam|Europe
Belgium|Brussels|Europe
Luxembourg|Luxembourg|Europe
Switzerland|Bern|Europe
Austria|Vienna|Europe
Denmark|Copenhagen|Europe
Norway|Oslo|Europe
Sweden|Stockholm|Europe
Finland|Helsinki|Europe
Iceland|Reykjavík|Europe
Poland|Warsaw|Europe
Czechia|Prague|Europe
Slovakia|Bratislava|Europe
Hungary|Budapest|Europe
Romania|Bucharest|Europe
Bulgaria|Sofia|Europe
Greece|Athens|Europe
Croatia|Zagreb|Europe
Slovenia|Ljubljana|Europe
Serbia|Belgrade|Europe
Albania|Tirana|Europe
Estonia|Tallinn|Europe
Latvia|Riga|Europe
Lithuania|Vilnius|Europe
Ukraine|Kyiv|Europe
Russia|Moscow|Europe and Asia
Turkey|Ankara|Europe and Asia
Georgia|Tbilisi|the Caucasus, at the boundary of Europe and Asia
China|Beijing|Asia
Japan|Tokyo|Asia
South Korea|Seoul|Asia
North Korea|Pyongyang|Asia
India|New Delhi|Asia
Pakistan|Islamabad|Asia
Bangladesh|Dhaka|Asia
Nepal|Kathmandu|Asia
Bhutan|Thimphu|Asia
Sri Lanka|Sri Jayawardenepura Kotte|Asia
Thailand|Bangkok|Asia
Vietnam|Hanoi|Asia
Cambodia|Phnom Penh|Asia
Laos|Vientiane|Asia
Malaysia|Kuala Lumpur|Asia
Singapore|Singapore|Asia
Philippines|Manila|Asia
Mongolia|Ulaanbaatar|Asia
Kazakhstan|Astana|Asia and Europe
Uzbekistan|Tashkent|Asia
Iran|Tehran|Asia
Iraq|Baghdad|Asia
Jordan|Amman|Asia
Lebanon|Beirut|Asia
Saudi Arabia|Riyadh|Asia
United Arab Emirates|Abu Dhabi|Asia
Qatar|Doha|Asia
Oman|Muscat|Asia
Kuwait|Kuwait City|Asia
Bahrain|Manama|Asia
Egypt|Cairo|Africa and Asia (the Sinai Peninsula)
Morocco|Rabat|Africa
Algeria|Algiers|Africa
Tunisia|Tunis|Africa
Nigeria|Abuja|Africa
Ghana|Accra|Africa
Senegal|Dakar|Africa
Kenya|Nairobi|Africa
Ethiopia|Addis Ababa|Africa
Uganda|Kampala|Africa
Rwanda|Kigali|Africa
Tanzania|Dodoma|Africa
South Africa|Pretoria|Africa
Botswana|Gaborone|Africa
Namibia|Windhoek|Africa
Zambia|Lusaka|Africa
Zimbabwe|Harare|Africa
Mozambique|Maputo|Africa
Madagascar|Antananarivo|Africa
Angola|Luanda|Africa
Cameroon|Yaoundé|Africa
Australia|Canberra|Oceania
New Zealand|Wellington|Oceania
Fiji|Suva|Oceania
Papua New Guinea|Port Moresby|Oceania
Samoa|Apia|Oceania
Tonga|Nukuʻalofa|Oceania
Solomon Islands|Honiara|Oceania
Vanuatu|Port Vila|Oceania"""
COUNTRIES = {name.casefold(): (name, capital, continent)
             for name, capital, continent in (row.split("|") for row in COUNTRY_ROWS.splitlines())}
COUNTRY_ALIASES = {
    "us": "united states", "usa": "united states", "u.s.": "united states", "u.s": "united states",
    "u.s.a.": "united states", "united states of america": "united states",
    "uk": "united kingdom", "u.k.": "united kingdom", "u.k": "united kingdom", "britain": "united kingdom",
    "uae": "united arab emirates", "czech republic": "czechia", "türkiye": "turkey",
}
SPECIAL_CAPITALS = {
    "south africa": "South Africa has three capitals: Pretoria (administrative), Cape Town (legislative), and Bloemfontein (judicial).",
    "bolivia": "Sucre is Bolivia's constitutional capital; La Paz is the seat of government.",
    "netherlands": "Amsterdam is the capital of the Netherlands; The Hague is the seat of government.",
    "switzerland": "Bern is Switzerland's federal city and serves as its de facto capital.",
    "sri lanka": "Sri Jayawardenepura Kotte is Sri Lanka's legislative capital; Colombo remains an executive and judicial center.",
    "malaysia": "Kuala Lumpur is Malaysia's capital; Putrajaya is its administrative center.",
}

FACTS = [
    (r"(?:how many (?:us |u\.s\. |american )?states are there|how many states (?:are in|does) (?:the )?(?:us|usa|united states)(?: have)?)",
     "The United States has 50 states. Washington, D.C. is a federal district, not a state."),
    (r"(?:what are|name|list|how many) (?:the )?(?:seven |7 )?continents(?: (?:are there|on earth))?",
     "The common seven-continent model is Africa, Antarctica, Asia, Europe, North America, South America, and Australia. Other conventions group continents differently; Oceania is a broader region including Australia."),
    (r"(?:what are|name|list|how many) (?:the )?(?:five |5 )?oceans(?: (?:are there|on earth))?",
     "The five commonly recognized oceans are the Pacific, Atlantic, Indian, Southern, and Arctic. They are connected parts of one global ocean."),
    (r"(?:what|which) (?:is (?:the )?)?(?:largest|biggest|deepest) ocean(?: (?:is largest|on earth|in the world))?",
     "The Pacific Ocean is the largest and deepest ocean on Earth."),
    (r"(?:what|which) (?:is (?:the )?)?(?:smallest) ocean(?: (?:on earth|in the world))?",
     "The Arctic Ocean is the smallest ocean."),
    (r"(?:what|which) is (?:the )?(?:largest|biggest) continent(?: (?:on earth|in the world))?",
     "Asia is the largest continent by land area and population."),
    (r"(?:what|which) is (?:the )?(?:largest|biggest) country(?: (?:by (?:land )?area|in the world))?",
     "Russia is the largest country by total area."),
    (r"(?:what|which) is (?:the )?smallest country(?: (?:by area|in the world))?",
     "Vatican City is the smallest sovereign state by area."),
    (r"(?:what|which) is (?:the )?(?:largest|biggest) (?:us |u\.s\. )?state(?: by area)?",
     "Alaska is the largest U.S. state by area."),
    (r"(?:what|which) is (?:the )?smallest (?:us |u\.s\. )?state(?: by area)?",
     "Rhode Island is the smallest U.S. state by area."),
    (r"(?:what|which) is (?:the )?(?:tallest|highest) mountain(?: (?:on earth|in the world))?",
     "Mount Everest is the highest mountain above sea level. It lies in the Himalayas on the Nepal-China border. Tallest measured from base to summit is a different comparison."),
    (r"(?:what|which) is (?:the )?(?:largest|biggest) desert(?: (?:on earth|in the world))?",
     "Antarctica is the largest desert overall because deserts receive very little precipitation. The Sahara is the largest hot desert."),
    (r"(?:what|which) is (?:the )?(?:largest|biggest) hot desert(?: (?:on earth|in the world))?",
     "The Sahara in North Africa is the largest hot desert."),
    (r"(?:what is|what's|explain) (?:the )?equator",
     "The equator is the imaginary line at 0 degrees latitude, dividing Earth into the Northern and Southern Hemispheres."),
    (r"(?:what is|what's|explain) (?:the )?prime meridian",
     "The prime meridian is the reference line at 0 degrees longitude, conventionally associated with Greenwich, London."),
    (r"(?:is antarctica a country|what country owns antarctica)",
     "Antarctica is a continent, not a country. Several countries claim parts of it, and the Antarctic Treaty governs international cooperation there."),
]


def _clean(message: str) -> str:
    return re.sub(r"\s+", " ", strip_emoji_context(message).casefold().replace("’", "'")).strip().rstrip("?!.")


def _place(value: str) -> tuple[str, str | None]:
    value = value.strip()
    kind = None
    prefix = re.match(r"(?:the )?(?:(us|u\.s\.) )?(state|country) of (.+)", value)
    if prefix:
        kind, value = prefix[2], prefix[3]
    suffix = re.fullmatch(r"(.+?)(?: (state|country)| \((state|country)\))", value)
    if suffix:
        value, kind = suffix[1], suffix[2] or suffix[3]
    value = re.sub(r"^the ", "", value)
    if kind != "country":
        value = STATE_ALIASES.get(value, value)
    if kind != "state":
        value = COUNTRY_ALIASES.get(value, value)
    return value, kind


def _capital(place: str) -> str | None:
    key, kind = _place(place)
    if key == "georgia" and kind is None:
        return "Atlanta is the capital of the U.S. state of Georgia; Tbilisi is the capital of the country Georgia."
    if key in STATES and kind != "country":
        name, _, capital = STATES[key]
        return f"The capital of {name} is {capital}."
    if key in COUNTRIES and kind != "state":
        name, capital, _ = COUNTRIES[key]
        display_name = f"the {name}" if key in {"united states", "united kingdom", "united arab emirates", "bahamas", "philippines", "dominican republic"} else name
        return SPECIAL_CAPITALS.get(key, f"The capital of {display_name} is {capital.rstrip('.')}.")
    return None


def _capital_target(text: str) -> str | None:
    match = re.fullmatch(
        r"(?:(?:what(?:'s|s| is)|name|tell me|can you tell me) (?:the )?)?(?:state )?capitals? (?:of|for) (.+)|"
        r"(?:what(?:'s| is) )?(.+?)'s (?:state )?capital|(.+?) (?:state )?capital", text,
    )
    return next((group for group in match.groups() if group), None) if match else None


def geography_response(message: str, history: Sequence[Mapping[str, str]] = ()) -> str | None:
    """Answer explicit supported geography questions; leave other tasks neural."""
    text = _clean(message)
    if re.fullmatch(r"(?:list|name|show(?: me)?) (?:all )?(?:50 )?(?:us |u\.s\. )?states and (?:their )?capitals", text):
        return "\n".join(f"{name}: {capital}" for name, _, capital in STATES.values())
    target = _capital_target(text)
    if target:
        reply = _capital(target)
        if reply:
            return reply
        parts = re.split(r",\s*|\s+and\s+", target)
        replies = [_capital(part) for part in parts]
        if len(parts) > 1 and all(replies):
            return "\n".join(replies)
        return None
    # Follow-ups inherit only a directly preceding capital question, never an
    # arbitrary location mentioned in older or unrelated conversation.
    followup = re.fullmatch(r"(?:what about|and) (.+)", text)
    if followup:
        previous = next((turn.get("content", "") for turn in reversed(history) if turn.get("role") == "user"), "")
        if _capital_target(_clean(previous)):
            return _capital(followup[1])
        return None
    location = re.fullmatch(r"(?:what|which) continent is (.+?) (?:in|on)|where is (.+?)(?: located)?", text)
    if location:
        key, kind = _place(location[1] or location[2])
        if key in COUNTRIES and kind != "state":
            name, _, continent = COUNTRIES[key]
            return f"{name} is in {continent}."
        return None
    reverse = re.fullmatch(r"(.+?) is (?:the )?capital of (?:what|which) (state|country)", text)
    if reverse:
        city, kind = reverse.groups()
        records = STATES.values() if kind == "state" else COUNTRIES.values()
        for row in records:
            capital = row[2] if kind == "state" else row[1]
            if city == capital.casefold():
                return _capital(f"{kind} of {row[0].casefold()}")
        return None
    for pattern, answer in FACTS:
        if re.fullmatch(pattern, text):
            return answer
    return None
