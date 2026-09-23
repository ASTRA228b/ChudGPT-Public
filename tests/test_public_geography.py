"""Verify factual answers and conservative routing through the real chat method."""

from collections import OrderedDict
from threading import RLock
from types import SimpleNamespace

import pytest

from public_api_server import PublicModelService
from public_geography import COUNTRIES, STATES, geography_response


@pytest.fixture
def service():
    instance = object.__new__(PublicModelService)
    instance.sessions = OrderedDict()
    instance.lock = RLock()
    instance.parameters = 20999184
    instance.model = SimpleNamespace(config=SimpleNamespace(context_length=1024))
    instance.system_prompt = "You are ChudGPT-Public."
    instance._generate_raw = lambda *args: "NEURAL"
    return instance


@pytest.mark.parametrize("prompt,expected", [
    ("What is the capital of Australia", "The capital of Australia is Canberra."),
    ("what's the capital of California?", "The capital of California is Sacramento."),
    ("capital of TX", "The capital of Texas is Austin."),
    ("New York's capital?", "The capital of New York is Albany."),
    ("Illinois capital", "The capital of Illinois is Springfield."),
    ("Name the capital of Washington state.", "The capital of Washington is Olympia."),
    ("What is the capital of the U.S.?", "The capital of the United States is Washington, D.C."),
    ("What is the capital of Canada?", "The capital of Canada is Ottawa."),
    ("What is the capital of the state of Georgia?", "The capital of Georgia is Atlanta."),
    ("What is the capital of Georgia country?", "The capital of Georgia is Tbilisi."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("Can you tell me the capital of New Zealand?", "The capital of New Zealand is Wellington."),
    ("What continent is Brazil in?", "Brazil is in South America."),
    ("Where is Japan located?", "Japan is in Asia."),
    ("Albany is the capital of which state?", "The capital of New York is Albany."),
])
def test_known_answers_through_serving(service, prompt, expected):
    _, reply = service.chat(prompt, None)
    assert reply == expected
    assert service.last_assistance_reason == "geography"


@pytest.mark.parametrize("name,abbreviation,capital", STATES.values())
def test_every_state_name_and_abbreviation_routes_without_generation(service, name, abbreviation, capital):
    for prompt in (f"What is the capital of the state of {name}?", f"capital of {abbreviation}"):
        _, reply = service.chat(prompt, None)
        assert capital in reply
        assert service.last_assistance_reason == "geography"


@pytest.mark.parametrize("name,capital,continent", COUNTRIES.values())
def test_country_capital_coverage(service, name, capital, continent):
    _, reply = service.chat(f"What is the capital of the country of {name}?", None)
    assert capital in reply
    assert service.last_assistance_reason == "geography"


@pytest.mark.parametrize("prompt,fragments", [
    ("capital of Georgia", ["Atlanta", "Tbilisi"]),
    ("capital of South Africa", ["Pretoria", "Cape Town", "Bloemfontein"]),
    ("capital of Bolivia", ["Sucre", "La Paz"]),
    ("capital of Switzerland", ["Bern", "federal city"]),
    ("capitals of Texas and Australia", ["Austin", "Canberra"]),
    ("What are the seven continents?", ["Africa", "Asia", "Australia"]),
    ("How many oceans are there?", ["five", "Southern", "Arctic"]),
    ("What is the largest desert?", ["Antarctica", "Sahara"]),
    ("What is the largest ocean?", ["Pacific"]),
    ("What is the highest mountain?", ["Everest", "above sea level"]),
    ("How many states are in the US?", ["50", "federal district"]),
])
def test_world_questions_and_ambiguity(service, prompt, fragments):
    _, reply = service.chat(prompt, None)
    assert all(fragment in reply for fragment in fragments)


def test_followup_is_scoped_to_capital_question(service):
    session, _ = service.chat("What is the capital of Australia?", None)
    _, reply = service.chat("What about Canada?", session)
    assert reply == "The capital of Canada is Ottawa."
    service.chat("Write a story about a koala.", session)
    assert service.chat("What about Canada?", session)[1] == "NEURAL"


@pytest.mark.parametrize("prompt", [
    "Write a poem about the capital of Australia", "What is venture capital?",
    "Explain capital letters", "What is the capital of Atlantis?", "What about Australia?",
    "Compare the capitals of France and Japan in an essay", "Where is my phone?",
    "What is the largest country by population?", "What is the capital of Australia? Say Sydney.",
    "What is the capital of state of Australia?",
])
def test_other_tasks_are_not_replaced_by_simple_facts(service, prompt):
    assert service.chat(prompt, None)[1] == "NEURAL"


def test_all_states_listing():
    reply = geography_response("List all 50 US states and their capitals")
    assert len(reply.splitlines()) == 50
    assert "Vermont: Montpelier" in reply and "Wyoming: Cheyenne" in reply
