import pytest
from public_recipes import recipe_response, RECIPES
from test_awareness_recovery import service


@pytest.mark.parametrize('recipe', RECIPES, ids=lambda r:r['name'])
def test_library_recipe_has_ingredients_and_steps(recipe):
    reply=recipe_response('How do I make '+recipe['aliases'][0]+'?')
    assert reply.startswith('## '+recipe['name'])
    assert 'Ingredients:' in reply and 'Steps:' in reply
    assert all(step in reply for step in recipe['steps'])


def test_scaling_and_followups(service):
    session,reply=service.chat('Recipe for pancakes for 4 people.',None)
    assert '2 cups all-purpose flour' in reply
    assert '1 1/2 cups milk' in reply
    assert service.last_assistance_reason=='recipe_library'
    _,reply=service.chat('just ingredients',session)
    assert '2 cups all-purpose flour' in reply and 'Steps:' not in reply
    _,reply=service.chat('make it for 2 people',session)
    assert '1 cup all-purpose flour' in reply
    _,reply=service.chat('steps',session)
    assert 'Steps:' in reply and 'Ingredients:' not in reply


@pytest.mark.parametrize('prompt',['How do I make a Unity game?', 'What is Gorilla Tag?', 'How do I make pancakes without eggs?', 'How do I make vegan pancakes?', 'Write a pancake recipe in Python', 'How do I make sushi?', 'make rice and pancakes'])
def test_unsupported_requests_remain_neural(service,prompt):
    assert service.chat(prompt,None)[1]=='NEURAL'


def test_no_stale_recipe_followup(service):
    session,_=service.chat('How do I make rice?',None)
    service.chat('Tell me about space',session)
    assert service.chat('ingredients',session)[1]=='NEURAL'


def test_catalog_and_serving_limits():
    assert 'Pancakes' in recipe_response('What recipes do you know?')
    assert '1–20' in recipe_response('Recipe for pancakes for 0 people')
    assert '1–20' in recipe_response('Recipe for pancakes for 999 people')
