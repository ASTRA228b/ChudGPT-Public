"""Small structured cooking library. Unsupported requests keep their normal route."""
import json
import re
from fractions import Fraction
from pathlib import Path

RECIPES = json.loads((Path(__file__).resolve().parent / 'data/recipes.json').read_text(encoding='utf-8'))
ALIASES = {alias: recipe for recipe in RECIPES for alias in recipe['aliases']}


def amount(value):
    parts = value.split()
    return sum((Fraction(p) for p in parts), Fraction())


def quantity(value):
    whole, remainder = divmod(value.numerator, value.denominator)
    return (str(whole) if whole else '') + ((' ' if whole else '') + f'{remainder}/{value.denominator}' if remainder else '')


def render(recipe, servings, section='full'):
    factor = Fraction(servings, recipe['servings'])
    lines = [f"## {recipe['name']}", f"Servings: {servings} · About {recipe['minutes']} minutes"]
    if section != 'steps':
        lines += ['', 'Ingredients:']
        for value, unit, ingredient in recipe['ingredients']:
            scaled = amount(value)*factor if value else None
            for singular in ['cup', 'clove', 'slice']:
                if unit in {singular,singular+'s'}:
                    unit = singular if scaled == 1 else singular+'s'
            if ingredient in {'egg','eggs'}:
                ingredient = 'egg' if scaled == 1 else 'eggs'
            lines.append('- ' + ' '.join(x for x in [quantity(scaled) if scaled is not None else '', unit, ingredient] if x))
    if section != 'ingredients':
        lines += ['', 'Steps:'] + [f'{i}. {step}' for i,step in enumerate(recipe['steps'],1)]
    if servings != recipe['servings']:
        lines += ['', 'Ingredient quantities are scaled; use extra pans or cook in batches as needed. Cooking times are approximate.']
    return '\n'.join(lines)


def recipe_response(message, history=()):
    text = re.sub(r'\s+', ' ', message.casefold().strip()).rstrip('?.!')
    if text in {'what recipes do you know','list recipes','show recipes','recipe library','what can i cook','what can i make for dinner'}:
        return 'Recipe library: ' + ', '.join(r['name'] for r in RECIPES) + '.\nAsk “How do I make pancakes?” or “Recipe for pancakes for 4 people.” You can then ask for just the ingredients or steps.'
    # Exact follow-ups reference only the latest assistant recipe, never stale topics.
    followup = re.fullmatch(r'(?:just (?:the )?)?(ingredients|steps)(?: please)?|(?:what ingredients do i need|how do i cook it)|(?:make (?:it|that)|scale (?:it|that)) for (\d+) (?:people|servings)', text)
    if followup:
        previous = next((t['content'] for t in reversed(history) if t.get('role')=='assistant'), '')
        header = re.match(r'## (.+)\nServings: (\d+)', previous)
        recipe = next((r for r in RECIPES if header and r['name']==header[1]), None)
        if recipe:
            count = int(followup[2] or header[2])
            if not 1 <= count <= 20:
                return 'The recipe library supports 1–20 servings per request. How many servings would you like?'
            section = followup[1] or ('ingredients' if text=='what ingredients do i need' else 'steps' if text=='how do i cook it' else 'full')
            return render(recipe,count,section)
    match = re.fullmatch(r'(?:please )?(?:how (?:do i|do you|can i|to) (?:make|cook|prepare)|(?:give|show) me (?:a |the )?recipe for|(?:a )?recipe for|make|cook) (.+?)(?: for (\d+) (?:people|servings))?(?: please)?',text)
    if not match:
        return None
    recipe = ALIASES.get(match[1])
    if not recipe:
        return None
    servings = int(match[2] or recipe['servings'])
    if not 1 <= servings <= 20:
        return 'The recipe library supports 1–20 servings per request. How many servings would you like?'
    return render(recipe,servings)
