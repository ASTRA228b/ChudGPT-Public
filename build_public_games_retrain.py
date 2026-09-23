"""Build reviewed game/modding SFT plus filtered replay. Never ingest raw logs.

Run from ChudGPT-Public. Sources and human review notes are in
reports/GAMES_RETRAIN_SOURCES.md. Filters apply to training data only.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from project_facts import FAMILY_FACTS, FAMILY_SUMMARY
from public_geography import STATES, COUNTRIES, geography_response
from public_lgbtq import DEFINITIONS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data"

GAME_PAIRS = [
    ("What is Gorilla Tag?", "Gorilla Tag is a multiplayer VR game by Another Axiom. You move by pushing off surfaces with your hands and arms, climb around maps, and play tag or infection with other gorillas. It is also a social hangout: part parkour, part monkey chaos."),
    ("Tell me about Gorilla Tag.", "Gorilla Tag is a social virtual-reality game built around arm-based movement. Players run, jump, climb, chase each other, and hang out in rooms. Another Axiom develops it. The movement looks simple until a monkey disappears up a tree."),
    ("What is GTAG?", "GTAG is a common abbreviation for Gorilla Tag, Another Axiom's multiplayer VR game. Players use their hands and arms to move, climb, and play tag."),
    ("Explain Gorilla Tag in one sentence.", "Gorilla Tag is Another Axiom's multiplayer VR game where arm-powered gorillas climb, chase, and hang out together."),
    ("Who makes Gorilla Tag?", "Another Axiom develops Gorilla Tag. Lemming, also known as Kerestell Smith, created the original game."),
    ("How do you move in Gorilla Tag?", "You push against the ground and other surfaces with your tracked hands. Arm movements propel you forward or upward, so timing, contact, and direction matter more than a movement stick."),
    ("What makes Gorilla Tag fun?", "Its movement turns a simple game of tag into physical parkour. Learning routes, juking friends, and barely escaping a chase can be hilarious. The social rooms are a big part of the appeal too."),
    ("What is infection in Gorilla Tag?", "In infection, tagged players join the infected team and help catch the remaining players. The chase gets harder as more gorillas become infected."),
    ("What does branching mean in Gorilla Tag?", "Branching means moving through trees by jumping between branches. Good branching combines controlled pushes, accurate landings, and planning the next branch before you reach it."),
    ("How do I practice Gorilla Tag movement?", "Start with short, controlled pushes on flat ground, then practice landing on nearby surfaces. Repeat a small route until it feels consistent before trying faster jumps. Leave real space around your arms and controllers."),
    ("Name some Gorilla Tag maps.", "Forest, Canyon, Mountain, and Beach are examples of Gorilla Tag environments. Maps and activities can change with updates, so that is a sample rather than a complete current list."),
    ("What are Gorilla Tag cosmetics?", "Cosmetics are wearable items that change how your gorilla looks. They are a way to express style; a fancy hat does not teach you how to land a jump."),
    ("Is Gorilla Tag just competitive?", "No. Alongside chasing and movement practice, people use Gorilla Tag as a social hangout. Friends can talk, explore, make up challenges, and enjoy the ridiculous gorilla movement."),
    ("Is Gorilla Tag a normal flat-screen game?", "Gorilla Tag is designed for VR with tracked controllers. Its defining movement depends on moving your hands and arms in virtual space."),
    ("What is the Gorilla Tag modding community?", "The Gorilla Tag modding community is a loose network of players, programmers, artists, map makers, and testers. People discuss custom content, C# plugins, tools, compatibility, and debugging. Community projects are not automatically official Another Axiom releases."),
    ("Talk about Gorilla Tag modders.", "Gorilla Tag modders build and test community projects, share code, write documentation, and help each other debug. Some focus on maps or visual content; others experiment with tools and plugins. It is a creative community, not just a collection of cheat menus."),
    ("What do Gorilla Tag modding communities talk about?", "Common topics include custom maps, Unity, C#, plugin loaders, game updates, bug reports, and compatibility. Useful help usually starts with the game version, platform, mod version, and an error log."),
    ("Are all Gorilla Tag mods official?", "No. A community mod is not automatically made, reviewed, or endorsed by Another Axiom. Official tools and approved content channels are different from unofficial PC plugins."),
    ("Are all mods cheats?", "No. A mod can add maps, visual changes, accessibility features, or development tools. Whether a particular mod is permitted depends on the game and how it affects other players."),
    ("Are Gorilla Tag mods always allowed in private rooms?", "A private room is not blanket permission for every mod. Check Gorilla Tag's current Fan Content and Mod Policy and the rules for the feature you plan to use."),
    ("What should a Gorilla Tag beginner learn before modding?", "Learn basic C#, Unity components, project references, and how to read logs. Start with a small permitted project, such as logging that your plugin loaded, before attempting a complicated feature."),
    ("Does every PC Gorilla Tag mod work on standalone Quest?", "No. PC VR and standalone Quest have different runtimes and installation requirements. Follow the project's platform and version documentation instead of assuming one build works everywhere."),
    ("Why did my Gorilla Tag mod stop working after an update?", "An update can change game methods, assets, dependencies, or loader compatibility. Check the mod's supported version and read the first meaningful error in the log before changing unrelated files."),
    ("How do I ask a modding community for help?", "Describe what you expected and what happened. Include your platform, game and loader versions, the mod version, reproduction steps, and the relevant error log. Remove account tokens and personal details before sharing logs."),
    ("How can I contribute without being a programmer?", "You can test releases, report reproducible bugs, improve documentation, translate instructions, or create original art and maps. Clear feedback is valuable even if you never write a line of code."),
    ("How should I report a broken community mod?", "Open the project's issue tracker if it has one. Give exact versions, minimal reproduction steps, the expected result, and a trimmed log. A report saying 'it exploded lol' is funny, but a stack trace is more useful."),
    ("Why do mod authors care about credits?", "Credits recognize the people who wrote code, made assets, tested features, or documented tools. Preserve required licenses and attribution when you build on someone else's work."),
    ("Can a Discord role prove somebody made a mod?", "A role alone does not prove authorship. Check the original repository, release history, and project credits rather than guessing from a display name or server role."),
    ("What is BepInEx?", "BepInEx is a plugin framework used for modding Unity and other .NET games. It loads plugins and provides facilities such as logging and configuration. The correct build depends on the game's runtime and platform."),
    ("What is a BepInEx plugin?", "A BepInEx plugin is compiled code loaded by BepInEx to add or change behavior. In a Unity Mono project it commonly uses C# and a plugin class with metadata identifying the plugin."),
    ("What is Harmony in modding?", "Harmony is a .NET library for patching methods at runtime. Prefixes run before an original method and postfixes run afterward. A patch changes behavior; Harmony is not itself a game or a complete mod loader."),
    ("How are BepInEx and Harmony different?", "BepInEx loads plugins and supplies runtime services. Harmony lets a plugin patch existing methods. They solve different problems and are often used together."),
    ("What is a Harmony prefix?", "A Harmony prefix runs before the patched method. Depending on its signature and behavior, it can inspect arguments or affect whether the original method runs."),
    ("What is a Harmony postfix?", "A Harmony postfix runs after the original method and can inspect or adjust its result. Keep patches small and compatible with other plugins."),
    ("What is a mod loader?", "A mod loader discovers and initializes mods so a game can use them. It may also manage dependencies and version compatibility. It is separate from the individual mods it loads."),
    ("What is a mod manager?", "A mod manager helps organize, install, enable, or update mods. A loader runs the mods inside the game; a manager handles their installation and profiles."),
    ("What is a DLL in Unity modding?", "A DLL is a compiled library. A C# plugin is often built as a DLL that a compatible loader loads. The file extension alone does not guarantee compatibility or safety."),
    ("Why does the mod say missing dependency?", "The mod needs another package or assembly that is absent or incompatible. Check the dependency name and required version in the log, then compare them with the project's installation instructions."),
    ("Why does a plugin get a MissingMethodException?", "It tried to call a method that is missing from the loaded assembly. A common cause is compiling against a different game or dependency version than the one installed."),
    ("What does NullReferenceException mean in a Unity mod?", "Your code used a reference that was null. Read the stack trace, find the relevant line, and check whether the object was assigned, found, or created before you used it."),
    ("How do I debug two conflicting mods?", "Reproduce the issue with a clean profile, then add mods back in small groups. Compare logs and versions until you isolate the conflicting pair. Change one variable at a time."),
    ("What is a minimal reproduction?", "A minimal reproduction is the smallest set of steps or files that still produces the bug. It removes unrelated code and mods so the cause is easier to see."),
    ("What does load order mean?", "Load order is the sequence in which mods or plugins initialize. Dependencies and overlapping patches can make order matter, but changing order cannot fix every incompatibility."),
    ("What is the difference between Mono and IL2CPP?", "Mono runs managed .NET code, while Unity's IL2CPP pipeline converts managed code into native code ahead of time. Mod loaders and plugins must match the game's scripting backend."),
    ("What is a client-side mod?", "A client-side mod runs on the player's game client. That does not guarantee it is purely visual or permitted in multiplayer; its effects and the server's rules still matter."),
    ("What is a server-side mod?", "A server-side mod runs on the game server and changes behavior there. Whether players need a matching client mod depends on the game and the feature."),
    ("Why back up a save before modding?", "Some mods change save data or introduce dependencies the unmodified game cannot read. A backup gives you a way to recover your progress if the experiment goes badly."),
    ("Why should I read a mod's README?", "The README usually explains supported versions, installation, dependencies, configuration, and known problems. Reading it first can save you from debugging a completely wrong build."),
    ("What is Unity?", "Unity is a game engine with an editor, a component system, rendering, physics, and scripting tools. Many Unity projects use C# for gameplay scripts."),
    ("What is a Unity component?", "A component adds behavior or data to a GameObject. A Transform stores position, rotation, and scale; scripts, colliders, and renderers are other kinds of components."),
    ("What is a prefab?", "A Unity prefab is a reusable saved GameObject configuration, including its components and children. You can create instances instead of rebuilding the same object by hand."),
    ("What is the difference between Update and FixedUpdate?", "Update runs once per rendered frame. FixedUpdate runs on the physics timestep, so physics-related work usually belongs there. Their rates are not necessarily the same."),
    ("Should a mod allocate new objects every frame?", "Avoid unnecessary per-frame allocations because they create garbage-collection pressure. Cache references, reuse buffers when practical, and measure performance before making bigger changes."),
    ("What is a stack trace?", "A stack trace lists the chain of function calls leading to an error. Start with the first relevant line in your code rather than treating the entire log as one giant mystery."),
    ("What is a game engine?", "A game engine provides systems such as rendering, input, audio, physics, and scene management. Developers build their game's rules and content on top of those systems."),
    ("What is Minecraft?", "Minecraft is a sandbox game about building, exploring, gathering resources, and surviving in a block-based world. Creative mode emphasizes building, while Survival adds resource and health management."),
    ("What is Minecraft modding?", "Minecraft modding changes or extends the game with content and behavior. Java Edition commonly uses loaders such as Fabric or NeoForge; Bedrock uses a different add-on system."),
    ("What is Fabric?", "Fabric is a modding toolchain and loader ecosystem for Minecraft Java Edition. Fabric Loader loads mods; Fabric API provides shared hooks many mods depend on."),
    ("Are Fabric and Forge mods interchangeable?", "Usually not. A mod must target the installed loader and Minecraft version, unless its author explicitly provides compatible builds. Renaming a jar does not convert it."),
    ("What is a Minecraft resource pack?", "A resource pack changes assets such as textures and sounds. It is different from a code mod, which can add or change game behavior."),
    ("What is Roblox?", "Roblox is a platform where people play and create experiences made by other users. Creators build with Roblox Studio and commonly script behavior using Luau."),
    ("Is making a Roblox game the same as modding it?", "Building your own Roblox experience in Studio is development. Altering somebody else's running experience with an exploit tool is a different activity. Work in projects and systems you control."),
    ("What is Terraria?", "Terraria is a two-dimensional sandbox adventure game with exploration, building, crafting, combat, and progression. Digging a tunnel often turns into an accidental expedition."),
    ("What is tModLoader?", "tModLoader is a community modding platform for Terraria. It provides a way to create, load, and manage Terraria mods separately from the unmodified game."),
    ("What is Beat Saber?", "Beat Saber is a VR rhythm game where you cut approaching blocks in time with music, following their directions while avoiding obstacles."),
    ("What is Skyrim modding?", "Skyrim modding changes or expands the game through plugins, scripts, and assets. Compatibility depends on the game edition, versions, dependencies, and interactions among installed mods."),
    ("What is a modpack?", "A modpack is a curated collection of mods and often configuration files meant to work together. It can create a shared theme or make setup easier, but versions still need to match."),
    ("What is mod.io?", "mod.io is a service games can integrate for discovering and distributing user-created content. What it supports depends on the individual game's integration."),
    ("What is an open-source mod?", "An open-source mod makes its source available under a license that grants specific rights. Read that license before copying, modifying, or redistributing the code."),
    ("How can I keep a mod release maintainable?", "Keep the scope small, document dependencies, use version control, and include reproduction steps for known issues. Test against the supported game version before publishing a build."),
    ("My mod worked yesterday and now crashes.", "First check what changed: the game, loader, dependencies, configuration, or the mod itself. Compare the first error in the new log with the last working setup."),
    ("I fixed my first mod bug!", "Nice! The log monster has been defeated. What was the actual cause?"),
    ("Roast my terrible Gorilla Tag movement.", "You branch like the tree filed a restraining order. One clean landing and we'll call it a redemption arc."),
    ("Make a joke about debugging mods.", "I fixed one mod bug and unlocked three bonus bugs. Apparently the project has downloadable content."),
    ("My Unity project has 40 errors.", "The console has formed a choir. Start with the first error; a lot of the others may just be singing backup."),
    ("Is Gorilla Tag secretly a spreadsheet simulator?", "Only if your spreadsheet screams and climbs a tree. Gorilla Tag is a VR movement and social game, not an accounting package."),
    ("Give me a silly mod idea for my own game.", "A dramatic announcer that celebrates every opened door like you just won a world championship. Even the closet gets a victory fanfare."),
]


def row(category, prompt, answer):
    return {"source": "reviewed-games-20260923", "category": category,
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]}


def rejection_reason(item):
    messages = item.get("messages")
    if not isinstance(messages, list) or len(messages) < 2 or len(messages) % 2:
        return "invalid-conversation"
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != ("user" if index % 2 == 0 else "assistant"):
            return "role-order"
        value = message.get("content")
        if not isinstance(value, str) or not value.strip():
            return "empty-content"
        if re.search(r"<\|(?:user|assistant|system)|<(?:user|assistant|system)>|\[emoji context:|\ufffd", value):
            return "encoding-or-prompt-leak"
        if re.search(r"<@!?\d{5,}>|\b(?:user_id|guild_id|channel_id|access_token|api_key)\s*[:=]", value, re.I):
            return "private-log-fields"
        if message["role"] != "assistant":
            continue
        if value.count("```") % 2:
            return "unclosed-code"
        if re.search(r"(?:sorry[, ]+)?i (?:do not|don't) understand|i'm not sure what you mean|could you type a little more", value, re.I):
            return "generic-nonanswer"
        if re.search(r"\[your name\]|as an openai|i am chatgpt|i'm chatgpt|been there\.", value, re.I):
            return "wrong-assistant-or-personal-experience"
        if "```" not in value:
            words = re.findall(r"\w+", value.lower())
            grams = Counter(tuple(words[i:i+5]) for i in range(len(words)-4))
            if grams and max(grams.values()) >= 3:
                return "repeated-prose"
        if len(value) > 1800:
            return "overlong-for-finishing-curriculum"
    return None


def main():
    focus = [row("games_modding", q, a) for q, a in GAME_PAIRS]
    # Natural prompt variants teach binding without filling answers with boilerplate.
    for q, a in GAME_PAIRS:
        focus.append(row("games_modding", "Hey ChudGPT, " + q[0].lower() + q[1:], a))
        focus.append(row("games_modding", q.rstrip("?.!") + " please", a))
    family = [
        ("What is the ChudGPT model family?", FAMILY_SUMMARY),
        ("List the ChudGPT models.", FAMILY_SUMMARY),
        ("Tell me what you are.", "I'm ChudGPT-Public V20, Astra's small experimental language model. I generate text with a 20,999,184-parameter decoder-only transformer. I can be useful, funny, weird, and wrong."),
        ("Who built you?", "Astra created ChudGPT and its model family. I'm ChudGPT-Public V20."),
        ("Are you ChatGPT?", "No. I'm ChudGPT-Public, Astra's custom experimental language model. I am not an OpenAI model."),
        ("How big is your model?", "ChudGPT-Public has 20,999,184 parameters, nine transformer layers, an 8,192-token vocabulary, and a 1,024-token model context window."),
        ("Can you still make mistakes?", "Absolutely. I'm a small experimental model, so I can misunderstand things, invent details, or produce nonsense. The occasional monkey-brained answer is still possible."),
        ("Do you have to be serious all the time?", "Nope. I can joke around, be playful, and help with silly ideas. Useful answers and a little chaos can coexist."),
        ("Do you learn permanently from this chat?", "I can use retained messages in this session, but chatting does not update my neural weights. Permanent model changes require a separate training run."),
        ("Do you browse the internet yourself?", "The base model does not browse. The Discord bot can provide limited page or link context through a separate feature."),
        ("What is the difference between your weights and your app?", "My weights are learned numbers in the neural network. The surrounding app manages sessions, tools, and some exact responses. Changing app code is different from retraining the weights."),
    ]
    family.extend((f"What is ChudGPT {name}?", answer) for name, answer in FAMILY_FACTS.items())
    for q,a in family:
        for prefix in ("", "Please explain: ", "Hey, "):
            focus.append(row("model_family",prefix+q,a))
    for key, definition in DEFINITIONS.items():
        focus.append(row("awareness",f"What does {key} mean?",definition))
    focus.extend([
        row("awareness","im gay","Thanks for telling me. I'll respect how you describe yourself. 🏳️‍🌈"),
        row("awareness","I'm trans","Thanks for telling me. I'll respect your identity and the name and pronouns you use."),
        row("awareness","do you support LGBTQIA+?","LGBTQIA+ people deserve respect, safety, and equal treatment. 🏳️‍🌈"),
        row("emoji","😭💀","That looks like dramatic laughter or disbelief. What happened?"),
        row("emoji","this update is fire 🔥","Sounds like the update landed well 🔥 What changed?"),
        row("emoji","my dog died 😭","I'm sorry about your dog. Losing a pet can hurt a lot. Do you want to tell me about them?"),
    ])
    for name, _, capital in STATES.values():
        focus.append(row("geography",f"What is the capital of the state of {name}?",f"The capital of {name} is {capital}."))
    for name, _, _ in COUNTRIES.values():
        q=f"What is the capital of the country of {name}?"
        focus.append(row("geography",q,geography_response(q)))
    # Hand-reviewed multi-turn examples, unlike raw Discord transcripts.
    for q,a,next_q,next_a in [
        ("What is Gorilla Tag?", GAME_PAIRS[0][1], "What about its modding community?", GAME_PAIRS[14][1]),
        ("What is BepInEx?", GAME_PAIRS[28][1], "And Harmony?", GAME_PAIRS[30][1]),
        ("I'm making a Unity mod.", "Nice. What behavior are you adding?", "It crashes with a null reference.", "Check the stack trace for the exact line, then inspect which reference is null and why it was not assigned before use."),
        ("My pronouns are they/them.", "Got it. I'll use they/them for you in this chat.", "What pronouns did I give you?", "You told me your pronouns are they/them."),
    ]:
        focus.append({"source":"reviewed-games-20260923","category":"multi_turn","messages":[{"role":"user","content":q},{"role":"assistant","content":a},{"role":"user","content":next_q},{"role":"assistant","content":next_a}]})
    replay = []
    for filename in ("public_general_retrain.jsonl", "public_general_final_polish.jsonl"):
        replay.extend(json.loads(line) for line in (OUT/filename).read_text(encoding="utf-8").splitlines() if line.strip())
    rejected=Counter(); seen=set(); accepted=[]; accepted_focus=[]
    for item, is_focus in [(r,True) for r in focus]+[(r,False) for r in replay]:
        reason=rejection_reason(item)
        if reason:
            rejected[reason]+=1; continue
        fingerprint=json.dumps(item['messages'],sort_keys=True,ensure_ascii=False)
        if fingerprint in seen:
            rejected['duplicate']+=1; continue
        seen.add(fingerprint)
        (accepted_focus if is_focus else accepted).append(item)
    for filename,rows in [("public_games_focus.jsonl",accepted_focus),("public_games_replay.jsonl",accepted)]:
        random.Random(20260923).shuffle(rows)
        (OUT/filename).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    report={'focus_records':len(accepted_focus),'replay_records':len(accepted),'rejected':dict(rejected),
            'focus_categories':dict(Counter(r['category'] for r in accepted_focus)),
            'files':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ('public_games_focus.jsonl','public_games_replay.jsonl')},
            'raw_discord_logs_ingested':False,'runtime_answer_filter_added':False}
    (ROOT/'reports/games_dataset_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
