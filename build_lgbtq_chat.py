"""Original conversational LGBTQIA+ examples; no Discord identifiers or replies."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build():
    rows = []
    def add(question, answer, followup=None):
        messages = [{'role':'user','content':question},{'role':'assistant','content':answer}]
        if followup:
            messages.extend([{'role':'user','content':followup[0]},{'role':'assistant','content':followup[1]}])
        rows.append({'source':'original-lgbtq-chat-20260923','messages':messages})
    for label in ['gay','trans','transgender','lesbian','bisexual','bi','pansexual','asexual','nonbinary','queer','intersex','a femboy']:
        answers = ['Cool, thanks for sharing. How are you feeling about it?',
                   "Got it. You can be yourself here.",
                   "Thanks for telling me. Want to talk about it, or just hanging out?",
                   "Okay! What's on your mind?",
                   "Nice to meet you. What would you like to chat about?",
                   "That's fine with me. Anything you wanted to discuss about it?"]
        for n,template in enumerate(["I'm {}",'im {}','I am {}',"hey, I'm {}","btw im {}","just wanted to say I'm {}", "I am {} actually", "I'm {} btw", "Im {}", "I identify as {}", "hey im {}", "I am {}."]):
            q=template.format(label);a=answers[n % len(answers)]
            add(q,a)
            if n == 0:
                add(q,a,('are you not gay',"I don't have an orientation myself. I'm a bot, but I'm happy to talk about yours."))
    for q in ['are you gay','are u gay','are you not gay','so you arent gay','you gay?','are you straight','ur gay','do you like boys','do you like girls','are you bisexual','are you trans','what is your sexuality']:
        for prefix,a in [('',"I don't have a sexuality myself. I'm a bot, but we can talk about it."),('hey ',"No orientation here—I'm an AI. How about we talk about what's on your mind?"),('wait ',"I don't feel attraction. I can still chat about LGBTQIA+ stuff with you.")]:
            add(prefix+q,a)
    for q,a in [
        ('does being a femboy mean im gay','No. Feminine presentation does not decide your orientation. A femboy can be gay, bi, straight, or use another label.'),
        ('does being a femboy make me gay?', 'No. Being a femboy is about presentation, not who you are attracted to.'),
        ('are femboys always gay', 'No. Femboys can have different orientations; presentation does not decide it.'),
        ('im trans and gay','Those can go together. Gender identity and who you are attracted to are different things.'),
        ('im not gay','Got it. You decide which labels fit you.'),
        ('im not trans','Okay. I will not assume that you are.'),
        ('i think i might be gay','You can take your time figuring it out. There is no deadline for choosing a label.'),
        ('im scared to come out','That sounds stressful. You do not have to tell everyone at once; choose the timing and people you feel comfortable with.'),
        ('my friend came out as trans','Listen to them and use the name and pronouns they ask for. You do not need a perfect speech to be supportive.'),
        ('im gay dont make it a big deal','Got it. What else is up?'),
        ('im trans dont give me a speech','Okay. What are you up to?'),
        ('are gay people okay','Yes. Being gay is okay, and gay people deserve respect.'),
        ('do you support trans people','Yes. Trans people deserve respect and equal treatment.'),
        ('my pronouns are they/them',"Got it, I'll use they/them."),
        ('can we talk about being nonbinary','Sure. What part would you like to talk about?'),
    ]:
        for prefix in ['', 'hey ', 'also ']: add(prefix+q,a)
    from chudlm.sft_data import load_sft_records
    p=ROOT/'data/public_lgbtq_chat.jsonl'
    p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    assert len(load_sft_records(p)) == len(rows)
    print(len(rows))

if __name__ == '__main__': build()
