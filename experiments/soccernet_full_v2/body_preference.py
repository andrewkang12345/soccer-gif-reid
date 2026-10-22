"""Prefer reference photographs that show a whole player, without discarding the ones that do not.

Queries are tall: the broadcast box is a standing or running player, median aspect 2.13. References
are not: 64% of the gallery is portrait-shaped, because much of the web publishes footballers as
head-and-shoulders. It is tempting to filter those out, and that was measured and **rejected** —
on test, a gallery of every reference beats body-only (+0.65, not distinguishable) and clearly
beats portrait-only (+7.46 [+2.97, +12.01]), while validation ranks portrait-only *above*
body-only. Body references are at worst equal and possibly better, but portraits still contribute,
so the sound move is to collect more of the better kind rather than throw away the other.

The cues below are measured on the 3,265 frozen references, not guessed, comparing each cue's
full-body share against the 36.0% baseline. The strongest signal is the source, not the wording:
Wikimedia Commons runs 63.3% full-body over 547 references, while squad-profile databases such as
zerozero and ogol run at 0.0% over 68 to 121 — those sites publish headshots by design.
"""
import re,unicodedata

BASELINE=.360

# (cue, observed full-body share, sample size) from the frozen gallery.
BODY_HOSTS=(('commons.wikimedia',.633,547),('upload.wikimedia',.615,595))
PORTRAIT_HOSTS=(('zerozero',.000,68),('ogol',.000,68),('playmakerstats',.042,71),
                ('futurecdn',.117,60))
BODY_WORDS=(('boots',.803,61),('shorts',.685,92),('dressed',.609,64),('ground',.595,74),
            ('grass',.593,91),('pitch',.560,0),('kick',.560,0),('running',.560,0))
PORTRAIT_WORDS=(('planteis',.000,121),('estatisticas',.000,70),('informacoes',.000,64),
                ('epoca',.008,118),('stats',.042,71),('jogador',.057,88),('tall',.062,97),
                ('profile',.110,210),('spieler',.117,60),('headshot',.122,90),
                ('trimmed',.132,91),('beard',.137,102),('portrait',.150,0),('cropped',.150,0))


def _flat(*parts):
    text=' '.join(str(p or '') for p in parts)
    return unicodedata.normalize('NFKD',text.lower()).encode('ascii','ignore').decode()


def body_likelihood(*fields):
    """Rough probability that a candidate shows a whole player, from measured source and word cues.

    A preference for ordering, not a gate: nothing is ever rejected on this score. Cues are
    combined in log-odds against the base rate so several weak ones can outweigh one strong one.
    """
    text=_flat(*fields)
    odds=BASELINE/(1-BASELINE)
    for cues,present in ((BODY_HOSTS,True),(PORTRAIT_HOSTS,True),(BODY_WORDS,False),(PORTRAIT_WORDS,False)):
        for cue,share,_ in cues:
            hit=cue in text if present else re.search(rf'\b{cue}\b',text) is not None
            if hit:
                share=min(max(share,.02),.98)
                odds*=(share/(1-share))/(BASELINE/(1-BASELINE))
    return odds/(1+odds)
