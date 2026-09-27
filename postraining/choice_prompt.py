"""Single-choice problems under the bare answer-fence contract.

A multiple-choice problem's bare form is its question followed by its labelled
options: the options are part of the problem, not framing, so canonicalization
keeps them and removes only the source's answer-format instructions.  The
completion's ``<answer>`` span then holds exactly one option label.

Two consumers share this module.

* SFT (``prepare_sft_corpus``, UltraData-SFT-2605 ``Knowledge``): the source
  prompt opens with a two-line header -- a sentence listing the admissible
  labels, then a quoted final-line template such as
  ``'The answer is $LETTER'`` -- and the teacher's presented answer ends with
  that template filled in.  The header is removed, the options must match the
  labels it declares, and the filled-in label becomes the answer span.  Labels
  are kept as the source printed them (``C``, ``c`` or ``3``). The teacher's
  answers are skewed toward early labels; ``prepare_sft_corpus`` caps each
  answer position by dropping rows, and never renames a label in a trace.
* RL (``scripts/build_ultradata_knowledge_rl_prompts.py``): options are parsed
  out of a source query and re-rendered in one fixed style with upper-case
  letters, so every RL target is a single upper-case letter graded by the
  existing ``exact`` answer style.

Parsing fails closed: an option block that does not read as one contiguous,
consistently labelled run ``A, B, C, ...`` (or ``a, b, ...``/``1, 2, ...``)
ending the problem is not a single-choice problem this module can vouch for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

LABEL_STYLES = {
    "upper": tuple("ABCDEFGHIJKLMNOPQRST"),
    "lower": tuple("abcdefghijklmnopqrst"),
    "digit": tuple(str(number) for number in range(1, 21)),
}

# RL rendering. The UltraData-SFT-2605 Knowledge rows SFT admits use
# upper-case, lower-case and digit labels about equally, each with ``.``,
# ``:``, ``)`` or ``-`` (NOTES 2026-09-23), so ``A. text`` is one layout SFT
# has taught, and the upper-case letter is the answer span the ``exact``
# style grades.
RENDER_SEPARATOR = ". "
RENDERED_LABELS = LABEL_STYLES["upper"]

_LABEL = r"(?:[A-Za-z]|[1-9]\d?)"
_OPTION_LINE = re.compile(
    rf"[ \t]*(?:\((?P<paren>{_LABEL})\)|(?P<bare>{_LABEL})[ \t]*(?:[.:)]|-))"
    r"[ \t]*(?P<text>\S.*?)[ \t]*"
)
# The UltraData header's quoted final-line template, e.g.
# ``'The correct answer is $LETTER'``.
_CONCLUSION_TEMPLATE = re.compile(r"'(?P<conclusion>[^'\n$]*?)[ \t]*\$LETTER'")
_LABEL_LIST = re.compile(
    rf"(?<![\w,])(?P<list>{_LABEL}(?:, {_LABEL})*,? or {_LABEL})(?![\w])"
)


def label_style(label: str) -> str | None:
    for style, labels in LABEL_STYLES.items():
        if label in labels:
            return style
    return None


@dataclass(frozen=True)
class ChoiceProblem:
    question: str
    labels: tuple[str, ...]
    options: tuple[str, ...]


def split_options(problem: str) -> ChoiceProblem | None:
    """Split a problem into its question and its trailing labelled options.

    The option block is the maximal run of option lines (blank lines between
    them allowed) that ends the problem and shares one label style. Its
    labels must be exactly the first ``n`` labels of that style, in order, so
    a numbered premise list inside the question, a skipped letter or a
    multi-line option all return ``None`` instead of a partial parse.
    """

    lines = problem.rstrip().split("\n")
    block: list[tuple[str, str, int]] = []
    style = None
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index]
        if not line.strip():
            continue
        match = _OPTION_LINE.fullmatch(line)
        if match is None:
            break
        label = match["paren"] or match["bare"]
        current = label_style(label)
        if current is None or (style is not None and current != style):
            break
        style = current
        block.append((label, match["text"], index))
    block.reverse()
    if len(block) < 2 or style is None:
        return None
    labels = tuple(label for label, _, _ in block)
    if labels != LABEL_STYLES[style][: len(labels)]:
        return None
    question = "\n".join(lines[: block[0][2]]).strip()
    if not question:
        return None
    return ChoiceProblem(
        question=question,
        labels=labels,
        options=tuple(text for _, text, _ in block),
    )


def render_choice_problem(question: str, options: tuple[str, ...]) -> str:
    """The canonical RL layout: question, blank line, ``A. option`` lines."""

    if len(options) < 2 or len(options) > len(RENDERED_LABELS):
        raise ValueError(f"cannot render {len(options)} options")
    if any("\n" in option or not option.strip() for option in options):
        raise ValueError("options must be non-empty single lines")
    rendered = "\n".join(
        f"{label}{RENDER_SEPARATOR}{option.strip()}"
        for label, option in zip(RENDERED_LABELS, options)
    )
    return f"{question.strip()}\n\n{rendered}"


@dataclass(frozen=True)
class ChoiceFraming:
    problem: str  # bare problem: question and options, header removed
    labels: tuple[str, ...]
    conclusion: str  # required final-line prefix, e.g. "The answer is"


def strip_choice_framing(content: str) -> ChoiceFraming | None:
    """Remove UltraData's two-line single-choice header.

    Returns ``None`` when the prompt carries no ``$LETTER`` template (not a
    templated single-choice prompt) and raises ``ValueError`` when it does but
    the header, the declared labels or the options do not agree.
    """

    if "$LETTER" not in content:
        return None
    lines = content.split("\n")
    if len(lines) < 3 or "$LETTER" not in lines[1]:
        raise ValueError("choice template is not the header's second line")
    if "$LETTER" in "\n".join([lines[0], *lines[2:]]):
        raise ValueError("choice template appears outside the header")
    template = _CONCLUSION_TEMPLATE.search(lines[1])
    if template is None or not template["conclusion"].strip():
        raise ValueError("choice header has no quoted final-line template")
    declared = {
        tuple(re.split(r",? or |, ", match["list"]))
        for line in lines[:2]
        for match in _LABEL_LIST.finditer(line)
    }
    if len(declared) != 1:
        raise ValueError(f"choice header declares {len(declared)} label sets")
    labels = declared.pop()
    problem = "\n".join(lines[2:]).strip()
    parsed = split_options(problem)
    if parsed is None or parsed.labels != labels:
        raise ValueError("options do not match the header's declared labels")
    return ChoiceFraming(
        problem=problem,
        labels=labels,
        conclusion=template["conclusion"].strip(),
    )


# --- Trace screens -----------------------------------------------------------
#
# A single-choice trace becomes the <think> body before the <answer> label, so
# whatever it says the student learns to say. ``choice_trace_defect`` is the
# one screen for both trace sources, UltraData Knowledge and distilled teacher
# traces, and the SFT adapters re-apply it at build time.

# A response drafted inside the reasoning. UltraData Knowledge traces often
# conclude, then write "**Step-by-step derivation:**" (or "reasoning",
# "explanation", "solution") on a line of its own and recap the reasoning as
# the visible response will present it: 532 of 3,182 traces that passed the
# knowledge v3 screens carry one. Others plan the response under a header:
# "5. **Drafting the Explanation:**", "Formulate the explanation:",
# "**Explanation Construction:**", "Formulate the step-by-step reasoning:",
# each usually ending "Select option F." (the sixth and seventh red teams,
# NOTES 2026-09-24). The episode's response is the ``<answer>`` span alone,
# so either is output planning the student would learn to write, and
# ``choice_trace_defect`` drops the trace. The one exception is
# ``cut_derivation_recap``: a Knowledge trace that concludes before a
# step-by-step header of its own line keeps the reasoning above it.
_DRAFT_HEADER = (
    r"^[\W\d]*step-by-step[ \t]+(?:derivation|reasoning|explanation|solution)\W*$"
)
_PLAN_HEADER = (
    r"(?:(?:draft|formulat|construct|structur|writ|compos|plann?|outlin)(?:e|es|ed|ing)?"
    r"(?:[ \t]+(?:the|my|an?))?(?:[ \t]+(?:final|step-by-step))?[ \t]+"
    r"(?:explanation|response|answer|output|reasoning|justification)s?"
    r"|(?:explanation|response|answer|output)[ \t]+(?:construction|plan|outline|drafting"
    r"|structure))"
)
_HEADER_END = r"(?:\W*$|[ \t]*(?:\*\*)?[ \t]*:(?:\*\*)?[ \t]*\S)"
_DRAFTED_RESPONSE = re.compile(
    rf"{_DRAFT_HEADER}|^[\W\d]*{_PLAN_HEADER}{_HEADER_END}",
    re.IGNORECASE | re.MULTILINE,
)


# The teacher's reasoning discussing the answer-format header. Removing the
# header from the prompt leaves such reasoning answering an instruction the
# student never sees, and SFT would teach it to invent one: in a 2,048-token
# build, 75% of admitted traces quote ``$LETTER`` and most of the rest plan a
# "last line" or a "required format" (NOTES 2026-09-23). "The user wants
# ..." restating the question is not format talk and is left alone.
_FORMAT_TALK = re.compile(
    r"(?i)\$LETTER|\b(?:last|final) line\b|without quotes"
    r"|\b(?:required|requested|answer|output|response|specific) format\b"
    r"|\bformat(?:ting)? (?:instruction|requirement|constraint)s?\b"
    # "the instructions" alone is as often biology ("DNA carries the
    # instructions for life") as format talk, so an instruction must be the
    # user's or the format's, be followed, or say something.
    r"|\b(?:format(?:ting)?|user'?s?|given|system|these)[ \t]+instructions?\b"
    r"|\b(?:follow(?:s|ed|ing)?|per|obey(?:ing)?)[ \t]+(?:the[ \t]+)?instructions?\b"
    r"|\binstructions?[ \t]+(?:say|said|ask|asked|require|required|want|wanted"
    r"|state|stated|specify|specified)s?\b"
    # Planning the removed template's conclusion: "Conclude with 'The answer
    # is b'", a quoted "So, the final answer is C", "Format: answer: 3",
    # "Format the output", "formatted as", "the concluding phrase".
    r"|\b(?:conclude|end|finish)(?:s|ing)?[ \t]+with[ \t]+(?:['‘\"“]|(?:the|a|this)"
    r"[ \t]+(?:phrase|sentence|line|statement|string))"
    r"|['‘\"“](?:so,?[ \t]+)?(?:the[ \t]+)?(?:correct[ \t]+|final[ \t]+|best[ \t]+)?"
    r"answer(?:[ \t]+is\b|:)"
    r"|^[\W\d]*format(?:ting)?\W*:"
    r"|\bformat(?:ting)?[ \t]+(?:the[ \t]+)?(?:output|conclusion|answer|response|final)\b"
    r"|\bformat(?:ted)?[ \t]+(?:it[ \t]+)?as\b|\bformat output\b"
    r"|\bconcluding[ \t]+(?:phrase|sentence|line|statement)\b"
    # Planning the output itself (knowledge v3, NOTES 2026-09-24): "Is the
    # format correct?", "Final string: ...", "end with the specific string",
    # "asks for a single letter", "Letter: G", "make sure I output the
    # letter", "fit the constraints", "Step-by-step derivation for the final
    # output:".
    # "The format of a hash table", "the final response of the immune
    # system", "the one-letter code G" and "denoted by the letter k" are
    # science, so a format, a response and a letter must be the output's.
    # "Is the format correct?", "check the format"; not "the options follow
    # the format 'eigenvectors of ...'".
    r"|\bthe[ \t]+format[ \t]+(?:is|was|should|must|requires?|says|asks|correct|right"
    r"|ok|okay|fine)\b"
    r"|\b(?:check|verify|confirm|match|fit|follow|respect|obey)(?:ing|s|ed|es)?[ \t]+the[ \t]+"
    r"(?:required[ \t]+|answer[ \t]+|output[ \t]+)?format\b(?![ \t]+(?:of\b|[\"'“‘]))"
    r"|\bfinal[ \t]+response\b(?![ \t]+of\b)"
    r"|\bfinal[ \t]+string\b|\bspecific[ \t]+string\b"
    r"|\b(?:single|one)[ \t]+letter\b(?![- \t]+(?:codes?|abbreviations?|symbols?"
    r"|names?|notations?))"
    r"|\bletter[ \t]*[:：]"
    r"|\b(?:select|output|give|write|choose|pick|provide|state|return|put|print"
    r"|asks?|asking|wants?|requires?)(?:ing|s|ed)?[ \t]+(?:for[ \t]+)?(?:the|a|one|single)"
    r"[ \t]+(?:\w+[ \t]+)?letter\b"
    r"|\bthe[ \t]+letter\b(?![ \t]+(?:[A-Za-z](?![A-Za-z])|\$|['‘\"“]))"
    r"|\b(?:correct|answer|option|choice)[ \t]+letter\b"
    r"|\bmake[ \t]+sure[ \t]+(?:that[ \t]+)?(?:I|to)[ \t]+(?:output|write|end|include"
    r"|state|give|put|select|answer|choose|pick)\b"
    r"|\bfit(?:s|ting)?[ \t]+(?:all[ \t]+)?the[ \t]+constraints\b"
    r"|\bstep-by-step[ \t]+(?:derivation|reasoning|explanation|solution)[ \t]+for\b"
    r"|\bfor[ \t]+the[ \t]+final[ \t]+(?:output|response|answer)\b"
    # "**Final formatting:**", "follow the output constraints", "wants the
    # answer as the number", "Reasoning for the output:", and the removed
    # header's own demand, "eliminate wrong options with explanation".
    r"|\bfinal[ \t]+format(?:ting)?\b|\boutput[ \t]+(?:constraints?|requirements?|rules?)\b"
    r"|\bwants?[ \t]+the[ \t]+answer[ \t]+(?:as|to[ \t]+be|in)\b"
    r"|\bfor[ \t]+the[ \t]+(?:output|response)\b"
    r"|\beliminat(?:e|ing)[ \t]+(?:the[ \t]+)?wrong[ \t]+options[ \t]+with[ \t]+explanations?\b"
    r"|\b(?:draft|construct|write|compose|formulate)(?:ing)?[ \t]+the[ \t]+"
    r"(?:final[ \t]+)?(?:output|response)\b"
    # The sixth red team's: "The question asks for a single symbol from 1 to
    # 10.", "a single symbol response", "Letter choice: I.", "I should output
    # 6.". "A single symbol denotes ..." is notation, and "the prompt
    # constraints" are as often the question's qualifiers.
    r"|\b(?:asks?|asking|wants?|requires?|requests?)(?:ing|s|ed)?[ \t]+(?:for[ \t]+)?"
    r"(?:a|one)[ \t]+single[ \t]+symbol\b"
    r"|\bsingle[ \t]+symbol[ \t]+(?:response|answer|output)\b"
    r"|\bletter[ \t]+choice\b"
    r"|\bI(?:'ll|[ \t]+(?:should|will|must|need[ \t]+to))[ \t]+output\b"
    r"|(?-i:\bLETTER\b)",
    re.MULTILINE,
)


def cites_answer_format(reasoning: str) -> bool:
    """Whether reasoning refers to the answer-format header."""

    return _FORMAT_TALK.search(reasoning) is not None


# An answer line inside the reasoning, wherever it starts: "Answer: C",
# "#### Answer: C", "The answer is: C", "Answer = C", "Answer：C", "Correct
# option: C", "Answer - C", "**Answer** C", or "Final answer" with the label
# alone on the next line. The final line is removed before the reasoning is
# judged, so any left is a second answer line, which the <think> body must
# not teach.
_ANSWER_LINE_IN_REASONING = re.compile(
    rf"(?i)\banswer(?:[ \t]+is)?[ \t]*(?:\*\*)?[ \t]*[:：=]"
    r"|\b(?:correct|final|right|chosen|selected|best)[ \t]+(?:option|choice|letter)"
    r"[ \t]*(?:\*\*)?[ \t]*[:：=]"
    rf"|\banswer[ \t]*(?:\*\*)?[ \t]*[-–—][ \t]*(?:\*\*)?\(?{_LABEL}\)?(?:\*\*)?"
    r"[ \t]*(?:[.\n]|\Z)"
    rf"|\banswer[ \t]*(?:\*\*)?[ \t]*\n[ \t]*(?:\*\*)?\(?{_LABEL}\)?(?:\*\*)?"
    r"[ \t]*\.?[ \t]*(?:\n|\Z)"
    rf"|\*\*answer\*\*[ \t]*\(?{_LABEL}(?![\w'’])"
    # "Final selection: C.", "**Final Selection:** G.", "Result: C.",
    # "Option: d.", "Number: 3." alone on a line.
    r"|^[ \t>*#-]*(?:\d+[.)][ \t]*)?(?:\*\*)?(?:final[ \t]+)?(?:selection|decision|result"
    r"|pick|option|choice"
    r"|number|label)"
    rf"(?:\*\*)?[ \t]*[:：](?:\*\*)?[ \t]*(?:option[ \t]+)?\(?{_LABEL}\)?(?:\*\*)?\.?"
    r"(?:\*\*)?[ \t]*$",
    re.MULTILINE,
)
# Distilled traces talking about the teacher's own instructions, which the
# student never sees: the audience, the brevity budget, the answer line.
_TEACHER_META = re.compile(
    r"(?i)\bstudent model\b|\bsmall student\b|\bworked solutions?\b"
    r"|\bword (?:limit|count|budget)\b|\bas instructed\b|\bsystem prompt\b"
)
# Answer-key reasoning: a trace arguing toward what a key or a question
# writer expects instead of what the science says ("the intended answer",
# "similar questions online"), which is post-hoc rationalisation of a letter.
_ANSWER_KEY_TALK = re.compile(
    r"(?i)\b(?:intended|expected|official|keyed|standard|textbook)[ \t]+answers?\b"
    r"|\banswer[ \t]+keys?\b|\bsimilar[ \t]+questions?\b"
    r"|\b(?:question|quiz|test|exam)[ \t]+(?:setters?|writers?|authors?|makers?"
    r"|designers?)\b"
    r"|\bthe[ \t]+question[ \t]+(?:intends|expects)\b"
    # Doubting the question instead of answering it: "the question contains a
    # typo, likely intending ...", "Is there a trick?", "for this type of
    # question", "the conventional answer". Re-reading the question or the
    # options, and calling the question "the prompt", are checks on the
    # problem the student also sees, and stay.
    # "Flawed" alone is as often science ("the analysis will be flawed").
    r"|\btypos?\b|\bgarbled\b|\bmis-?(?:typed|printed|worded)\b"
    r"|\b(?:question|problem|premise|wording|item)s?\b[^.!?\n]{0,30}\bflawed\b"
    r"|\bflawed[ \t]+(?:question|problem|premise|wording|item)s?\b"
    r"|\btrick(?:y)?[ \t]+questions?\b|\b(?:a|the|is[ \t]+there[ \t]+a)[ \t]+trick\b"
    r"|\b(?:question|problem|prompt|author|writer|setter)s?\b[^.!?\n]{0,40}"
    r"\bintend(?:s|ed|ing)?\b"
    r"|\b(?:this|that|these|such)[ \t]+(?:types?|kinds?|sorts?)[ \t]+of"
    r"[ \t]+(?:[\w-]+[ \t]+)?questions?\b"
    r"|\b(?:conventional|accepted|commonly[ \t]+accepted|usual|typical)[ \t]+answers?\b"
    # Exam strategy instead of science (the fourth red team: 55 of 751
    # knowledge documents): "in multiple-choice questions like this, X is
    # the target answer", "the 'gold standard' correct answer", "the
    # standard 'wrong' answer", "questions of this nature", "in board exams",
    # "often tested", "a play on words". Being asked a multiple-choice
    # question ("The user wants me to answer a multiple-choice question
    # about ...") is the problem itself and stays.
    r"|\b(?:multiple[- ]choice|mcq|exam|board|test|quiz)[ \t]+questions\b"
    r"|\b(?:in|for|on)[ \t]+(?:(?:these|such|most|many|typical|standard|similar)[ \t]+)?"
    r"(?:multiple[- ]choice|mcq|exam|board|test|quiz)[ \t]+(?:questions?|exams?|items?)\b"
    r"|\bquestions?[ \t]+(?:like|such[ \t]+as)[ \t]+(?:this|these|that)\b"
    r"|\bquestions?[ \t]+of[ \t]+this[ \t]+(?:types?|nature|kinds?|sorts?)\b"
    r"|\bboard[ \t]+(?:exams?|examinations?|reviews?|questions?)\b"
    r"|\b(?:usmle|mcat|nclex)\b|\bap[ \t]+exams?\b"
    r"|\b(?:classic|standard|gold[- ]standard|target|typical|expected|textbook|intended"
    r"|official|keyed|conventional|accepted|usual|traditional)\W{0,3}(?:(?:correct|wrong|right"
    r"|incorrect)\W{0,3})?answers?\b"
    r"|\b(?:often|commonly|frequently)[ \t]+tested\b|\btest[- ]?taking\b|\bexaminers?\b"
    r"|\bplay[ \t]+on[ \t]+words\b|\briddle\b|[\"'“‘]trick[\"'”’]"
    # The fifth red team's: "in multiple-choice contexts", "a common
    # distractor", "strongly hints that this is the correct answer", "in the
    # context of general physics questions", "such questions usually
    # require", "the longest option". Calling an option a distractor after
    # refuting it is science and stays.
    r"|\b(?:in|for)[ \t]+(?:an?[ \t]+)?(?:multiple[- ]choice|mcq|exam|test)[ \t]+"
    r"(?:contexts?|settings?|formats?)\b"
    r"|\b(?:common|classic|typical|frequent|standard|usual|obvious)[ \t]+distractors?\b"
    r"|\bhints?[ \t]+(?:that|at)\b[^.!?\n]{0,40}\b(?:correct|right|intended)[ \t]+"
    r"(?:answer|option|choice)\b"
    r"|\bin[ \t]+the[ \t]+context[ \t]+of[ \t]+(?:[\w-]+[ \t]+){0,2}questions\b"
    r"|\bquestions[ \t]+(?:usually|typically|often|generally)[ \t]+(?:require|expect|want"
    r"|ask|test)\b"
    r"|\b(?:longest|shortest)[ \t]+(?:option|answer|choice)s?\b"
    # The sixth red team's (about 25 of 1,765 knowledge v4 documents): "in
    # such MCQs, you pick the closest value", "specificity usually wins in
    # multiple choice", "Usually, in physics/optics questions, ...", a
    # "board-exam style clue", "high-yield", "buzzword", "'f' is likely
    # intended to be false", "likely a distractor", "Consider the
    # 'Distractor' Information", recalling a dataset's label ("In PubMedQA,
    # the label for this question is typically 'yes'") and citing sources
    # the student cannot check ("Ref: UpToDate on ...").
    r"|\b(?:in|for)[ \t]+(?:these|such|most|many|typical|standard|similar)[ \t]+"
    r"(?:multiple[- ]choice|mcqs?)\b|\bin[ \t]+mcqs\b"
    r"|\bwins[ \t]+in[ \t]+multiple[- ]choice\b"
    # Generic "questions" (the seventh red team: 37 of 38 v5 dry-build
    # documents saying it argued from exam habit): "in the context of
    # standard organic synthesis questions", "Usually, clinical vignette
    # questions look for", "these questions test"; and the test-taking rule
    # that absolute words make an option false. "Research questions" and
    # "open questions" are science.
    r"|\b(?:in|for|these|such|usually|typically|often|generally|sometimes|standard"
    r"|textbook|typical|basic|general)\b,?(?:[ \t]+[\w/\"'“”-]+){0,6}?"
    r"(?<!research)(?<!open)(?<!unanswered)(?<!unresolved)[ \t]+questions\b"
    r"|\b(?:absolute|limiting|extreme)[ \t]+(?:words?|qualifiers?|language)\b"
    r"|\bboard-(?:exams?|examinations?|reviews?|questions?|style)\b"
    # "High-yield" as exam jargon, not a synthesis's yield.
    r"|\bhigh-yield[ \t]+(?:facts?|concepts?|associations?|topics?|clues?|points?"
    r"|pearls?|examples?|exams?|questions?|answers?)\b|\bbuzz-?words?\b"
    r"|\b(?:likely|probably|possibly|presumably|clearly)[ \t]+(?:intended|meant|designed)"
    r"[ \t]+(?:to[ \t]+be[ \t]+|as[ \t]+)?(?:an?[ \t]+)?(?:distractor|false|true|wrong"
    r"|correct|incorrect|trap)\b"
    r"|\b(?:likely|probably|possibly|presumably)[ \t]+(?:an?|the)[ \t]+(?:[\w-]+[ \t]+)?"
    r"distractors?\b"
    r"|\bdistractor[ \t]+testing\b|['‘\"“]distractors?\b"
    r"|\b(?:pubmedqa|medqa|medmcqa|mmlu|sciq|openbookqa|gpqa)\b"
    r"|\b(?:dataset|benchmark)[ \t]+labels?\b|\blabel[ \t]+for[ \t]+this[ \t]+question\b"
    r"|^[\W\d]*(?:refs?|references?|citations?)\.?[ \t]*(?:\*\*)?[ \t]*:"
    # An inline citation ("Vander Heiden et al., Science 2009") is the same
    # unverifiable recall: 107 v5 dry-build documents carried one.
    r"|\buptodate\b|\bet[ \t]+al\b",
    re.MULTILINE,
)

# A sentence of this length or more said this many times in a row is a loop,
# when it is prose. Scattered repeats are not: a trace that walks the options
# repeats its verdict ("This statement is false.") once per option, and 34 of
# the 235 UltraData Knowledge v2 traces do so with 0.75-0.99 distinct word
# 4-grams. Repeated math is not either: identical matrix rows, Punnett cells
# and arithmetic steps were 5 of the 13 knowledge v3 rows the unrestricted
# rule dropped.
_REPEATED_SENTENCE_CHARS = 20
_REPEATED_SENTENCE_TIMES = 3
_REPEATED_SENTENCE_WORDS = 4
# A trace this long whose word 4-grams are mostly repeats is a loop.
_DISTINCT_GRAM_MIN_WORDS = 60
_DISTINCT_GRAM_MIN_SHARE = 0.5


def repetitive_trace(reasoning: str) -> bool:
    """Whether a trace loops on a prose sentence or on its phrasing."""

    sentences = [
        " ".join(sentence.split()).casefold()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", reasoning)
        if sentence.strip()
    ]
    run = 1
    for previous, sentence in zip(sentences, sentences[1:]):
        run = run + 1 if sentence == previous else 1
        if (
            run >= _REPEATED_SENTENCE_TIMES
            and len(sentence) >= _REPEATED_SENTENCE_CHARS
            and len(re.findall(r"[^\W\d_]{2,}", sentence)) >= _REPEATED_SENTENCE_WORDS
        ):
            return True
    words = reasoning.casefold().split()
    if len(words) < _DISTINCT_GRAM_MIN_WORDS:
        return False
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    return len(set(grams)) / len(grams) < _DISTINCT_GRAM_MIN_SHARE


# A short unit (with a word character) repeated 20+ times: "The. The. The.",
# "\\chi\\chi\\chi", "ERERER". Separator rules ("-----") have no word character.
_DEGENERATE_RUN = re.compile(r"(.{1,12}?)\1{19,}", re.DOTALL)
# Trailing debris after the last sentence: "correct answer.cw", "write.cltr".
_TRAILING_DEBRIS = re.compile(r"[.!?][a-z]{2,4}\Z")
# A trace cut off mid-sentence: "In the femtosecond regime (", "... because
# chloroplasts contain", "C, i.e.", "C...". Every trace kept from 3,182
# knowledge and 512 teacher candidates ends on a full stop, so a trace must
# end on closing punctuation, not after an ellipsis or an abbreviation,
# with its inline math closed, and not on a word a sentence cannot end on
# ("C and.", "the value of the."). A label is exempt: "the answer is a."
# ends on option a.
_CLOSED_END = re.compile(r"[.!?](?P<closers>[\"'”’)\]*]*)\Z")
_TRAILING_OFF = re.compile(r"(?i)(?:\.\.|…|\b(?:i\.e|e\.g|vs|cf))[.!?]?\Z")
_FUNCTION_WORD_END = re.compile(
    r"(?i)(?<![^\W_])(?:the|a|an|and|or|but|nor|than|whose|its|their|his|her|our"
    r"|such|very|because|since|while|when|where|if)\Z"
)
# Inline math by pandoc's rule: an opening "$" has a non-space to its right,
# a closing one a non-space to its left and no digit after it, so "$4s^1$"
# is math and "$5" is money. Display "$$...$$" spans go first. Math is open
# when an opener is left once the closed spans are removed.
_DISPLAY_MATH = re.compile(r"(?<!\\)\$\$.*?(?<!\\)\$\$")
_INLINE_MATH = re.compile(r"(?<!\\)\$(?=[^\s$])[^$\n]*?(?<=[^\s\\])\$(?!\d)")
_MATH_OPENER = re.compile(r"(?<!\\)\$(?=[^\s\d$])")


def truncated_trace(reasoning: str, labels: tuple[str, ...]) -> bool:
    """Whether a trace stops mid-sentence."""

    text = reasoning.rstrip()
    closed = _CLOSED_END.search(text)
    if closed is None:
        return True
    body = text[:closed.start()]
    if _TRAILING_OFF.search(text[:closed.start("closers")]):
        return True
    last_line = _INLINE_MATH.sub("", _DISPLAY_MATH.sub("", body.split("\n")[-1]))
    if _MATH_OPENER.search(last_line) or "$$" in last_line:
        return True
    word = _FUNCTION_WORD_END.search(body)
    return word is not None and word[0] not in labels


_DERIVATION_RECAP = re.compile(_DRAFT_HEADER, re.IGNORECASE | re.MULTILINE)


def cut_derivation_recap(reasoning: str) -> str:
    """``reasoning`` above its first step-by-step header line, if any.

    UltraData Knowledge traces reason to a conclusion, then recap it under
    "**Step-by-step derivation:**" as the visible response will present it.
    Rejecting drafted traces leaves 83 documents in a 2,048-token build,
    cutting at this header 828 after balancing, and the median cut
    trace keeps 70% of its text (NOTES 2026-09-24). The cut is only at that
    header, which ends the reasoning proper; the kept part still passes
    every ``choice_trace_defect`` screen, including a planning header above
    the cut. Teacher traces are never cut.
    """

    header = _DERIVATION_RECAP.search(reasoning)
    return reasoning if header is None else reasoning[: header.start()].rstrip()


def choice_trace_defect(
    reasoning: str, choice: ChoiceProblem, answer: str | None = None
) -> str | None:
    """Why a single-choice trace is unusable as SFT reasoning, or ``None``.

    ``reasoning`` is the <think> body: an UltraData ``reasoning_content`` or
    a teacher completion without its final answer line. The screens are
    mechanical, as in rejection-sampling fine-tuning: correctness is the
    final label's alone, graded by the caller, and the prose is not parsed
    for a conclusion, so hedges, "wait" and rejected options stay as the
    teacher wrote them (NOTES 2026-09-24). A direct assertion that the
    verified answer option is wrong is rejected, including a later
    self-correction: it would teach contradictory option labels. The body
    must hold no answer line of its own and no drafted response, must not
    talk about the answer format, the teacher's instructions or an answer
    key, and must not be
    degenerate, end in debris, stop mid-sentence or loop.
    """

    text = reasoning.rstrip()
    if _ANSWER_LINE_IN_REASONING.search(text):
        return "reasoning_holds_answer_line"
    if _DRAFTED_RESPONSE.search(text):
        return "reasoning_drafts_response"
    if cites_answer_format(text):
        return "reasoning_cites_answer_format"
    if _TEACHER_META.search(text):
        return "reasoning_cites_teacher_instructions"
    if _ANSWER_KEY_TALK.search(text):
        return "reasoning_cites_answer_key"
    if answer is not None and re.search(
        rf"\b(?:option|choice)\s+{re.escape(answer)}\b"
        rf"(?:\s*\([^)]{{0,80}}\))?\s+(?:is|was)\s+"
        r"(?:clearly\s+)?(?:incorrect|wrong|not\s+(?:correct|right))\b",
        text,
        re.IGNORECASE,
    ):
        return "reasoning_rejects_answer"
    if any(re.search(r"\w", match[1]) for match in _DEGENERATE_RUN.finditer(text)):
        return "degenerate_trace"
    if _TRAILING_DEBRIS.search(text):
        return "trace_trailing_debris"
    if truncated_trace(text, choice.labels):
        return "trace_truncated"
    if repetitive_trace(text):
        return "repetitive_trace"
    return None


def concluded_label(presented: str, framing: ChoiceFraming) -> str | None:
    """The label the presented answer's final line commits to, or ``None``.

    The last non-empty line must be the header's template filled in with one
    declared label, and every other filled-in template line in the answer
    must name the same label: a response that concludes twice with different
    labels has no single answer to teach.
    """

    pattern = re.compile(
        rf"[ \t]*{re.escape(framing.conclusion)}[ \t]*"
        rf"(?P<label>{_LABEL})[ \t]*\.?[ \t]*"
    )
    lines = [line for line in presented.split("\n") if line.strip()]
    if not lines:
        return None
    final = pattern.fullmatch(lines[-1])
    if final is None or final["label"] not in framing.labels:
        return None
    named = {
        match["label"]
        for line in lines
        if (match := pattern.fullmatch(line)) is not None
    }
    return final["label"] if named == {final["label"]} else None


# --- Label lists --------------------------------------------------------------
#
# ``choice_rl_pool`` refuses options that name other options by bare label
# ("Both A and C"), which a shuffle would leave pointing at the wrong texts.

_LIST_JOINER = re.compile(r"\)?(?:\*\*)?[ \t]*(?:,|/|&|,?[ \t]*(?:and|or))[ \t]*\(?(?:\*\*)?")


def _label_tokens(labels: tuple[str, ...]) -> re.Pattern:
    alternatives = "|".join(
        sorted(map(re.escape, labels), key=len, reverse=True)
    )
    # Standalone: not inside a word, a number or a LaTeX command, and not a
    # contraction ("I'm", the "d" of "I'd"). Quotes, "$", braces, underscores
    # and subscripts do not hide a token: "'C'", "$D$", "__D__" and
    # "\boxed{B}" are tokens. ``[^\W_]`` is a word character other than "_",
    # so "__D__" and "x_A" expose their label rather than hide it.
    return re.compile(
        rf"(?<![^\W_])(?<!\\)(?<![^\W_]['’])(?:{alternatives})(?![^\W_])"
        rf"(?!['’](?:m|ll|ve|d|re)\b)"
    )


def bare_label_list(text: str, labels: tuple[str, ...]) -> bool:
    """Whether two label tokens are joined as a list ("A and C", "B, D")."""

    tokens = list(_label_tokens(labels).finditer(text))
    return any(
        _LIST_JOINER.fullmatch(text[first.end():second.start()])
        for first, second in zip(tokens, tokens[1:])
    )
