import glob
import json
from collections import Counter
import pandas as pd
import re


def post_process_enum(sentences):
    """
    Post-process sentences to:
    1. Further split enumerations like "oils.7, 8 Observational"
    2. Attach numeric citations (numbers separated by commas) to the previous sentence

    Author: Antoine D. Lain
    """
    processed = []

    enum_pattern = re.compile(r"(\.\s*)(\d+(?:,\s*\d+)*\s+)(?=[A-Z])")

    for sent in sentences:
        last_idx = 0
        splits = []

        # Step 1: split enumerations
        for m in enum_pattern.finditer(sent):
            start, end = m.span()
            # Sentence before enumeration
            pre_enum = sent[last_idx : start + len(m.group(1))].strip()
            if pre_enum:
                splits.append(pre_enum)
            # Enumeration part
            enum_part = sent[start + len(m.group(1)) : end].strip()
            if enum_part:
                splits.append(enum_part)
            last_idx = end

        # Remaining part after last enumeration
        remainder = sent[last_idx:].strip()
        if remainder:
            splits.append(remainder)

        # Step 2: attach numeric citations to previous sentence
        for s in splits:
            if re.fullmatch(r"[\d,\s]+", s):
                if processed:
                    processed[-1] = processed[-1].rstrip() + s
            else:
                processed.append(s)
    return processed


def split_section_headers(sentences):
    """
    Post-process to split incorrectly joined section headers such as:
    'diet.Methods:' → 'diet.' + 'Methods:'
    Handles multiple occurrences within a sentence.
    """
    processed = []
    # Pattern: dot + capitalized word(s) + colon
    pattern = re.compile(
        r"\.(?=[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*:)", flags=re.UNICODE
    )

    for sent in sentences:
        # Split at each match, keep delimiters (the section names)
        parts = re.split(pattern, sent)
        for part in parts:
            cleaned = part.strip()
            if cleaned:
                processed.append(cleaned)
    return processed


def split_section_headers_defined(sentences):
    """
    Split sentences where section-like keywords (e.g. '.Summary', '.Objective')
    appear immediately after a dot.

    Example:
    '... as seen in the graph.Summary To conclude ...'
    → ['... as seen in the graph.', 'Summary To conclude ...']
    """
    _check_list = [
        ".Aim",
        ".Motivation",
        ".Objective",
        ".Method",
        ".Methods",
        ".Data",
        ".Materials",
        ".Result",
        ".Results",
        ".Finding",
        ".Summary",
        ".Discussion",
        ".Conclusion",
    ]

    # Regex pattern for any of the section markers
    pattern = r"(" + "|".join(re.escape(item) for item in _check_list) + r")(?=\b)"

    processed = []
    for s in sentences:
        # Insert a split marker before the section header (replace the dot)
        new_s = re.sub(pattern, lambda m: ".|||SPLIT|||" + m.group(0)[1:], s)
        # Split on the marker
        parts = [p.strip() for p in new_s.split("|||SPLIT|||") if p.strip()]
        processed.extend(parts)

    return processed


def split_at_celsius(sentences):
    """
    Split sentences at '°C.'.

    Example:
    'stored at -80°C. Frozen'
    → ['stored at -80°C.', 'Frozen']

    Args:
        sentences (list of str): List of sentences to process.

    Returns:
        list of str: List of sentences after splitting at '°C.'.
    """
    processed = []

    pattern = re.compile(r"(°C\.)\s*")  # Match '°C.' followed by optional spaces

    for sent in sentences:
        parts = []
        last_idx = 0

        for m in pattern.finditer(sent):
            start, end = m.span()
            # Sentence up to and including °C.
            pre_c = sent[last_idx:end].strip()
            if pre_c:
                parts.append(pre_c)
            last_idx = end

        # Any remaining part after last °C.
        remainder = sent[last_idx:].strip()
        if remainder:
            parts.append(remainder)

        processed.extend(parts)

    return processed


def merge_reference_fragments(sentences):
    merged = []
    for s in sentences:
        if not merged:
            merged.append(s)
            continue

        # Case 1: sentence is only numbers, commas, and spaces
        if re.fullmatch(r"[\d,\s]+", s.strip()):
            merged[-1] = merged[-1].rstrip() + " " + s.strip()
            continue

        # Case 2: sentence starts with numbers, commas, and spaces, then text
        m = re.match(r"^([\d,\s]+)([A-Z].*)", s.strip())
        if m:
            nums_part, text_part = m.groups()
            merged[-1] = merged[-1].rstrip() + " " + nums_part.strip()
            merged.append(text_part.strip())
        else:
            merged.append(s)

    return merged


def protect_dots_in_parentheses(text, max_lookahead=500):
    chars = []
    inside_par = 0
    n = len(text)
    i = 0

    while i < n:
        ch = text[i]

        if ch == "(":
            # Only count as inside parentheses if there's a closing ')' ahead
            lookahead_end = min(i + max_lookahead, n)
            if ")" in text[i:lookahead_end]:
                inside_par += 1
            chars.append(ch)

        elif ch == ")":
            # Close only if we’re currently inside parentheses
            if inside_par > 0:
                inside_par -= 1
            chars.append(ch)

        elif ch == "." and inside_par > 0:
            # Protect only if we are inside valid parentheses
            chars.append(r"/\/\/")
        else:
            chars.append(ch)

        i += 1

    return "".join(chars)


def split_dash_numeric_sentences(sentences):
    """
    Split sentences where a dot is followed by a token containing at least one digit and one dash,
    e.g., 'HDL-C (p < 0.05). 5-Aminolevulinate' → ['HDL-C (p < 0.05).', '5-Aminolevulinate...']
    Also handles 'FADS alleles. n-3 PUFA'
    """
    abbreviations = [
        "et al.",
        "i.e.",
        "e.g.",
        "sp.",
        "spp.",
        "ssp.",
        "vs.",
        "fig.",
        "Fig.",
        "Figure.",
        "Table.",
        "Ref.",
        "Refs.",
        "Eq.",
        "Eqn.",
        "Tab.",
        "Tabs.",
        "Dr.",
        "Prof.",
        "Mr.",
        "Mrs.",
        "Ms.",
        "No.",
        "no.",
        "St.",
        "etc.",
        "Ph.D.",
        "Figs.",
        "figs.",
        "min.",
        "approx.",
        "v.",
        "ca.",
        "n.",
        "ver.",
        "ref.",
        "i.p.",
        "a.m.",
        "p.m.",
        "subsp.",
        "Govt.",
        "incl.",
        "i. p.",
        "Jr.",
    ]

    processed = []
    for s in sentences:
        s = protect_dots_in_parentheses(s)
        s = s.replace(".0", r"/\/\/0")
        s = re.sub(
            r"\b(?:[A-Za-z]\.){2,}(?=[A-Za-z]?)",
            lambda m: m.group(0).replace(".", r"/\/\/"),
            s,
            flags=re.IGNORECASE,
        )
        # Protect abbreviations
        for abbr in abbreviations:
            escaped_abbr = re.escape(abbr)
            # Match abbreviation only if not surrounded by letters (case-insensitive)
            pattern = rf"(?<![A-Za-z]){escaped_abbr}(?![A-Za-z])"
            s = re.sub(pattern, abbr.replace(".", r"/\/\/"), s, flags=re.IGNORECASE)
        s = re.sub(
            r"(?<![A-Z-])([A-Z])\.(?=\s*[a-z])|(?<=\b)(\d[A-Z])\b", r"\1\2/\/\/", s
        )

        # Split pattern:
        # - look for a dot followed by space
        # - next token contains a digit and a dash
        split_pattern = re.compile(r"\.\s+(?=[A-Za-z\-]*\d+[A-Za-z\-]*[\sA-Z]*)")

        s = split_pattern.sub(".|||SPLIT|||", s)

        # Restore abbreviations
        s = s.replace(r"/\/\/.", ".").replace(r"/\/\/", ".")

        # Split into list
        split_parts = [p.strip() for p in s.split("|||SPLIT|||") if p.strip()]
        processed.extend(split_parts)

    return processed


def text_to_token(text):
    # --- 1. Protect dot in paranthesis ---
    text = protect_dots_in_parentheses(text)
    text = text.replace(".0", r"/\/\/0")
    text = re.sub(
        r"\b(?:[A-Za-z]\.){2,}(?=[A-Za-z]?)",
        lambda m: m.group(0).replace(".", r"/\/\/"),
        text,
        flags=re.IGNORECASE,
    )
    # --- 1. Protect abbreviations and initials ---
    abbreviations = [
        "et al.",
        "i.e.",
        "e.g.",
        "sp.",
        "spp.",
        "ssp.",
        "vs.",
        "fig.",
        "Fig.",
        "Figure.",
        "Table.",
        "Ref.",
        "Refs.",
        "Eq.",
        "Eqn.",
        "Tab.",
        "Tabs.",
        "Dr.",
        "Prof.",
        "Mr.",
        "Mrs.",
        "Ms.",
        "No.",
        "no.",
        "St.",
        "etc.",
        "Ph.D.",
        "Figs.",
        "figs.",
        "min.",
        "approx.",
        "v.",
        "ca.",
        "n.",
        "ver.",
        "ref.",
        "i.p.",
        "a.m.",
        "p.m.",
        "subsp.",
        "Govt.",
        "incl.",
        "i. p.",
        "Jr.",
    ]

    for abbr in abbreviations:
        escaped_abbr = re.escape(abbr)
        # Match abbreviation only if not surrounded by letters (case-insensitive)
        pattern = rf"(?<![A-Za-z]){escaped_abbr}(?![A-Za-z])"
        text = re.sub(pattern, abbr.replace(".", r"/\/\/"), text, flags=re.IGNORECASE)

    # Protect single-letter panels (A., B., etc.)
    text = re.sub(
        r"(?<![A-Z-])([A-Z])\.(?=\s*[a-z])|(?<=\b)(\d[A-Z])\b", r"\1\2/\/\/", text
    )

    # --- 2. Normalise spaces ---
    text = re.sub(r"\s+", " ", text.strip())
    # --- 3. Split sentences ---
    # Match periods followed by digits, commas, hyphens, en-dash, or bracketed references
    # Only split if the period is NOT immediately before a comma
    split_pattern = r"((?<!\d)\.(?:[0-9,\-–]|\[(?:[0-9,\-–,]+)\])*(?<!\,))\s+(?=[A-Z])"
    parts = re.split(split_pattern, text)

    sentences = []
    for i in range(0, len(parts), 2):
        seg = parts[i].strip()
        delim = parts[i + 1] if (i + 1) < len(parts) else ""
        sentence = (seg + delim).strip()
        if sentence:
            sentences.append(sentence)

    # --- 4. Restore protected periods ---
    sentences = [
        s.replace(r"/\/\/.", ".").replace(r"/\/\/", ".").strip()
        for s in sentences
        if s.strip()
    ]
    sentences = post_process_enum(sentences)
    sentences = split_section_headers(sentences)
    sentences = split_dash_numeric_sentences(sentences)
    sentences = split_section_headers_defined(sentences)
    sentences = split_at_celsius(sentences)
    sentences = merge_reference_fragments(sentences)

    return sentences


import difflib


def sentence_split_spans(sentence):

    # preprocessing spaces
    sentence = re.sub(r"\s+", " ", sentence.strip())

    sentences = text_to_token(sentence)
    spans = []
    current_pos = 0

    for sent in sentences:
        # start_idx = sentence.find(sent, current_pos)
        sentence_lower = sentence.lower()
        sent_lower = sent.lower()

        start_idx = sentence_lower.find(sent_lower, current_pos)

        if start_idx != -1:
            end_idx = start_idx + len(sent)

        if start_idx == -1:
            matcher = difflib.SequenceMatcher(None, sentence.lower(), sent.lower())
            match = matcher.find_longest_match(0, len(sentence), 0, len(sent))

            start_idx = match.a
            end_idx = match.a + 30

            print("split sent       : ", sent)
            print("original sentence: ", sentence)

        spans.append((start_idx, end_idx))
        current_pos = end_idx

    return spans
