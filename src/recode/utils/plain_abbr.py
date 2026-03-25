import copy
import logging
import re
import textwrap
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import regex as re2

replace_dict = {"A": "α", "B": "β", "1": "one", "2": "two", "3": "three", "γ": "gamma", "α": "alpha", "β": "beta", "/": "and"}

stop_words_str = \
    textwrap.dedent(
        """
        i
        me
        my
        myself
        we
        our
        ours
        ourselves
        you
        you're
        you've
        you'll
        you'd
        your
        yours
        yourself
        yourselves
        he
        him
        his
        himself
        she
        she's
        her
        hers
        herself
        it
        it's
        its
        itself
        they
        them
        their
        theirs
        themselves
        what
        which
        who
        whom
        this
        that
        that'll
        these
        those
        am
        is
        are
        was
        were
        be
        been
        being
        have
        has
        had
        having
        do
        does
        did
        doing
        a
        an
        the
        and
        but
        if
        or
        because
        as
        until
        while
        of
        at
        by
        for
        with
        about
        against
        between
        into
        through
        during
        before
        after
        above
        below
        to
        from
        up
        down
        in
        out
        on
        off
        over
        under
        again
        further
        then
        once
        here
        there
        when
        where
        why
        how
        all
        any
        both
        each
        few
        more
        most
        other
        some
        such
        no
        nor
        not
        only
        own
        same
        so
        than
        too
        very
        s
        t
        can
        will
        just
        don
        don't
        should
        should've
        now
        d
        ll
        m
        o
        re
        ve
        y
        ain
        aren
        aren't
        couldn
        couldn't
        didn
        didn't
        doesn
        doesn't
        hadn
        hadn't
        hasn
        hasn't
        haven
        haven't
        isn
        isn't
        ma
        mightn
        mightn't
        mustn
        mustn't
        needn
        needn't
        shan
        shan't
        shouldn
        shouldn't
        wasn
        wasn't
        weren
        weren't
        won
        won't
        wouldn
        wouldn't
        """)

stop_words = []
for stop_word in stop_words_str.split('\n'):
    stop_word = stop_word.strip()
    if stop_word != "":
        stop_words.append(stop_word)


def abbre_pattern(abbreviation: str):
    pattern = 'w'
    for ch in ['/', '\\', ',', '.', '-', '_', '&', '+']:
        if ch in abbreviation:
            abbreviation = abbreviation.replace(ch, ' ').strip()
    for c in abbreviation:
        if c == ' ':
            pattern += ' '
        elif c.isdigit() & (pattern[-1] in ['w', ' ']):
            pattern += 'n'
        elif not c.isdigit():
            pattern += 'w'

    return pattern[1:].replace(' ', '')


def find_shortest_candidate(arr_sentence: list, abbreviation: str, start_idx, end_idx):
    one_def = []
    temp_sentence = arr_sentence.copy()
    temp_abbre = abbreviation
    for i in range(start_idx, end_idx):
        if len(one_def) == 0:
            if temp_sentence[i][0].lower() == temp_abbre[0].lower():
                temp_sentence[i] = temp_sentence[i][1:]
                temp_abbre = temp_abbre[1:]
                one_def.append(i)
            else:
                continue
        while len(temp_abbre) != 0:
            temp_letter = temp_abbre[0]
            temp_word = temp_sentence[i]
            if temp_letter.lower() in temp_word.lower():
                one_def.append(i)
                temp_sentence[i] = temp_word[(temp_word.lower().index(temp_letter.lower()) + 1):]
                temp_abbre = temp_abbre[1:]
            else:
                break


def find_all_candidate(arr_sentence: list, abbreviation: str, start_idx, end_idx):
    all_candidate = []
    clean_abbreviation = abbreviation

    for ch in ['/', '\\', ',', '.', '-', '_', '&', '+']:
        if ch in clean_abbreviation:
            clean_abbreviation = clean_abbreviation.replace(ch, ' ').strip()
    separate_abbre = []
    numbers = ''
    if len(clean_abbreviation) > 10:
        return []

    for i in range(len(clean_abbreviation)):
        if clean_abbreviation[i] == ' ':
            if len(numbers) > 0:
                separate_abbre.append(numbers)
                numbers = ''
            continue
        elif clean_abbreviation[i].isnumeric():
            numbers += clean_abbreviation[i]
            if i == len(clean_abbreviation) - 1:
                separate_abbre.append(numbers)
                numbers = ''
        else:
            if len(numbers) > 0:
                separate_abbre.append(numbers)
                numbers = ''
            separate_abbre.append(clean_abbreviation[i])

    temp_candidates = []
    temp_sentence = []

    for i in range(len(separate_abbre)):
        abbre_char = separate_abbre[i]
        replace_char = replace_dict.get(abbre_char)
        if i == 0:
            for j in range(start_idx, end_idx):
                if abbre_char.isnumeric() and arr_sentence[j].lower():
                    all_candidate.append([(j, -1)])
                elif abbre_char.lower() == arr_sentence[j].lower()[0]:
                    all_candidate.append([(j, 0)])
                elif replace_char is not None and replace_char.lower() == arr_sentence[j].lower()[0]:
                    all_candidate.append([(j, -2)])
        elif len(all_candidate) > 0:
            for one in all_candidate:
                tmp_oneCandidate = one.copy()
                if len(one) == i:
                    last_flag = one[-1]
                    temp_sentence = arr_sentence.copy()
                    temp_sentence[last_flag[0]] = temp_sentence[last_flag[0]][
                                                  (last_flag[1] + len(separate_abbre[i - 1])):]
                    for m in range(last_flag[0], end_idx):
                        if m - last_flag[0] > 2:
                            break
                        else:
                            if abbre_char.lower() in temp_sentence[m].lower():
                                idxes = [e.start() for e in re.finditer(abbre_char.lower(), temp_sentence[m].lower())]
                                para = 1 if m == last_flag[0] else 0
                                for idx in idxes:
                                    tmp_oneCandidate.append(
                                        (m, idx + para * (last_flag[1] + len(separate_abbre[i - 1]))))
                                    temp_candidates.append(tmp_oneCandidate)
                                    tmp_oneCandidate = one.copy()
                            if replace_char is not None:
                                if replace_char in temp_sentence[m].lower():
                                    idxes = [e.start() for e in
                                             re.finditer(replace_char.lower(), temp_sentence[m].lower())]
                                    para = 1 if m == last_flag[0] else 0
                                    for idx in idxes:
                                        tmp_oneCandidate.append(
                                            (m, idx + para * (last_flag[1] + len(separate_abbre[i - 1]))))
                                        temp_candidates.append(tmp_oneCandidate)
                                        tmp_oneCandidate = one.copy()
            if len(temp_candidates) == 0:
                all_candidate = []
                break
            else:
                all_candidate = temp_candidates.copy()
                temp_candidates = []
    return all_candidate


def separate_sentence(sentence: str, abbre: str):
    tmp_sen = sentence.replace(abbre, '<<FLAG>>')
    char_to_replace = {'(': ' ',
                       ')': ' ',
                       '{': ' ',
                       '}': ' ',
                       '[': ' ',
                       ']': ' ',
                       ', ': ' ',
                       '; ': ' ',
                       '-': ' ',
                       '+': ' ',
                       '_': ' ',
                       ',': ' ',
                       '‐': ' '}
    for key, value in char_to_replace.items():
        tmp_sen = tmp_sen.replace(key, value)
    arr_sentence = tmp_sen.split(' ')
    arr_sentence = list(filter(lambda a: a != '', arr_sentence))
    new_arr_sentence = []
    for item in arr_sentence:
        if '<<FLAG>>' == item:
            new_arr_sentence.append(abbre)
            continue
        elif '<<FLAG>>' in item:
            item = item.replace('<<FLAG>>', abbre)
        match = re.match(r"([^0-9]+)([0-9]+)", item, re.I)
        if match:
            new_arr_sentence.extend(match.groups())
        else:
            new_arr_sentence.append(item)
    return new_arr_sentence


def generate_potential_definitions(sentence: str, abbreviation: str):
    abb_pattern = abbre_pattern(abbreviation)
    arr_sentence = separate_sentence(sentence, abbreviation)

    max_len = min(len(abb_pattern) + 5, len(abb_pattern) * 2)
    if abbreviation not in arr_sentence:
        return None, None
    idx_abb = arr_sentence.index(abbreviation)
    start_idx = (idx_abb - max_len) if (idx_abb - max_len) > 0 else 0
    end_idx = (idx_abb + max_len) if (idx_abb + max_len) < (len(arr_sentence) - 1) else (len(arr_sentence) - 1)

    before_abb = find_all_candidate(arr_sentence, abbreviation, start_idx, idx_abb)
    after_abb = find_all_candidate(arr_sentence, abbreviation, idx_abb + 1, end_idx + 1)
    return before_abb, after_abb


def formationRules_and_definition_patterns(sentence: str, abbreviation: str, candidates: list):
    if len(candidates) == 0:
        return '', [], []
    else:
        abb_pattern = abbre_pattern(abbreviation)
        formation_rules = []
        def_patterns = []
        arr_sentence = separate_sentence(sentence, abbreviation)
        for item in candidates:
            one_def = 'z'
            one_candidate_rule = []
            last_idx = -1
            for i in range(len(item)):
                if abb_pattern[i] == 'n':
                    one_candidate_rule.append((item[i][0], 'e'))
                    one_def += 'n'
                else:
                    if arr_sentence[item[i][0]] in stop_words:
                        one_def += 's'
                    else:
                        if last_idx != item[i][0]:
                            one_def += 'w'
                    tmp_n = item[i][1]
                    if tmp_n == 0:
                        one_candidate_rule.append((item[i][0], 'f'))
                    elif 0 < tmp_n < len(arr_sentence[item[i][0]]) - 1:
                        one_candidate_rule.append((item[i][0], 'i'))
                    elif tmp_n == len(arr_sentence[item[i][0]]) - 1:
                        one_candidate_rule.append((item[i][0], 'l'))
                    elif tmp_n == -1:
                        one_candidate_rule.append((item[i][0], 'e'))
                    elif tmp_n == -2:
                        one_candidate_rule.append((item[i][0], 'r'))
                last_idx = item[i][0]
            if one_def[1] != 's':
                formation_rules.append(one_candidate_rule)
                def_patterns.append(one_def[1:])
    return abb_pattern, formation_rules, def_patterns


def find_best_candidate(a_pattern: str, d_patterns: list, formation_rules: list, sentence: str, abbreviation: str):
    res = {}
    for i in range(len(d_patterns)):
        cons = 1
        len_abb = len(a_pattern)
        if len_abb == len(d_patterns[i]):
            cons = 1
        elif len_abb < len(d_patterns[i]):
            cons = 0.9
        elif len_abb > len(d_patterns[i]):
            cons = 0.8
        score = 0

        idx_list = []
        for item in formation_rules[i]:
            idx_list.append(item[0])
            if item[1] == 'f' or item[1] == 'e':
                score += 3
            elif item[1] == 'r':
                score += 2.5
            elif item[1] == 'i':
                score += 2
            elif item[1] == 'l':
                score += 1
        res[i] = score * cons - np.var(idx_list)
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    definition = ''
    score = -1
    for key in res.keys():
        tmp_definition = find_definition(sentence, formation_rules[key], abbreviation)
        if tmp_definition in abbreviation or abbreviation in tmp_definition:
            continue
        else:
            score = res[key]
            definition = tmp_definition
            break
    return definition, score


def find_definition(sentence: str, formation_rules: list, abbre: str):
    """

    :type formation_rules: object
    """

    arr_sentence = separate_sentence(sentence, abbre)
    abbreviation_idx = [i for i, e in enumerate(arr_sentence) if e == abbre]
    start_word = arr_sentence[formation_rules[0][0]]
    end_word = arr_sentence[formation_rules[-1][0]]
    appearances_start = [i for i, x in enumerate(arr_sentence) if x == start_word]
    appearances_end = [i for i, x in enumerate(arr_sentence) if x == end_word]
    tmp_indexes_start = [idx for idx in range(len(sentence)) if sentence.startswith(start_word, idx)]
    tmp_indexes_end = [idx for idx in range(len(sentence)) if sentence.startswith(end_word, idx)]
    min_start_idx = len(''.join(arr_sentence[0:max(appearances_start[0], 0)]))
    min_end_idx = len(''.join(arr_sentence[0:max(appearances_end[0], 0)]))
    indexes_start = [i for i in tmp_indexes_start if i >= min_start_idx]
    indexes_end = [i for i in tmp_indexes_end if i >= min_end_idx]
    idx_start = indexes_start[appearances_start.index(formation_rules[0][0])]
    idx_end = indexes_end[appearances_end.index(formation_rules[-1][0])] + len(end_word) - 1


    if idx_start - 1 > -1:
        if sentence[idx_start - 1] == '-':
            for i in range(idx_start - 1, -1, -1):
                if sentence[i] != ' ':
                    idx_start = i
                else:
                    break

    if idx_end + 1 < len(sentence):
        if sentence[idx_end + 1] == '-':
            for i in range(idx_end + 1, len(sentence)):
                if sentence[i] != ' ':
                    idx_end = i
                else:
                    break

    start_bracket_counts = 0
    end_bracket_counts = 0
    for i in range(idx_start - 1, -1, -1):
        if sentence[i] == '(':
            start_bracket_counts -= 1
        elif sentence[i] == ')':
            start_bracket_counts += 1
        if start_bracket_counts == -1:
            idx_start = i + 1
            break

    for i in range(idx_end + 1, len(sentence)):
        if sentence[i] == '(':
            end_bracket_counts += 1
        elif sentence[i] == ')':
            end_bracket_counts -= 1
        if end_bracket_counts == -1:
            idx_end = i - 1
            break

    output = sentence[idx_start:(idx_end + 1)].replace('\n', ' ').strip()
    if output != '':
        return output
    else:
        words = arr_sentence[formation_rules[0][0]:(formation_rules[-1][0] + 1)]
        output = ' '.join(words).replace('\n', ' ').strip()
        return output


def complete_abbreviations(abbs: list, sentence: str):
    dic_abbTimes = {}
    new_abbs = abbs.copy()
    for abb in abbs:
        if dic_abbTimes.get(abb) is None:
            dic_abbTimes[abb] = 1
        else:
            dic_abbTimes[abb] += 1
    start_idx_list = []
    stops = [',', '.', ';']
    for abb in abbs:
        indexes_start = [idx for idx in range(len(sentence)) if sentence.startswith(abb, idx)]
        start_idx_list.append(indexes_start[len(indexes_start) - dic_abbTimes[abb]])
        dic_abbTimes[abb] -= 1
    for j in range(len(start_idx_list)):
        idx_start = start_idx_list[j]
        idx_end = idx_start + len(abbs[j]) - 1
        start_bracket_counts = 0
        end_bracket_counts = 0
        new_idx_start = idx_start
        new_idx_end = idx_end
        for n in range(idx_start - 1, max(idx_start - 20, -1), -1):
            if sentence[n] in stops:
                break
            if sentence[n] == '(':
                start_bracket_counts -= 1
            elif sentence[n] == ')':
                start_bracket_counts += 1
            if start_bracket_counts == -1:
                new_idx_start = n + 1
                break
        for n in range(idx_end + 1, min(len(sentence), idx_end + 20)):
            if sentence[n] in stops:
                break
            if sentence[n] == '(':
                end_bracket_counts += 1
            elif sentence[n] == ')':
                end_bracket_counts -= 1
            if end_bracket_counts == -1:
                new_idx_end = n - 1
                break
        if new_idx_end != idx_end or new_idx_start != idx_start:
            new_abbs[j] = sentence[new_idx_start:(new_idx_end + 1)].strip()
    return new_abbs


def Hybrid_definition_mining(sentence: str, abbreviation: str) -> object:
    ls_can1, ls_can2 = generate_potential_definitions(sentence, abbreviation)
    if ls_can2 is None and ls_can1 is None:
        return '', -1
    a1, formation_rules1, definition_patterns1 = formationRules_and_definition_patterns(sentence, abbreviation, ls_can1)
    a2, formation_rules2, definition_patterns2 = formationRules_and_definition_patterns(sentence, abbreviation, ls_can2)
    if len(formation_rules1) + len(formation_rules2) == 0:
        return '', -1
    res_str, score = find_best_candidate(a1, (definition_patterns1 + definition_patterns2),
                                         (formation_rules1 + formation_rules2), sentence, abbreviation)
    return (res_str, round(score, 2))


class Candidate(str):
    def __init__(self, value):
        super().__init__()
        self.start = 0
        self.stop = 0

    def set_position(self, start, stop):
        self.start = start
        self.stop = stop

class AbbreviationExtractor:

    def __init__(self):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        self.log = logging.getLogger(__name__)

    def __conditions(self, candidate):
        r"""
        Based on Schwartz&Hearst

        2 <= len(str) <= 10
        len(tokens) <= 2
        re.search(r'\p{L}', str)
        str[0].isalnum()

        and extra:
        if it matches (\p{L}\.?\s?){2,}
        it is a good candidate.

        :param candidate: candidate abbreviation
        :return: True if this is a good candidate
        """
        LF_in_parentheses = False
        viable = True
        if re2.match(r'(\p{L}\.?\s?){2,}', candidate.lstrip()):
            viable = True
        if len(candidate) < 2 or len(candidate) > 10:
            viable = False
        if len(candidate.split()) > 2:
            viable = False
            LF_in_parentheses = True
        if candidate.islower():
            viable = False
        if not re2.search(r'\p{L}', candidate):
            viable = False
        if not candidate[0].isalnum():
            viable = False

        return viable
        
    def extract(self, paragraphs, config=None, pmcid=None):
        self.paragraphs = paragraphs
        self.config = config
        abbrev_json, Hybrid_scores, potential_abbreviations = self.get_abbreviation(paragraphs, config=config, pmcid=pmcid)
        return {
            'abbrev_json': abbrev_json, 
            'Hybrid_scores': Hybrid_scores, 
            'potential_abbreviations': potential_abbreviations,
        }

    def __re_find_abbreviation2(self, main_text, para_num, definition_dict):
        re_letter = r'\b[A-Z](?:[-_.///a-z]?[A-Z0-9α-ωΑ-Ω])+[a-z]*\b'
        re_digit = r'\b[0-9](?:[-_.,:///]?[a-zA-Z0-9α-ωΑ-Ω])+[A-Z]{1}(?:[,:&_.-///]?[a-zA-Z0-9α-ωΑ-Ω])*\b'
        re_symble = r'[[α-ωΑ-Ω][0-9A-Z](?:[-_.///]?[A-Z0-9])+[]](?:[_&-.///]?[A-Z0-9])+\b'

        temp = {}
        sentence_iterator = enumerate(self.__yield_lines_from_doc(main_text))
        for i, sentence in sentence_iterator:
            res1 = re2.findall(re_letter, sentence)
            res2 = re2.findall(re_digit, sentence)
            res3 = re2.findall(re_symble, sentence)
            if len(res1) + len(res2) + len(res3) > 0:
                all_abb = list(set(res1 + res2 + res3))
                for abbre in all_abb:
                    previously_found = definition_dict.get(abbre)
                    definition_score_tuple = Hybrid_definition_mining(sentence, abbre)
                    if temp.get(abbre) is not None:
                        if definition_score_tuple not in temp[abbre]:
                            temp[abbre].append(definition_score_tuple)
                    elif previously_found is None:
                        temp[abbre] = [definition_score_tuple]
                    else:
                        if definition_score_tuple not in previously_found:
                            previously_found.append(definition_score_tuple)
                            temp[abbre] = previously_found
        return temp

    def __get_definition(self, candidate, sentence):
        """
        Takes a candidate and a sentence and returns the definition candidate.

        The definition candidate is the set of tokens (in front of the candidate)
        that starts with a token starting with the first character of the candidate

        :param candidate: candidate abbreviation
        :param sentence: current sentence (single line from input file)
        :return: candidate definition for this abbreviation
        """
        # Take the tokens in front of the candidate
        tokens = re2.split(r'[\s\-]+', sentence[:candidate.start - 2].lower())
        # the char that we are looking for
        key = candidate[0].lower()

        # Count the number of tokens that start with the same character as the candidate
        first_chars = [t[0] for t in filter(None, tokens)]

        definition_freq = first_chars.count(key)
        candidate_freq = candidate.lower().count(key)

        # Look for the list of tokens in front of candidate that
        # have a sufficient number of tokens starting with key
        if candidate_freq <= definition_freq:
            # we should at least have a good number of starts
            count = 0
            start = 0
            start_index = len(first_chars) - 1
            while count < candidate_freq:
                if abs(start) > len(first_chars):
                    raise ValueError("candidate {} not found".format(candidate))
                start -= 1
                # Look up key in the definition
                try:
                    start_index = first_chars.index(key, len(first_chars) + start)
                except ValueError:
                    pass

                # Count the number of keys in definition
                count = first_chars[start_index:].count(key)

            # We found enough keys in the definition so return the definition as a definition candidate
            start = len(' '.join(tokens[:start_index]))
            stop = candidate.start - 1
            candidate = sentence[start:stop]

            # Remove whitespace
            start = start + len(candidate) - len(candidate.lstrip())
            stop = stop - len(candidate) + len(candidate.rstrip())
            candidate = sentence[start:stop]

            new_candidate = Candidate(candidate)
            new_candidate.set_position(start, stop)
            return new_candidate

        else:
            raise ValueError('There are less keys in the tokens in front of candidate than there are in the candidate')

    def __select_definition(self, definition, abbrev):
        """
        Takes a definition candidate and an abbreviation candidate
        and returns True if the chars in the abbreviation occur in the definition

        Based on
        A simple algorithm for identifying abbreviation definitions in biomedical texts, Schwartz & Hearst
        :param definition: candidate definition
        :param abbrev: candidate abbreviation
        :return:
        """

        if len(definition) < len(abbrev):
            raise ValueError('Abbreviation is longer than definition')

        if abbrev in definition.split():
            raise ValueError('Abbreviation is full word of definition')

        s_index = -1
        l_index = -1

        while 1:
            try:
                long_char = definition[l_index].lower()
            except IndexError:
                raise

            short_char = abbrev[s_index].lower()

            if not short_char.isalnum():
                s_index -= 1

            if s_index == -1 * len(abbrev):
                if short_char == long_char:
                    if l_index == -1 * len(definition) or not definition[l_index - 1].isalnum():
                        break
                    else:
                        l_index -= 1
                else:
                    l_index -= 1
                    if l_index == -1 * (len(definition) + 1):
                        raise ValueError("definition {} was not found in {}".format(abbrev, definition))

            else:
                if short_char == long_char:
                    s_index -= 1
                    l_index -= 1
                else:
                    l_index -= 1

        new_candidate = Candidate(definition[l_index:len(definition)])
        new_candidate.set_position(definition.start, definition.stop)
        definition = new_candidate

        tokens = len(definition.split())
        length = len(abbrev)

        if tokens > min([length + 5, length * 2]):
            raise ValueError("did not meet min(|A|+5, |A|*2) constraint")

        # Do not return definitions that contain unbalanced parentheses
        if definition.count('(') != definition.count(')'):
            raise ValueError("Unbalanced parentheses not allowed in a definition")

        return definition

    def __best_candidates(self, sentence):
        """
        :param sentence: line read from input file
        :return: a Candidate iterator
        """

        if '(' in sentence:
            # Check some things first
            if sentence.count('(') != sentence.count(')'):
                raise ValueError("Unbalanced parentheses: {}".format(sentence))

            if sentence.find('(') > sentence.find(')'):
                raise ValueError("First parentheses is right: {}".format(sentence))

            close_index = -1
            while 1:
                # Look for open parenthesis. Need leading whitespace to avoid matching mathematical and chemical formulae
                open_index = sentence.find(' (', close_index + 1)

                if open_index == -1: break

                # Advance beyond whitespace
                open_index += 1

                # Look for closing parentheses
                close_index = open_index + 1
                open_count = 1
                skip = False
                while open_count:
                    try:
                        char = sentence[close_index]
                    except IndexError:
                        # We found an opening bracket but no associated closing bracket
                        # Skip the opening bracket
                        skip = True
                        break
                    if char == '(':
                        open_count += 1
                    elif char in [')', ';', ':']:
                        open_count -= 1
                    close_index += 1

                if skip:
                    close_index = open_index + 1
                    continue

                # Output if conditions are met
                start = open_index + 1
                stop = close_index - 1
                candidate = sentence[start:stop]

                # Take into account whitespace that should be removed
                start = start + len(candidate) - len(candidate.lstrip())
                stop = stop - len(candidate) + len(candidate.rstrip())
                candidate = sentence[start:stop]

                if self.__conditions(candidate):
                    new_candidate = Candidate(candidate)
                    new_candidate.set_position(start, stop)
                    yield new_candidate


    def __yield_lines_from_doc(self, doc_text):
        for line in doc_text.split("."):
            yield line.strip()

    def __extract_abbreviation_definition_pairs(self, doc_text=None, most_common_definition=False,
                                                first_definition=False, all_definition=True):
        abbrev_map = dict()
        list_abbrev_map = defaultdict(list)
        counter_abbrev_map = dict()
        omit = 0
        written = 0
        sentence_iterator = enumerate(self.__yield_lines_from_doc(doc_text))

        collect_definitions = False
        if most_common_definition or first_definition or all_definition:
            collect_definitions = True

        for i, sentence in sentence_iterator:
            # Remove any quotes around potential candidate terms
            clean_sentence = re2.sub(r'([(])[\'"\p{Pi}]|[\'"\p{Pf}]([);:])', r'\1\2', sentence)
            try:
                for candidate in self.__best_candidates(clean_sentence):
                    try:
                        definition = self.__get_definition(candidate, clean_sentence)
                    except (ValueError, IndexError) as e:
                        self.log.debug("{} Omitting candidate {}. Reason: {}".format(i, candidate, e.args[0]))
                        omit += 1
                    else:
                        try:
                            definition = self.__select_definition(definition, candidate)
                        except (ValueError, IndexError) as e:
                            self.log.debug(
                                "{} Omitting definition {} for candidate {}. Reason: {}".format(i, definition,
                                                                                                candidate, e.args[0]))
                            omit += 1
                        else:
                            # Either append the current definition to the list of previous definitions ...
                            if collect_definitions:
                                list_abbrev_map[candidate].append(definition)
                            else:
                                # Or update the abbreviations map with the current definition
                                abbrev_map[candidate] = definition
                            written += 1
            except (ValueError, IndexError) as e:
                self.log.debug("{} Error processing sentence {}: {}".format(i, sentence, e.args[0]))
        self.log.debug("{} abbreviations detected and kept ({} omitted)".format(written, omit))

        # Return most common definition for each term
        if collect_definitions:
            if most_common_definition:
                # Return the most common definition for each term
                for k, v in list_abbrev_map.items():
                    counter_abbrev_map[k] = Counter(v).most_common(1)[0][0]
            elif first_definition:
                # Return the first definition for each term
                for k, v in list_abbrev_map.items():
                    counter_abbrev_map[k] = v
            elif all_definition:
                for k, v in list_abbrev_map.items():
                    counter_abbrev_map[k] = v
            return counter_abbrev_map

        # Or return the last encountered definition for each term
        return abbrev_map

    
    def __extract_abbreviation(self, main_text):
        pairs = self.__extract_abbreviation_definition_pairs(doc_text=main_text, most_common_definition=True)

        return pairs


    def get_abbreviation(self, paragraphs, pmcid=None, config=None):
        all_abbreviations = {}
        hybrid_all_abbreviations = {}
        para_num = 1

        for main_text in paragraphs:
            pairs = self.__extract_abbreviation(main_text)
            all_abbreviations.update(pairs)
            hybrid_pairs = self.__re_find_abbreviation2(main_text, para_num, hybrid_all_abbreviations)
            hybrid_all_abbreviations.update(hybrid_pairs)
            para_num += 1

        abbrev_json = {}
        Hybrid_scores = {}
        potential_abbreviations = {}

        for key in all_abbreviations:
            clean_def = all_abbreviations[key].replace("\n", " ")
            if key in abbrev_json:
                if clean_def in abbrev_json[key].keys():
                    abbrev_json[key][clean_def].append("fulltext")
                else:
                    abbrev_json[key][clean_def] = ["fulltext"]
            else:
                abbrev_json[key] = {clean_def: ["fulltext"]}

        for one_abb in hybrid_all_abbreviations:
            definitions = hybrid_all_abbreviations[one_abb]
            if definitions is None:
                continue
            if len(definitions) == 1 and definitions[0][1] == -1:
                potential_abbreviations[one_abb] = 'Not Found Yet'
                continue
            Hybrid_scores[one_abb] = []
            if one_abb in abbrev_json:
                for definition_score in definitions:
                    if definition_score[1] == -1:
                        continue
                    Hybrid_scores[one_abb].append(definition_score)
                    if definition_score[0] in abbrev_json[one_abb].keys():
                        abbrev_json[one_abb][definition_score[0]].append("HybriDK+")
                    else:
                        abbrev_json[one_abb][definition_score[0]] = ["HybriDK+"]
            else:
                abbrev_json[one_abb] = {}
                for definition_score in definitions:
                    if definition_score[1] == -1:
                        continue
                    Hybrid_scores[one_abb].append(definition_score)
                    abbrev_json[one_abb][definition_score[0]] = ["HybriDK+"]

        return abbrev_json, Hybrid_scores, potential_abbreviations

    def biocify_abbreviations(self, abbreviations, Hybrid_scores, potential_abbreviations, doc_id, file_format="json"):
        offset = 0
        template = {
            "source": "Auto-CORPus (abbreviations)",
            "date": f'{datetime.today().strftime("%Y%m%d")}',
            "key": "autocorpus_abbreviations.key",
            "documents": [
                {
                    "id": doc_id,
                    "inputfile": doc_id + "." + file_format,
                    "passages": []
                }
            ]
        }

        hybrid_template = copy.deepcopy(template)
        potentialAbb_template = copy.deepcopy(template)
        passages = template["documents"][0]["passages"]
        HybridScore_passages = hybrid_template["documents"][0]["passages"]
        potentialAbbre_passages = potentialAbb_template["documents"][0]["passages"]

        for short in abbreviations.keys():
            counter = 1
            shortTemplate = {
                "text_short": short
            }
            for long in abbreviations[short].keys():
                shortTemplate[F"text_long_{counter}"] = long.replace("\n", " ")
                shortTemplate[F"extraction_algorithm_{counter}"] = ", ".join(abbreviations[short][long])
                counter += 1
            passages.append(shortTemplate)

        for short in Hybrid_scores.keys():
            counter = 1
            score_shortTemplate = {
                "text_short": short
            }
            for long in Hybrid_scores[short]:
                score_shortTemplate[F"text_long_{counter}"] = long[0]
                score_shortTemplate[F"Hybrid_score_{counter}"] = long[1]
                counter += 1
            HybridScore_passages.append(score_shortTemplate)

        for short in potential_abbreviations.keys():
            counter = 1
            potential_shortTemplate = {
                "text_short": short,
                "text_long": potential_abbreviations[short]
            }
            potentialAbbre_passages.append(potential_shortTemplate)

        return template, hybrid_template, potentialAbb_template



