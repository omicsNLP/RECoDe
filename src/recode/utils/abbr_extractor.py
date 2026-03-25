import json
import os

from tqdm import tqdm

import recode

from .plain_abbr import AbbreviationExtractor


class AbbrExtractor():
    def __init__(self, input_dir, output_dir, ver='plain_abbr'):
        self.input_dir = input_dir
        self.input_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

        if len(self.input_files) == 0:
            raise ValueError(f"No files in the input directory {input_dir}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.exisiting_output_files = os.listdir(output_dir)

        self.output_dir = output_dir
        self.ver = ver
        if ver == 'plain_abbr':
            self.abbr = AbbreviationExtractor()

    def extract_abbr_and_save_to_dir(self):
        for file in tqdm(self.input_files):
            if file in self.exisiting_output_files:
                continue

            instance = recode.reader.parse_json_instance(
                os.path.join(self.input_dir, file)
            )
            passages = [passage.text for passage in instance.documents[0].passages]

            abbr_dict, hybrid_dict, potential_dict = self.abbr.get_abbreviation(
                passages, pmcid=instance.infons.pmcid
            )

            abbrs = []
            for abbr_txt, long_form_dict in abbr_dict.items():
                abbrs.append(recode.Abbreviation(
                    short_form=abbr_txt,
                    long_forms=[recode.LongForm(
                        text=long_form_txt,
                        extraction_algorithms=extraction_algorithms
                    ) for long_form_txt, extraction_algorithms in long_form_dict.items()]
                ))

            hybrids = []
            for abbr_txt, hybrid_scores in hybrid_dict.items():
                hybrid_abbrs = []
                for hybrid_txt, score in hybrid_scores:
                    hybrid_abbrs.append(recode.LongForm(
                        text=hybrid_txt,
                        score=score
                    ))
                hybrids.append(recode.Abbreviation(
                    short_form=abbr_txt,
                    long_forms=hybrid_abbrs
                ))

            potentials = []
            for abbr_txt, potential_str in potential_dict.items():
                potentials.append(recode.Abbreviation(
                    short_form=abbr_txt,
                    long_forms=[recode.LongForm(text=potential_str)]
                ))

            instance.infons.abbreviations = abbrs
            instance.infons.hybrid_abbreviations = hybrids
            instance.infons.potential_abbreviations = potentials

            with open(os.path.join(self.output_dir, file), 'w') as f:
                json.dump(instance.dict(), f, indent=2)
