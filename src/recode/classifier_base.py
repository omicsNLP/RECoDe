import re
from collections import Counter
from openai import OpenAI


class ClassifierBase:
    def __init__(self, current_labels, current_label_to_label, **kwargs):
        self.current_labels = current_labels
        self.current_label_to_label = current_label_to_label

        self.num_return_sequences = kwargs.get("num_return_sequences", 3)
        self.num_trials = kwargs.get("num_trials", 3)
        self.num_reasoning_trials = kwargs.get("num_reasoning_trials", 5)
        self.num_max_tokens = kwargs.get("num_max_tokens", 256)
        self.do_sample = kwargs.get("do_sample", True)
        self.temperature = kwargs.get("temperature", 0.2)
        self.top_p = kwargs.get("top_p", 0.8)
        self.top_k = kwargs.get("top_k", 3)
        self.is_multi_response = kwargs.get("is_multi_response", False)

        self.pattern = re.compile(
            "|".join(
                [
                    rf"\({chr(65+i)}\) {label}"
                    for i, label in enumerate(self.current_labels)
                ]
            )
        )

    def prompt(self, entity1, entity2, sentence):
        raise NotImplementedError

    def extract_answer(self, response):
        if response is None:
            return "Unknown"
        matches = self.pattern.findall(response)
        return matches[-1] if matches else "Unknown"

    def generate_multi_responses(
        self, system_prompt, user_prompt, base_url, model_name, api_key
    ):
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.num_max_tokens,
            n=self.top_k,
            model=model_name,
        )

        if completion is None or completion.choices is None:
            return None
        else:
            results = []
            for choice in completion.choices:
                if choice.message is None or choice.message.content is None:
                    pass
                else:
                    results.append(choice.message.content.strip())
            return results

    def classify(self, entity1, entity2, sentence, base_url, model_name, api_key):
        system_prompt, user_prompt = self.prompt(entity1, entity2, sentence)

        preds = []
        responses = []

        max_retries = 3
        for trial in range(self.num_trials):
            generated_responses = None
            for retry in range(max_retries):
                generated_responses = self.generate_multi_responses(
                    system_prompt, user_prompt, base_url, model_name, api_key
                )
                if generated_responses:
                    break

            if not generated_responses:
                continue

            for response in generated_responses:
                parsed_answer = self.extract_answer(response)
                preds.append(parsed_answer)
                responses.append(response)

            if trial == 0:
                if len(set(preds)) == 1:
                    break

        preds_counter = Counter(preds)

        if not preds_counter:
            majority_pred = None
        else:
            majority_pred = self.current_label_to_label.get(
                preds_counter.most_common(1)[0][0], None
            )

        res = {
            "entity1": entity1,
            "entity2": entity2,
            "sentence": sentence,
            "responses": responses,
            "preds": preds,
            "majority_pred": majority_pred,
            "pred_cnt": dict(preds_counter.most_common()),
        }
        return res
