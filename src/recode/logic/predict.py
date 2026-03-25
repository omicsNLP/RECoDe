from .c1first import C1First
from .c2asso import C2Asso
from .c3corr import C3Corr
from .c4middleasso import C4MiddleAsso
from .c5assocausal import C5AssoCausal
from .c6containsubsti import C6ContainSubsti
from .c7pos_corr_reflection import C7PositiveCorrReflectionWithContains

labels = [
    "association",
    "increaseAssociation",
    "decreaseAssociation",
    "positiveCorrelation",
    "negativeCorrelation",
    "consists",
    "causalEffect",
    "substitution",
    "NoAssociation",
    "Unrelated",
]
special_tokens = ["<entity1>", "</entity1>", "<entity2>", "</entity2>"] + labels


# row['Node1_str'], row['Node2_str'], row['transformed_text'],base_url=args.base_url, model_name=args.model_name, api_key=args.api_key, temperature=0.2, top_p=0.8, num_max_tokens=512, num_trials=3)
def predict(
    entity1,
    entity2,
    sentence,
    base_url,
    model_name,
    api_key,
    temperature,
    top_p,
    num_trials,
    num_max_tokens,
):

    def _get_pred(pred):
        p = pred["majority_pred"]
        if p is None:
            return "association"
        return p

    c1_pred = C1First(
        temperature=temperature,
        top_p=top_p,
        num_trials=num_trials,
        num_max_tokens=num_max_tokens,
    ).classify(entity1, entity2, sentence, base_url, model_name, api_key)

    if _get_pred(c1_pred) is None:
        return "association"
    if _get_pred(c1_pred) in ["NoAssociation", "Unrelated"]:
        return _get_pred(c1_pred)

    c2_pred = C2Asso(
        temperature=temperature,
        top_p=top_p,
        num_trials=num_trials,
        num_max_tokens=num_max_tokens,
    ).classify(entity1, entity2, sentence, base_url, model_name, api_key)

    if _get_pred(c2_pred) is None:
        return "association"

    if _get_pred(c2_pred) == "MiddleAssociation":
        c4_pred = C4MiddleAsso(
            temperature=temperature,
            top_p=top_p,
            num_trials=num_trials,
            num_max_tokens=num_max_tokens,
        ).classify(entity1, entity2, sentence, base_url, model_name, api_key)

        if _get_pred(c4_pred) in ["decreaseAssociation", "increaseAssociation"]:
            return _get_pred(c4_pred)
        elif _get_pred(c4_pred) == "GeneralAssociation":
            c5_pred = C5AssoCausal(
                temperature=temperature,
                top_p=top_p,
                num_trials=num_trials,
                num_max_tokens=num_max_tokens,
            ).classify(entity1, entity2, sentence, base_url, model_name, api_key)
            return _get_pred(c5_pred)

    elif _get_pred(c2_pred) == "MiddleCorrelation":
        c3_pred = C3Corr(
            temperature=temperature,
            top_p=top_p,
            num_trials=num_trials,
            num_max_tokens=num_max_tokens,
        ).classify(entity1, entity2, sentence, base_url, model_name, api_key)

        if _get_pred(c3_pred) == "positiveCorrelation":
            c7_pred = C7PositiveCorrReflectionWithContains(
                temperature=temperature,
                top_p=top_p,
                num_trials=num_trials,
                num_max_tokens=num_max_tokens,
            ).classify(entity1, entity2, sentence, base_url, model_name, api_key)
            return _get_pred(c7_pred)
        else:
            return _get_pred(c3_pred)
    elif _get_pred(c2_pred) == "Contains":
        c6_pred = C6ContainSubsti(
            temperature=temperature,
            top_p=top_p,
            num_trials=num_trials,
            num_max_tokens=num_max_tokens,
        ).classify(entity1, entity2, sentence, base_url, model_name, api_key)
        return _get_pred(c6_pred)

    return "association"
