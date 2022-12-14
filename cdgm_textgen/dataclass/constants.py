from fairseq.dataclass import ChoiceEnum

GENERATOR_TYPES = ChoiceEnum(["seq2seq", "nat", "custom"])
