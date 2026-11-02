from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.utils.data import Dataset, random_split


DATA_DIR = Path(r"NaturalLanguageProcessing\Seq2Seq\data")


class BaseSeq2SeqDataset(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_train_dataset(self) -> Dataset:
        pass

    @abstractmethod
    def get_test_dataset(self) -> Dataset:
        pass

    @abstractmethod
    def get_vocab_size(self) -> int:
        pass

    @abstractmethod
    def get_id_to_char(self) -> dict:
        pass


class CharSeq2SeqDataset(Dataset):
    def __init__(self, questions, answers, char_to_id, reverse=False):
        self.questions = questions
        self.answers = answers
        self.char_to_id = char_to_id
        self.reverse = reverse

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        question = self.questions[index]
        answer = self.answers[index]
        if self.reverse:
            question = question[::-1]
            answer = answer[0] + answer[1:][::-1]
        x = [self.char_to_id[c] for c in question]
        t = [self.char_to_id[c] for c in answer]
        return torch.tensor(x, dtype=torch.long), torch.tensor(t, dtype=torch.long)


def load_pairs(file_path):
    questions, answers = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            idx = line.find("_")
            questions.append(line[:idx])
            answers.append(line[idx:-1])
    return questions, answers


def build_vocab(questions, answers):
    chars = sorted(set("".join(questions) + "".join(answers)))
    char_to_id = {c: i for i, c in enumerate(chars)}
    id_to_char = {i: c for c, i in char_to_id.items()}
    return char_to_id, id_to_char


class CharSeq2SeqCorpus(BaseSeq2SeqDataset):
    def __init__(self, file_name, test_rate=0.1, seed=42, limit=None, reverse=False):
        super().__init__()
        self.questions, self.answers = load_pairs(DATA_DIR / file_name)
        if limit is not None:
            self.questions = self.questions[:limit]
            self.answers = self.answers[:limit]
        self.char_to_id, self.id_to_char = build_vocab(self.questions, self.answers)
        self.test_rate = test_rate
        self.seed = seed
        self.reverse = reverse

    def _split(self):
        total = len(self.questions)
        test_size = int(total * self.test_rate)
        train_size = total - test_size
        return random_split(range(total), [train_size, test_size],
                            generator=torch.Generator().manual_seed(self.seed))

    def _subset(self, indices):
        return CharSeq2SeqDataset(
            [self.questions[i] for i in indices],
            [self.answers[i] for i in indices],
            self.char_to_id,
            reverse=self.reverse,
        )

    def get_train_dataset(self) -> Dataset:
        train_idx, _ = self._split()
        return self._subset(train_idx)

    def get_test_dataset(self) -> Dataset:
        _, test_idx = self._split()
        return self._subset(test_idx)

    def get_vocab_size(self) -> int:
        return len(self.char_to_id)

    def get_id_to_char(self) -> dict:
        return self.id_to_char

    def get_start_id(self) -> int:
        return self.char_to_id["_"]

    def get_answer_len(self) -> int:
        return len(self.answers[0]) - 1


class AdditionCorpus(CharSeq2SeqCorpus):
    def __init__(self, **kwargs):
        kwargs.setdefault("reverse", True)
        super().__init__("addition.txt", **kwargs)


class DateCorpus(CharSeq2SeqCorpus):
    def __init__(self, **kwargs):
        kwargs.setdefault("reverse", False)
        super().__init__("date.txt", **kwargs)


CORPORA = {
    "addition": AdditionCorpus,
    "date": DateCorpus,
}


def build_corpus(name, **kwargs):
    if name not in CORPORA:
        raise ValueError(f"unknown corpus {name!r}, expected one of {list(CORPORA)}")
    return CORPORA[name](**kwargs)


if __name__ == "__main__":
    for name in CORPORA:
        ds = build_corpus(name)
        train, test = ds.get_train_dataset(), ds.get_test_dataset()
        id_to_char = ds.get_id_to_char()
        x, t = train[0]
        print(f"[{name}] vocab={ds.get_vocab_size()} train={len(train)} test={len(test)} "
              f"reverse={ds.reverse}")
        print(f"    x={''.join(id_to_char[i.item()] for i in x)!r}")
        print(f"    t={''.join(id_to_char[i.item()] for i in t)!r}")
        print(f"    chars={''.join(sorted(ds.char_to_id))!r}")
