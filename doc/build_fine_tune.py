import argparse
import json
import os
import sys
import textwrap
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import List

sys.path.append(os.getcwd())

import numpy as np
import pandas as pd


class BlockType(Enum):
    PROSE = 'prose'
    CODE = 'code'


class Role(Enum):
    USER = 'user'
    ASSISTANT = 'assistant'


@dataclass(frozen=True)
class MessageBlock:
    type: BlockType
    content: str


@dataclass(frozen=True)
class Message:
    role: Role
    blocks: List[MessageBlock]

    def to_text(self) -> str:
        parts = []
        for block in self.blocks:
            if block.type == BlockType.PROSE:
                parts.append(block.content.strip())
            elif block.type == BlockType.CODE:
                parts.append(f'```python\n{block.content.strip()}\n```')
        return '\n\n'.join(parts)

    def get_code_blocks(self) -> List[str]:
        return [b.content for b in self.blocks if b.type == BlockType.CODE]


@dataclass(frozen=True)
class Messages:
    user: Message
    assistant: Message

    def to_jsonl_line(self) -> str:
        return json.dumps(
            {
                'messages': [
                    {'role': self.user.role.value, 'content': self.user.to_text()},
                    {
                        'role': self.assistant.role.value,
                        'content': self.assistant.to_text(),
                    },
                ]
            }
        )

    def to_markdown(self, i: int) -> str:
        return (
            f'### Example {i}\n\n'
            f'**User:**\n\n{self.user.to_text()}\n\n'
            f'**Assistant:**\n\n{self.assistant.to_text()}\n\n---'
        )

    def validate_code(self):
        import static_frame as sf

        g = globals()
        l = locals()
        g['sf'] = sf
        g['np'] = np
        g['pd'] = pd

        for code in self.assistant.get_code_blocks():
            c = textwrap.dedent(code)
            exec(c, g, l)  # noqa: S102


# -------------------------------------------------------------------------------
@dataclass
class Corpus:
    examples: List[Messages] = field(default_factory=list)

    def write_jsonl(self, path: Path):
        path.write_text(
            '\n'.join(m.to_jsonl_line() for m in self.examples), encoding='utf-8'
        )

    def write_markdown(self, path: Path):
        path.write_text(
            '\n\n'.join(m.to_markdown(i + 1) for i, m in enumerate(self.examples)),
            encoding='utf-8',
        )

    def validate_all(self):
        fail = 0
        for i, msg in enumerate(self.examples, 1):
            print(f'📋 Example {i}')
            try:
                msg.validate_code()
                print(f'✅ Example {i} code validated')
            except Exception as e:
                print(f'❌ Example {i} code execution failed\n{e}')
                fail += 1
        if fail:
            raise RuntimeError(f'Found {fail} failures.')


EXAMPLES = [
    Messages(
        user=Message(
            Role.USER,
            [
                MessageBlock(BlockType.PROSE, 'Can I use `.rolling()` in StaticFrame?'),
                MessageBlock(
                    BlockType.PROSE,
                    'Or, how do I compute a rolling median of a Frame in StaticFrame?',
                ),
            ],
        ),
        assistant=Message(
            Role.ASSISTANT,
            [
                MessageBlock(
                    BlockType.PROSE,
                    "StaticFrame does not support a `.rolling()` method like Pandas. Instead, use `iter_window_array(size=...)` combined with `.reduce.from_map_func(...)`. Here's a complete example using an `IndexDate` and multiple columns:",
                ),
                MessageBlock(
                    BlockType.CODE,
                    """\
    import static_frame as sf
    import numpy as np

    index = sf.IndexDate.from_date_range('2020-01-01', '2020-01-07')
    f1 = sf.Frame.from_records(
        [
            [1, 10],
            [2, 20],
            [3, 30],
            [4, 40],
            [5, 50],
            [6, 60],
            [7, 70],
        ],
        index=index,
        columns=('A', 'B'),
    )

    f2 = (
        f1.iter_window_array(size=3)
         .reduce.from_map_func(np.mean)
         .to_frame(index_constructor=sf.IndexAutoConstructorFactory)
    )
    print(f2)
    """,
                ),
            ],
        ),
    ),
    # ---------------------------------------------------------------------------
    Messages(
        user=Message(
            Role.USER,
            [
                MessageBlock(
                    BlockType.PROSE,
                    'How do I use `IndexDate.from_range()` to get 10 days starting from 2020-01-01?',
                )
            ],
        ),
        assistant=Message(
            Role.ASSISTANT,
            [
                MessageBlock(
                    BlockType.PROSE,
                    (
                        '`IndexDate.from_range()` is not a valid method in StaticFrame.\n\n'
                        'In Pandas, you might write:'
                    ),
                ),
                MessageBlock(
                    BlockType.CODE,
                    """\
    import pandas as pd

    index = pd.date_range('2020-01-01', periods=10, freq='D')
    print(index)
    """,
                ),
                MessageBlock(
                    BlockType.PROSE,
                    (
                        'In StaticFrame, use `IndexDate.from_date_range(start, stop, step=1)`, where `stop` is inclusive. '
                        'To create 10 days starting from 2020-01-01:'
                    ),
                ),
                MessageBlock(
                    BlockType.CODE,
                    """\
    import static_frame as sf

    index = sf.IndexDate.from_date_range('2020-01-01', '2020-01-10', step=1)
    print(index)
    """,
                ),
            ],
        ),
    ),
    # ---------------------------------------------------------------------------
    Messages(
        user=Message(
            Role.USER,
            [
                MessageBlock(
                    BlockType.PROSE,
                    "Can I do `df['col'] = df['col'] * 2` in StaticFrame?",
                )
            ],
        ),
        assistant=Message(
            Role.ASSISTANT,
            [
                MessageBlock(
                    BlockType.PROSE,
                    (
                        'StaticFrame objects are immutable. You cannot modify a column in-place like in Pandas. '
                        'Instead, construct a new Frame using `.assign`. You can assign a column directly, or apply a function. '
                        'Here are two equivalent approaches:'
                    ),
                ),
                MessageBlock(
                    BlockType.CODE,
                    """\
    import static_frame as sf

    f = sf.Frame.from_records(
        [
            [2, 4],
            [3, 6],
            [4, 8],
        ],
        columns=('A', 'B'),
    )

    # Approach 1: direct column reassignment
    f1 = f.assign['A'](f['A'] * 2)
    print(f1)

    # Approach 2: apply a function to column 'A'
    f2 = f.assign['A'].apply(lambda s: s * 2)
    print(f2)
    """,
                ),
            ],
        ),
    ),
    # ---------------------------------------------------------------------------
    Messages(
        user=Message(
            Role.USER,
            [
                MessageBlock(
                    BlockType.PROSE,
                    "In Pandas I use `.rename(columns={...})` to change column names. What's the StaticFrame equivalent, and how is it different from `.relabel` and `.reindex`?",
                )
            ],
        ),
        assistant=Message(
            Role.ASSISTANT,
            [
                MessageBlock(
                    BlockType.PROSE, ('In Pandas, you rename column labels like this:')
                ),
                MessageBlock(
                    BlockType.CODE,
                    """\
    import pandas as pd

    df = pd.DataFrame([[1, 2]], columns=["A", "B"])
    df_renamed = df.rename(columns={"A": "X"})
    print(df_renamed)
    """,
                ),
                MessageBlock(
                    BlockType.PROSE,
                    ('In StaticFrame, the equivalent is `.relabel(columns={...})`:\n'),
                ),
                MessageBlock(
                    BlockType.CODE,
                    """\
    import static_frame as sf

    f = sf.Frame.from_records([[1, 2]], columns=("A", "B"))
    f_relabel = f.relabel(columns={"A": "X"})
    print(f_relabel)
    """,
                ),
                MessageBlock(
                    BlockType.PROSE,
                    (
                        'Important distinction:\n\n'
                        '- `.relabel(...)` is for relabelling via a mapping or equal-sized iterable.\n'
                        '- `.reindex(...)` is for reordering, adding, or removing axis labels, potentially inserting missing entries.\n'
                    ),
                ),
                MessageBlock(
                    BlockType.CODE,
                    """\
    f2 = sf.Frame.from_records(
        [[10], [20]],
        index=("r1", "r2"),
        columns=("A",),
    )

    # Renaming a label
    print(f2.relabel(index={"r1": "row1"}))

    # Reindexing to new index, inserting NaNs for missing labels
    print(f2.reindex(index=["r2", "r3"]))

    # Selecting a subset of rows
    print(f2.loc[["r2"]])  # This is the right way to project a subset
    """,
                ),
            ],
        ),
    ),
    # ---------------------------------------------------------------------------
    Messages(
        user=Message(
            Role.USER,
            [
                MessageBlock(
                    BlockType.PROSE,
                    'In Pandas, I can set a name on a Series or Index. Can I do the same in StaticFrame, and what happens if I call `.rename()` on a Frame?',
                )
            ],
        ),
        assistant=Message(
            Role.ASSISTANT,
            [
                MessageBlock(
                    BlockType.PROSE,
                    (
                        'Yes — in StaticFrame, **all major containers** (including `Frame`) support a `.name` attribute.\n\n'
                        'If you call `.rename("MyName")` with no axis argument, it sets the name of the `Frame` object itself, not any axis or label.'
                    ),
                ),
                MessageBlock(
                    BlockType.CODE,
                    """\
    import static_frame as sf

    f = sf.Frame.from_records([[1, 2]], columns=("A", "B"))
    print(f.name)  # None by default

    f_named = f.rename(\"MyFrame\")
    print(f_named.name)  # Now has name 'MyFrame'
    """,
                ),
                MessageBlock(
                    BlockType.PROSE,
                    (
                        'This is different from Pandas, where only `Series` and `Index` have a `.name` attribute.\n'
                        "StaticFrame's `name` is useful for identification and tracking through pipelines."
                    ),
                ),
            ],
        ),
    ),
    # ---------------------------------------------------------------------------
    Messages(
        user=Message(
            Role.USER,
            [
                MessageBlock(
                    BlockType.PROSE,
                    'In Pandas, I use `.groupby("A").mean()` to group rows by column A and take the mean. How do I do that in StaticFrame?',
                )
            ],
        ),
        assistant=Message(
            Role.ASSISTANT,
            [
                MessageBlock(
                    BlockType.PROSE,
                    ('In Pandas, grouping and aggregating by a column looks like this:'),
                ),
                MessageBlock(
                    BlockType.CODE,
                    """\
    import pandas as pd

    df = pd.DataFrame({
        "A": ["x", "x", "y", "y"],
        "B": [10, 20, 30, 40],
        "C": [1, 2, 3, 4],
    })

    print(df.groupby("A").mean())
    """,
                ),
                MessageBlock(
                    BlockType.PROSE,
                    (
                        'In StaticFrame, there are several ways to achieve the same result:\n\n'
                        '**1. Manual iteration with `from_items`**'
                    ),
                ),
                MessageBlock(
                    BlockType.CODE,
                    """\
    import static_frame as sf
    import numpy as np

    f = sf.Frame.from_records(
        [
            ["x", 10, 1],
            ["x", 20, 2],
            ["y", 30, 3],
            ["y", 40, 4],
        ],
        columns=("A", "B", "C")
    )

    items = (
        (label, frame.mean())
        for label, frame in f.iter_group_items("A", drop=True)
    )

    result1 = sf.Frame.from_concat_items(items)
    print(result1)
    """,
                ),
                MessageBlock(
                    BlockType.PROSE,
                    '**2. Functional reduction with `.reduce.from_func(...)`**',
                ),
                MessageBlock(
                    BlockType.CODE,
                    """\
    result2 = (
        f.iter_group_items("A", drop=True)
         .reduce.from_func(lambda label, frame: frame.mean(axis=0).rename(label))
         .to_frame()
    )

    print(result2)
    """,
                ),
                MessageBlock(
                    BlockType.PROSE, '**3. High-level chaining with `sf.Batch`**'
                ),
                MessageBlock(
                    BlockType.CODE,
                    """\
    result3 = sf.Batch(f.iter_group_items("A", drop=True)).mean().to_frame()
    print(result3)
    """,
                ),
                MessageBlock(
                    BlockType.PROSE,
                    (
                        'All three approaches yield the same result. Choose based on style and need:\n\n'
                        '- Use `Batch` for compact pipelines\n'
                        '- Use `reduce.from_func` for functional customization\n'
                        '- Use `from_items` when composing manually\n\n'
                        "This shows the flexibility and composability of StaticFrame's design."
                    ),
                ),
            ],
        ),
    ),
    # ---------------------------------------------------------------------------
    Messages(
        user=Message(
            Role.USER,
            [
                MessageBlock(
                    BlockType.PROSE,
                    'In Pandas, I can use `pd.DataFrame(...)` with a list, dict, or ndarray. How do I create frames in StaticFrame?',
                )
            ],
        ),
        assistant=Message(
            Role.ASSISTANT,
            [
                MessageBlock(
                    BlockType.PROSE,
                    (
                        'In Pandas, `DataFrame(...)` is a very flexible constructor, accepting many different input types. For example:'
                    ),
                ),
                MessageBlock(
                    BlockType.CODE,
                    """\
    import pandas as pd

    # list of lists
    print(pd.DataFrame([[1, 2], [3, 4]]))

    # dict of lists
    print(pd.DataFrame({"A": [1, 2], "B": [3, 4]}))

    # 2D NumPy array
    import numpy as np
    print(pd.DataFrame(np.array([[5, 6], [7, 8]])))
    """,
                ),
                MessageBlock(
                    BlockType.PROSE,
                    (
                        'StaticFrame takes a different approach: constructors are **explicit** and based on data orientation.\n\n'
                        'Examples include:'
                    ),
                ),
                MessageBlock(
                    BlockType.CODE,
                    """\
    import static_frame as sf
    import numpy as np

    # Row-oriented construction
    f1 = sf.Frame.from_records([[1, 2], [3, 4]], columns=("A", "B"))
    print(f1)

    # Column-oriented construction
    f2 = sf.Frame.from_dict({"A": [1, 2], "B": [3, 4]})
    print(f2)

    # From a single values
    f3 = sf.Frame.from_element(42, index=range(2), columns=("X", "Y"))
    print(f3)

    # From columnar Series
    f4 = sf.Frame.from_items([
        ("A", sf.Series((1, 2))),
        ("B", sf.Series((3, 4))),
    ], index=(0, 1))
    print(f4)
    """,
                ),
                MessageBlock(
                    BlockType.PROSE,
                    (
                        'This explicitness avoids ambiguity and ensures all inputs are shaped and labeled intentionally. '
                        'StaticFrame favors being **declarative and safe** over being flexible but unpredictable.'
                    ),
                ),
            ],
        ),
    ),
    # ---------------------------------------------------------------------------
]


def get_corpus() -> Corpus:
    return Corpus(EXAMPLES)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Manage and export StaticFrame fine-tuning examples.'
    )
    parser.add_argument('--jsonl', type=Path, help='Output .jsonl path')
    parser.add_argument('--markdown', type=Path, help='Output .md path')
    parser.add_argument('--validate', action='store_true', help='Validate code blocks')
    args = parser.parse_args()

    corpus = get_corpus()

    if args.jsonl:
        corpus.write_jsonl(args.jsonl)
        print(f'✅ Wrote JSONL to {args.jsonl}')

    if args.markdown:
        corpus.write_markdown(args.markdown)
        print(f'✅ Wrote Markdown to {args.markdown}')

    if args.validate:
        corpus.validate_all()
