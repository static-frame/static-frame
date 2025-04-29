from dataclasses import dataclass, field
from typing import List
from enum import Enum
import json
from pathlib import Path
import argparse


class BlockType(Enum):
    PROSE = "prose"
    CODE = "code"


class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"


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
                parts.append(f"```python\n{block.content.strip()}\n```")
        return "\n\n".join(parts)

    def get_code_blocks(self) -> List[str]:
        return [b.content for b in self.blocks if b.type == BlockType.CODE]


@dataclass(frozen=True)
class Messages:
    user: Message
    assistant: Message

    def to_jsonl_line(self) -> str:
        return json.dumps({
            "messages": [
                {"role": self.user.role.value, "content": self.user.to_text()},
                {"role": self.assistant.role.value, "content": self.assistant.to_text()},
            ]
        })

    def to_markdown(self, i: int) -> str:
        return (
            f"### Example {i}\n\n"
            f"**User:**\n\n{self.user.to_text()}\n\n"
            f"**Assistant:**\n\n{self.assistant.to_text()}\n\n---"
        )

    def validate_code(self, context: dict = None):
        for code in self.assistant.get_code_blocks():
            exec(code, context if context is not None else {})


@dataclass
class Corpus:
    examples: List[Messages] = field(default_factory=list)

    def write_jsonl(self, path: Path):
        path.write_text(
            "\n".join(m.to_jsonl_line() for m in self.examples),
            encoding="utf-8"
        )

    def write_markdown(self, path: Path):
        path.write_text(
            "\n\n".join(m.to_markdown(i + 1) for i, m in enumerate(self.examples)),
            encoding="utf-8"
        )

    def validate_all(self, context: dict = None):
        for i, msg in enumerate(self.examples, 1):
            try:
                msg.validate_code(context)
                print(f"✅ Example {i}: code validated")
            except Exception as e:
                print(f"❌ Example {i}: code execution failed\n{e}")


# Example Messages data
EXAMPLES = [
    Messages(
        user=Message(Role.USER, [
            MessageBlock(BlockType.PROSE, "Can I use `.rolling()` in StaticFrame?")
        ]),
        assistant=Message(Role.ASSISTANT, [
            MessageBlock(BlockType.PROSE, "StaticFrame does not support a `.rolling()` method like Pandas. Instead, use `iter_window_array(size=...)` combined with `.reduce.from_map_func(...)`. For example:"),
            MessageBlock(BlockType.CODE, """
result = (
    f.iter_window_array(size=3)
     .reduce.from_map_func(np.mean)
     .to_frame(index_constructor=sf.IndexAutoConstructorFactory)
)
""")
        ])
    ),
    Messages(
        user=Message(Role.USER, [
            MessageBlock(BlockType.PROSE, "How do I use `IndexDate.from_range()` to get 10 days starting from 2020-01-01?")
        ]),
        assistant=Message(Role.ASSISTANT, [
            MessageBlock(BlockType.PROSE, "There is no `IndexDate.from_range()` method in StaticFrame. Instead, use `IndexDate.from_date_range(start, stop, step=1)`, where `stop` is inclusive. To create a 10-day range:"),
            MessageBlock(BlockType.CODE, "sf.IndexDate.from_date_range('2020-01-01', '2020-01-10', step=1)")
        ])
    ),
    Messages(
        user=Message(Role.USER, [
            MessageBlock(BlockType.PROSE, "Can I do `df['col'] = df['col'] * 2` in StaticFrame?")
        ]),
        assistant=Message(Role.ASSISTANT, [
            MessageBlock(BlockType.PROSE, "StaticFrame objects are immutable. You cannot mutate columns in place like in Pandas. Instead, construct a new Frame with the updated column:"),
            MessageBlock(BlockType.CODE, "f_updated = f.assign['col'](f['col'] * 2)")
        ])
    ),
    Messages(
        user=Message(Role.USER, [
            MessageBlock(BlockType.PROSE, "How do I compute a rolling median of a Frame in StaticFrame?")
        ]),
        assistant=Message(Role.ASSISTANT, [
            MessageBlock(BlockType.PROSE, "Use `iter_window_array` with `.reduce.from_map_func(np.median)` and reconstruct a Frame. Example:"),
            MessageBlock(BlockType.CODE, """
f.iter_window_array(size=5)
 .reduce.from_map_func(np.median)
 .to_frame(index_constructor=sf.IndexAutoConstructorFactory)
""")
        ])
    )
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage and export StaticFrame fine-tuning examples.")
    parser.add_argument("--jsonl", type=Path, help="Output .jsonl path")
    parser.add_argument("--markdown", type=Path, help="Output .md path")
    parser.add_argument("--validate", action="store_true", help="Validate code blocks")
    args = parser.parse_args()

    corpus = Corpus(EXAMPLES)

    if args.jsonl:
        corpus.write_jsonl(args.jsonl)
        print(f"✅ Wrote JSONL to {args.jsonl}")

    if args.markdown:
        corpus.write_markdown(args.markdown)
        print(f"✅ Wrote Markdown to {args.markdown}")

    if args.validate:
        print("🔍 Validating code blocks...")
        corpus.validate_all(context={"np": __import__("numpy")})

