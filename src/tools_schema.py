"""Frozen Phase 1 tool contract for the LLM supervisor."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

DIRS = ("up", "down", "left", "right")


class GoToArgs(BaseModel):
    x: int
    y: int


class RunPathArgs(BaseModel):
    directions: list[Literal["up", "down", "left", "right"]] = Field(min_length=1)

    @field_validator("directions")
    @classmethod
    def known(cls, value: list[str]) -> list[str]:
        bad = [d for d in value if d not in DIRS]
        if bad:
            raise ValueError(f"unknown directions: {bad}")
        return value


class UseCutArgs(BaseModel):
    pass


class MashTextArgs(BaseModel):
    max_frames: int = Field(default=600, ge=1, le=10_000)


class ChooseMoveArgs(BaseModel):
    slot: int = Field(ge=1, le=4)


class SwitchPokemonArgs(BaseModel):
    slot: int = Field(ge=1, le=6)


TOOL_MODELS: dict[str, type[BaseModel]] = {
    "go_to": GoToArgs,
    "run_path": RunPathArgs,
    "use_cut": UseCutArgs,
    "mash_text": MashTextArgs,
    "choose_move": ChooseMoveArgs,
    "switch_pokemon": SwitchPokemonArgs,
}

TOOL_DESCRIPTIONS: dict[str, str] = {
    "go_to": "Walk toward a tile (x, y) on the current map.",
    "run_path": "Take exact steps. Each direction is up, down, left, or right.",
    "use_cut": "Use HM Cut on the tree blocking the path.",
    "mash_text": "Advance dialogue. max_frames caps how long to mash.",
    "choose_move": "Pick a battle move. slot is 1 through 4.",
    "switch_pokemon": "Switch the active Pokémon. slot is 1 through 6.",
}


class ToolCall(BaseModel):
    name: Literal["go_to", "run_path", "use_cut", "mash_text", "choose_move", "switch_pokemon"]
    arguments: dict


def validate_arguments(name: str, arguments: dict | None) -> dict:
    if name not in TOOL_MODELS:
        raise ValueError(f"unknown tool {name!r}")
    model = TOOL_MODELS[name]
    parsed = model.model_validate(arguments or {})
    return parsed.model_dump()


def parse_tool_call(name: str, arguments: dict | None) -> ToolCall:
    cleaned = validate_arguments(name, arguments)
    return ToolCall(name=name, arguments=cleaned)


def anthropic_tools() -> list[dict]:
    tools = []
    for name, model in TOOL_MODELS.items():
        schema = model.model_json_schema()
        schema.pop("title", None)
        schema.pop("$defs", None)
        tools.append(
            {
                "name": name,
                "description": TOOL_DESCRIPTIONS[name],
                "input_schema": schema,
            }
        )
    return tools


def tool_names() -> list[str]:
    return list(TOOL_MODELS)
