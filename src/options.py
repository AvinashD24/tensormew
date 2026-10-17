"""wOptions bits from pret ram_constants.asm. Resolved by symbol, not by address."""

TEXT_DELAY_FAST = 0b001
TEXT_DELAY_MED = 0b011
TEXT_DELAY_SLOW = 0b101
BATTLE_SHIFT_BIT = 6  # 1 = SET
BATTLE_SCENE_BIT = 7  # 1 = OFF


def training_options(value: int) -> int:
    value = (value & ~0b111) | TEXT_DELAY_FAST
    value |= 1 << BATTLE_SHIFT_BIT
    value |= 1 << BATTLE_SCENE_BIT
    return value & 0xFF


def options_ok(value: int) -> bool:
    return not options_warnings(value)


def options_warnings(value: int) -> list[str]:
    warnings = []
    delay = value & 0b111
    if delay != TEXT_DELAY_FAST:
        warnings.append("text speed is not FAST")
    if not (value & (1 << BATTLE_SHIFT_BIT)):
        warnings.append("battle style is not SET")
    if not (value & (1 << BATTLE_SCENE_BIT)):
        warnings.append("battle scene is not OFF")
    return warnings


def options_text(value: int) -> str:
    delay = {TEXT_DELAY_FAST: "FAST", TEXT_DELAY_MED: "MED", TEXT_DELAY_SLOW: "SLOW"}.get(value & 0b111, "OTHER")
    style = "SET" if value & (1 << BATTLE_SHIFT_BIT) else "SHIFT"
    scene = "OFF" if value & (1 << BATTLE_SCENE_BIT) else "ON"
    warn = "" if options_ok(value) else " WARN"
    return f"options text={delay} style={style} scene={scene}{warn}"
