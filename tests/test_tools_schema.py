import unittest

from pydantic import ValidationError

from src.tools_schema import TOOL_DESCRIPTIONS, anthropic_tools, parse_tool_call, tool_names, validate_arguments


class ToolSchemaTests(unittest.TestCase):
    def test_exports_six_tools(self):
        names = tool_names()
        self.assertGreaterEqual(len(names), 6)
        self.assertEqual(
            set(names),
            {"go_to", "run_path", "use_cut", "mash_text", "choose_move", "switch_pokemon"},
        )
        exported = anthropic_tools()
        self.assertEqual([t["name"] for t in exported], names)
        for tool in exported:
            self.assertIn("input_schema", tool)
            self.assertEqual(tool["description"], TOOL_DESCRIPTIONS[tool["name"]])

    def test_validate_and_roundtrip(self):
        args = validate_arguments("go_to", {"x": 4, "y": 9})
        call = parse_tool_call("go_to", args)
        self.assertEqual(call.name, "go_to")
        self.assertEqual(call.arguments, {"x": 4, "y": 9})
        again = parse_tool_call(call.name, call.arguments)
        self.assertEqual(again, call)

    def test_defaults_and_bounds(self):
        self.assertEqual(validate_arguments("use_cut", {}), {})
        self.assertEqual(validate_arguments("mash_text", {}), {"max_frames": 600})
        self.assertEqual(validate_arguments("run_path", {"directions": ["up", "left"]})["directions"], ["up", "left"])
        with self.assertRaises(ValidationError):
            validate_arguments("choose_move", {"slot": 5})
        with self.assertRaises(ValidationError):
            validate_arguments("switch_pokemon", {"slot": 0})
        with self.assertRaises(ValidationError):
            validate_arguments("run_path", {"directions": ["start"]})

    def test_unknown_tool(self):
        with self.assertRaises(ValueError):
            parse_tool_call("press_a", {})


if __name__ == "__main__":
    unittest.main()
