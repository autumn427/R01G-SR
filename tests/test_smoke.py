import copy
import unittest

from src.repair_section_ownership import repair
from src.llm import parse_json


class SmokeTests(unittest.TestCase):
    def test_unique_owner_move_preserves_claim(self):
        report = {'sections': [{'id': '1', 'title': 'Situation overview',
            'subsections': [{'id': '1a', 'title': 'Fatalities', 'bullets': [
                {'id': '1a.1', 'text': 'Synthetic claim.', 'confidence': 'high',
                 'evidence': ['synthetic-1']}]}]}]}
        before = copy.deepcopy(report)
        result, moves = repair(report)
        self.assertEqual(report, before)
        self.assertEqual(len(moves), 1)
        self.assertEqual(result['sections'][0]['id'], '3')
        bullet = result['sections'][0]['subsections'][0]['bullets'][0]
        self.assertEqual(bullet['text'], 'Synthetic claim.')
        self.assertEqual(bullet['evidence'], ['synthetic-1'])

    def test_no_move_returns_equal_copy(self):
        original = {'sections': [{'id': '1', 'title': 'Situation overview',
            'subsections': [{'id': '1a', 'title': 'Overview', 'bullets': []}]}]}
        result, moves = repair(original)
        self.assertEqual(result, original)
        self.assertIsNot(result, original)
        self.assertEqual(moves, [])

    def test_json_parser(self):
        self.assertEqual(parse_json('```json\n{"sections": []}\n```'), {'sections': []})


if __name__ == '__main__':
    unittest.main()
