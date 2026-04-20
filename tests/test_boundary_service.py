from __future__ import annotations

from pathlib import Path
import unittest

from app.services.boundary_service import BoundaryService


class BoundaryServiceTests(unittest.TestCase):
    def test_parses_ningbo_kml_and_contains_known_point(self) -> None:
        root = Path(__file__).resolve().parents[1]
        service = BoundaryService.from_kml(root / "宁波市.kml")
        self.assertTrue(service.contains_point(121.544, 29.8683))
        self.assertFalse(service.contains_point(121.4737, 31.2304))
        self.assertEqual(len(service.bbox), 4)


if __name__ == "__main__":
    unittest.main()
