"""Smoke tests for the Streamlit pages.

``AppTest`` runs a page script in-process, so these catch import errors, layout
mistakes and exceptions without a browser. They resolve paths relative to this
file, which is why the repository root is computed explicitly.
"""

import os
from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_LAB = PROJECT_ROOT / "pages" / "9_研究實驗室.py"
DESK_OVERVIEW = PROJECT_ROOT / "pages" / "0_總覽.py"
SIGNAL_FLOW = PROJECT_ROOT / "pages" / "1_訊號流.py"


def test_research_lab_renders_with_offline_data():
    os.environ["SECTOR_ROTATION_TEST_MODE"] = "1"
    try:
        app = AppTest.from_file(str(RESEARCH_LAB), default_timeout=60)
        app.run()

        assert not app.exception
        assert app.title[0].value == "Institutional Fund Flow & Rotation Research Lab"
        assert any("最新模型訊號" in item.value for item in app.subheader)
        assert any("哪些產業正在轉強" in item.value for item in app.markdown)
        assert any(
            button.label == "下載產業輪動與原因 CSV"
            for button in app.download_button
        )
        assert len(app.metric) >= 9
    finally:
        del os.environ["SECTOR_ROTATION_TEST_MODE"]


def test_desk_overview_renders_against_the_local_database():
    app = AppTest.from_file(str(DESK_OVERVIEW), default_timeout=120)
    app.run()

    assert not app.exception
    assert app.title[0].value.endswith("自主市場情報交易台")
    # Regime, universe size, signal count, P1 count and elapsed time.
    assert len(app.metric) >= 5


def test_signal_flow_page_renders():
    app = AppTest.from_file(str(SIGNAL_FLOW), default_timeout=120)
    app.run()

    assert not app.exception
    assert app.title[0].value == "訊號流"
