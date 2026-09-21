"""測試共用設定：將專案根目錄加入 sys.path，讓 core / agents 可被匯入。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
