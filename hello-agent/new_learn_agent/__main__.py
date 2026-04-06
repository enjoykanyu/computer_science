"""
new_learn_agent 包入口点

企业级运行方式：
    cd /Users/kanyu/Desktop/project/kanyu_server/new_project/computer_science/hello-agent
    python -m new_learn_agent
"""

import sys
from dotenv import load_dotenv
from .core.my_main import main

if __name__ == "__main__":
    load_dotenv()
    sys.exit(main())
