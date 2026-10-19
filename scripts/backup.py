"""按 config.yaml 的 backup.* 在线备份 SQLite 库。"""
from ragagent.monitor.backup import main

if __name__ == '__main__':
    raise SystemExit(main())
