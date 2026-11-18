import os
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE, "raw_data", "original")


class MyHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        if event.is_directory:
            return
        if not event.src_path.lower().endswith(".csv"):
            return
        time.sleep(1)
        try:
            subprocess.run(["python", "eda_process.py"], cwd=BASE, check=True)
            print("✅ CSV变动，清洗完成")
        except Exception as e:
            print(f"❌清洗异常:{e}")


if __name__ == "__main__":
    handler = MyHandler()
    obs = Observer()
    obs.schedule(handler, INPUT_FOLDER, recursive=False)
    print(f"👀监控启动，监听 {INPUT_FOLDER}")
    obs.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        obs.stop()
    obs.join()
