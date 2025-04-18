import os
import re
from datetime import datetime


log_path = os.path.join(os.getcwd(), "logs")


class Logger:
    def __init__(self, filename, is_debug, folder=None):

        self.filename = re.sub(r"[^\w\d_]", "_", filename.replace(" ", "_"))
        if folder is not None:
            self.path = os.path.join(os.getcwd(), "logs", folder)
        else:
            self.path = os.path.join(os.getcwd(), "logs")
        self.log_ = not is_debug

        if not os.path.exists(self.path):
            os.makedirs(self.path)
            print(f"日志文件夹 {self.path} 创建成功。")

    def logging(self, s):
        s = str(s)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        log_message = f"{timestamp}: {s}"
        print(log_message)

        if self.log_:
            log_file_path = os.path.join(self.path, self.filename)
            with open(log_file_path, "a+") as f_log:
                f_log.write(log_message + "\n")
