import time


class LogSystem:
    print_flag = True
    level_dirc = {1: "INFO", 2: "WARN", 3: "ERROR"}

    @staticmethod
    def print_log(msg, level=1):
        if not LogSystem.print_flag:
            return
        date = time.asctime(time.localtime(time.time()))
        print("[%s %s]  %s" % (date, LogSystem.level_dirc[level], msg))
if __name__ == '__main__':
    l = LogSystem()
    k = "osddsd"
    l.print_log("424sd %s"%(k))

