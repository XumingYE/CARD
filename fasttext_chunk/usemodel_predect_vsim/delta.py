import deltas as delta
import fasttext_chunk.usemodel_predect_vsim.log_v1 as logging


def get_diff_file(a, b, fp=None):
    log = logging.LogSystem()
    logging.LogSystem.print_flag = False
    log.print_log(a)
    log.print_log(b)
    diff_it = delta.sequence_matcher.diff(a, b)
    diff_data = str()
    if fp is not None:
        for data in diff_it:
            diff_data += str(data)
    length_b = len(b)
    length_r = len(diff_data)
    if diff_data is "":
        return "success but not store"
    elif length_r >= length_b:
        path = str(fp).replace("diff", "origin")
        diff_file = open(path, "ab+")
        diff_file.write(b)
    else:
        diff_file = open(fp, "a+")
        diff_file.write(diff_data)

