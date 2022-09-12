import xdelta3


def get_diff_file(a, b, fp=None):
    # print(a)
    # print(b)
    diff_data = str()
    try:
        diff_data = xdelta3.encode(a, b)
    except BaseException:
        pass
    length_b = len(b)
    length_r = len(diff_data)
    if diff_data is "":
        return "success but not store"
    elif length_r >= length_b:
        path = str(fp).replace("diff", "origin")
        diff_file = open(path, "a+")
        diff_file.write(str(b)[2:-1])
        diff_file.close()
    else:
        diff_file = open(fp, "ab+")
        diff_file.write(diff_data)
        diff_file.close()
