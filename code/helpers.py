#!/usr/bin/env python3


def get_methods(object, spacing=20):
    methodList = []
    for method_name in dir(object):
        try:
            if callable(getattr(object, method_name)):
                methodList.append(str(method_name))
        except Exception:
            methodList.append(str(method_name))
    processFunc = (lambda s: " ".join(s.split())) or (lambda s: s)
    for method in methodList:
        try:
            print(
                str(method.ljust(spacing))
                + " "
                + processFunc(str(getattr(object, method).__doc__)[0:90])
            )
        except Exception:
            print(method.ljust(spacing) + " " + " getattr() failed")


def get_dd(group):
    with open(os.path.join(data_dir, "participants.tsv"), "r") as f:
        data = pd.read_csv(f, sep="\t")

    data = dict(zip(data["participant_id"], data["group"]))
    # print(data)

    def which_group(out, subj):
        # print(out)
        # print(data["sub-" + subj])
        if data["sub-" + subj] == "DD":
            return {"DD": out["DD"] + [subj], "TA": out["TA"]}
        else:
            return {"DD": out["DD"], "TA": out["TA"] + [subj]}

    out = ft.reduce(which_group, group["subjs"], {"DD": [], "TA": []})
    return out
