import os


def get_contents_in_dir(dir_path, notstartswith, endswith):
    """List files in ``dir_path``.

    Excludes names starting with any entry of ``notstartswith``, and keeps
    only names ending with any entry of ``endswith``. If nothing matches
    (e.g. when listing directories), all contents are returned.
    """
    contents = os.listdir(dir_path)
    for e in notstartswith:
        contents = [c for c in contents if not c.startswith(e)]
    res = []
    for e in endswith:
        res += [c for c in contents if c.endswith(e)]
    if len(res) == 0:  # for directory
        res = contents

    return sorted(os.path.join(dir_path, r) for r in res)
