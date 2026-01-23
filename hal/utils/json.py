

def make_jsonable(data, seen=None):
    if seen is None:
        seen = set()

    if isinstance(data, (str, int, float, bool, type(None))):
        return data
    elif isinstance(data, dict):
        return {key: make_jsonable(value, seen) for key, value in data.items()}
    elif isinstance(data, list):
        return [make_jsonable(item, seen) for item in data]
    elif isinstance(data, tuple):
        return tuple(make_jsonable(item, seen) for item in data)
    elif hasattr(data, "__dict__"):
        obj_id = id(data)
        if obj_id in seen:
            return f"<Circular reference: {obj_id}>"
        seen.add(obj_id)
        return make_jsonable(data.__dict__, seen)
    elif hasattr(data, "_asdict"):  # for namedtuple or similar
        return make_jsonable(data._asdict(), seen)
    else:
        return str(data)
