"""数据管线组合."""


class Compose:
    """将多个 transform 组合为管线.

    接受已构建的 callable 或 Lazy 描述符。
    """

    def __init__(self, transforms):
        from flashocc.config.lazy import Lazy

        self.transforms = []
        for t in transforms:
            if isinstance(t, Lazy):
                self.transforms.append(t.build())
            elif callable(t):
                self.transforms.append(t)
            else:
                raise TypeError(f"transform 须为 Lazy 或 callable, 得到 {type(t)}")

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        s = f"{self.__class__.__name__}(\n"
        for t in self.transforms:
            s += f"    {t}\n"
        s += ")"
        return s


__all__ = ["Compose"]
