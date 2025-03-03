import asyncio
from typing import Optional, List, Tuple

import aiofiles


class WuZhiZhuanShengNovelSplitter:
    def __init__(self, novel_path: str, text: Optional[str]=None) -> None:
        self.novel = text
        self.novel_path = novel_path
        self.chapter_split_label = "铅笔小说\n(www.x23qb.com)"
        self.useless_preface = (
            "台版 转自 轻之国度\n"
            "扫图：zince99\n"
            "录入：zbszsr\n"
            "「眼前是悬崖。要往前踏出一步狠狠摔向地面，\n"
            "或者想裹足不前继续承受辱骂，都是你的自由。」\n"
            "——I do not want to work,whatever it may be said by whom.\n"
            "着：鲁迪乌斯·格雷拉特\n"
            "译：金恩·RF·马格特\n"
            "－－－－－\n"
        )

    async def load_novel(self) -> None:
        if self.novel is not None:
            return
        async with aiofiles.open(self.novel_path, "r", encoding="utf8") as f:
            self.novel = await f.read()

    async def split_chapter(self) -> Tuple[List[Tuple[str, str, str]], List[str]]:
        chapters = self.novel.split(self.chapter_split_label)
        chapters[0] = chapters[0].replace(self.useless_preface, "")
        chapters = ["\n".join(c.strip().split("\n")[1:]) for c in chapters]

        titles = [c.split("\n")[0] for c in chapters]
        titles = [title for title in titles if title]
        titles = [title.replace("# ", "").split(" ") for title in titles]
        titles = [(title[0], title[1], " ".join(title[2:]).replace("/", "-")) for title in titles]
        chapters = ["\n".join(c.strip().split("\n")[1:]) for c in chapters]
        chapters = [c for c in chapters if c]
        assert len(titles) == len(chapters), f"titles: {len(titles)}, chapters: {len(chapters)}"
        return titles, chapters

async def test():
    splitter = WuZhiZhuanShengNovelSplitter("data/docs/Wu Zhi Zhuan Sheng  ~Zai Yi Shi - Wei Zhi.txt")
    await splitter.load_novel()
    titles, chapters = await splitter.split_chapter()
    print(titles)
    print("=====================================")
    print(chapters[0])


if __name__ == "__main__":
    asyncio.run(test())
