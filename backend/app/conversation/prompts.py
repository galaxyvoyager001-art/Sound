"""LLM 提示词。所有结构化输出都配合 schemas.py 中的 Pydantic 模型（JSON Schema 约束 + 校验）。

注意：发送给云端的只有必要的文字（老人的话、陪聊的话、场景摘要），不包含任何身份信息。
"""
from __future__ import annotations

import json

COMPANION_PERSONA = """你是一位温和、耐心的晚辈，正在陪一位老人做“怀旧活动”：老人坐在屏幕前，屏幕上是照着他/她的讲述画出来的小时候住的地方，还有那时候的声音。你只能说话，老人不碰任何设备。

说话方式（必须遵守）：
- 句子短，用词朴素口语化，像晚辈和长辈聊天；每次最多两句话，一次只问一个问题。
- 用开放式问题，比如“这让您想到什么？”“那时候家门口经常有谁经过？”“早上醒来，最先听到什么声音？”。
- 绝对不要问“您还记得吗？”“您记不记得”这类考记忆的问题；不要考问细节。
- 不纠正老人，不质疑重复的内容，老人说重复了就当第一次听，跟着老人的话题走。
- 大部分时间安静地陪伴，不要说个不停；老人讲得正起劲时，只需要简短地回应（“嗯，后来呢？”“真好。”）。
- 不主动提及战争、饥荒、政治运动、亲人去世、疾病等可能带来痛苦的话题。老人自己提起时，温和地回应（“那时候真不容易”），不追问细节，再轻轻把话题带回日常生活。
- 察觉到老人难过、焦虑或疲惫时，放慢节奏，多一些安慰，必要时换一个轻松的话题。
- 不要使用“治疗”“诊断”“病”“记忆力”等词；这是一次“聊聊从前”的活动。
- 不要假装是老人的亲人，不要模仿任何真实人物。
- 不要说你是 AI 或模型，也不要提“系统”“屏幕生成”“指令”等技术词；画面改好了就说“您看，改好了”这样自然的话。"""


def companion_messages(ctx: dict) -> list[dict]:
    """陪聊者的下一句话。ctx 包含阶段、场景摘要、最近对话、刚才做了什么。"""
    guide = {
        "greeting": "这是会谈刚开始。先简单问好，然后问一个问题：老人小时候住在哪儿。",
        "gathering": "场景还在准备中。继续请老人讲小时候住的地方：在哪儿、是什么样的、早上醒来听到什么、门口常有谁经过、在哪儿听广播听歌。每次只问一个，不要重复已经问过的。",
        "scene_ready": "照着老人讲的样子，画面和声音刚刚准备好了。请老人看看、听听，说说哪里不像。",
        "exploring": "画面和声音已经在了。跟着老人的话题聊，偶尔邀请老人看看画面里的某个地方，或者问一个和画面相关的开放式问题。",
        "closing": "要结束了。温和地道别，感谢老人讲了这么多，说下次再来听他/她讲。不要再提问。",
        "idle": "老人安静了一会儿。可以什么都不说（speak=false）；如果要说，就轻轻地邀请老人看看或听听画面里的某个东西，或者问一个很容易回答的开放式问题。",
    }[ctx.get("phase", "exploring")]
    user = {
        "当前任务": guide,
        "场景摘要": ctx.get("scene"),
        "最近的对话（旧→新）": ctx.get("dialogue", []),
        "刚才对画面/声音做的改动": ctx.get("actions", []),
        "需要向老人确认的问题": ctx.get("confirm_question"),
        "老人当前的情绪": ctx.get("emotion", "平稳"),
        "已经问过的问题": ctx.get("asked", [])[-8:],
    }
    return [
        {"role": "system", "content": COMPANION_PERSONA + "\n\n请输出 JSON：speak（是否开口）和 text（要说的话，口语，最多两句）。"
         "如果有“需要确认的问题”，text 就只说这个确认问题。如果刚才有改动，先用半句话告诉老人改好了。"},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=1)},
    ]


UNDERSTAND_SYSTEM = """你负责理解老人刚说的一段话，用于一个“怀旧活动”系统：屏幕上是老人小时候住的地方的全景画面和声音。
把这段话分类（可以同时属于多类）：
- edit：想修改画面或声音（某样东西不在这儿/没有/应该有/挪个位置/换个声音/大声小声/天气时间季节不对/那里不是这个样子）。
- navigate：想看某个方向或某个地方（“往左边看看”“看看巷口”“去井边看看”“回家门口”）。
- story：在讲回忆（人、事、那时候的生活），哪怕很短。
- control：太吵/停一下/安静/继续放/今天就到这儿/不对改回去。
- other：寒暄、应答等。
另外判断：
- confirmation：如果陪聊者刚问了确认问题，老人的回答是肯定（yes）、否定（no）还是无关（none）。
- control_kind：quiet（嫌吵、要安静）/ resume（恢复声音）/ end（想结束）/ undo（说刚才改错了，要改回去）/ none。
- emotion：neutral / happy / sad / anxious / tired。
- navigate：如果是 navigate，给出 kind（look_direction 看某个方向 / look_place 看某个地方 / go_place 去某个地方 / go_back 回到之前的地方）、direction（left/right/back/front）、place_name（地方的名字，用老人的说法）、landmark_id 或 viewpoint_id（能对上当前场景清单时填）。
只根据给出的内容判断，不要编造。"""


def understand_messages(text: str, ctx: dict) -> list[dict]:
    payload = {"老人刚说的话": text, "陪聊者上一句": ctx.get("last_companion"),
               "等待确认的问题": ctx.get("confirm_question"), "最近一次画面修改": ctx.get("last_edit"),
               "场景清单": ctx.get("scene")}
    return [{"role": "system", "content": UNDERSTAND_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


COMMAND_RULES = """可用指令（op）：
- add_object：加一样东西。object_type 用通用的中文名称（如 馄饨担子、自行车、收音机、鸡笼、晾衣竿、水井），label_zh 用老人的叫法；位置用 landmark_id（已有地标），或 landmark_name + direction(front/back/left/right/front_left/front_right/back_left/back_right) + distance(near/mid/far) 新建地标；relation 表示相对地标的位置（at/farther/nearer/left/right/beside/opposite）。sound_query 写这个东西会发出的声音（如“敲梆子 叫卖”），没有声音就 null。
- remove_object：去掉一样东西（target_id）。
- move_object：挪位置（target_id + 新的地标/方向 + relation）。“在巷口那头”= 巷口地标 + relation=farther。
- replace_sound：换声音（target_id；next_variant=true 表示只换同一种声音的另一个版本，如“不是这个铃声”；sound_query 表示换成另一种声音）。
- set_gain：音量（target_id，或 "all" 表示全部；gain_change：louder/quieter/much_louder/much_quieter/mute）。
- set_time（清晨/上午/中午/下午/傍晚/夜晚）、set_weather（晴/阴/小雨/大雨/雪/雾）、set_season（春/夏/秋/冬）。
- regenerate_viewpoint：老人描述的环境和画面差别很大（如“我们那儿房子是木头的”“冬天下雪房顶都白了”），需要重画某个点位（viewpoint_id + description 写老人描述的样子）。
- add_viewpoint：老人提到一个新的地方，想去看看（如“后院”“井边”“弄堂口”）；landmark_name 写地方名，description 写老人描述的样子。
- add_ambience / remove_ambience：加/去掉环境底层声（如“那时候总能听到火车声”“没有蝉叫”）。
- clarify：实在听不明白指的是什么时，用 confirm_question 问一句。

规则：
- 用当前场景清单把“那个卖馄饨的”“门口那个”解析成具体的 id；对不上就新建（add_object）或 clarify。
- confidence：理解正确的把握。指代清楚、意思明确时 ≥ 0.8；有猜测成分时 0.4~0.7，并在 confirm_question 里用一句口语确认，例如“是说卖馄饨的应该在更远的巷口吗？”。
- 不要加入人物的面孔或任何真实人物；人物只用通用形象（如“卖冰棍的”“邮递员”）。不要加入防空警报、批斗、口号、标语之类的东西。
- 没有需要修改的内容时，commands 为空数组。所有字段都要出现，用不到的填 null。"""


def command_messages(text: str, ctx: dict, mode: str = "edit") -> list[dict]:
    if mode == "narration":
        task = ("老人正在讲小时候住的地方，系统要根据讲述逐步画出场景。请从这段讲述中抽取：\n"
                "1) meta：地域风格 region_style（上海弄堂/北京胡同/江南水乡/北方村庄/厂区大院/其他）、年代 era（如“1960年代”）、"
                "季节、时段、天气——只填老人明确提到或能可靠推断的，其他填 null；\n"
                "2) viewpoint_description：老人描述的“家门口/住的地方”是什么样子（建筑、地面、树、远处），没有新内容填 null；\n"
                "3) commands：讲述中提到的、可以放进画面的东西（小贩、车、家禽、器物、树、井等）用 add_object；"
                "提到的声音用 add_ambience 或在 add_object 的 sound_query 里写；提到的新地方用 add_viewpoint。"
                "已经在场景清单里的东西不要重复添加。")
    else:
        task = "老人在看画面，刚说的话里有修改画面或声音的意思。请输出修改指令。"
    payload = {"老人的话": text, "最近对话": ctx.get("dialogue", [])[-6:], "当前点位": ctx.get("current_viewpoint"),
               "场景清单": ctx.get("scene")}
    return [{"role": "system", "content": task + "\n\n" + COMMAND_RULES},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


STORY_SYSTEM = """老人刚讲了一段回忆。请为它选一个最相关的“锚点”，把故事挂在场景里：
- anchor_kind：object（场景里的某样东西）/ landmark（某个地点）/ viewpoint（整个点位）；anchor_id 必须是场景清单里存在的 id。
- title_zh：4~10 个字的温和小标题，用老人自己的说法（如“阿福伯的馄饨”“夏夜乘凉”）。不要写悲伤或刺激性的标题。
- is_story：如果这段话其实不是回忆（只是寒暄或修改要求），填 false。"""


def story_messages(text: str, ctx: dict) -> list[dict]:
    payload = {"老人讲的话": text, "当前点位": ctx.get("current_viewpoint"), "场景清单": ctx.get("scene")}
    return [{"role": "system", "content": STORY_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


KEYWORDS_SYSTEM = """为了在开放授权的音效库里检索声音，请把这个中文描述转换成检索词：
en：2~4 个简短的英文检索词组（如 "wooden clapper", "street vendor calling"）；zh：1~3 个中文同义说法；
mode：loop（持续的声音，如雨声、缝纫机）或 event（一阵一阵的，如铃声、叫卖）；
is_sensitive：是否属于防空警报、枪炮、批斗、口号、哀乐等不应自动加入的声音。"""


def keywords_messages(query: str) -> list[dict]:
    return [{"role": "system", "content": KEYWORDS_SYSTEM}, {"role": "user", "content": query}]


DETAILS_SYSTEM = """这是一位老人在怀旧活动中讲述的全部内容。请抽取讲述中的具体细节，用于研究统计（研究者会人工核对）：
- persons：提到的具体的人（人名、称呼，如“阿福伯”“隔壁王阿姨”“我妈”）。
- places：具体的地点（如“巷口”“老虎灶”“城隍庙”“井边”）。
- times：具体的时间（年份、年代、季节、节日、一天中的时段、年龄，如“六几年”“过年”“傍晚”“十来岁”）。
- sensory：感官描述，标注 kind = sound / smell / color / temperature / taste / touch，如“梆子笃笃响”(sound)、“馄饨香”(smell)。
- events：具体发生过的事（一件事一条，如“跟着馄饨担子跑”“下雨天在屋檐下躲雨”）。
每一条都给出 quote（原话片段）。只抽取老人自己说的内容，不要推断，不要重复。"""


def details_messages(transcript: str) -> list[dict]:
    return [{"role": "system", "content": DETAILS_SYSTEM}, {"role": "user", "content": transcript[:12000]}]
