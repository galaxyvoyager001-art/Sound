"""研究报告：单场会谈的 HTML 报告，跨会谈 / 跨条件的对比表（HTML + CSV）。"""
from __future__ import annotations

import html
import io
import csv
import json
import math
from statistics import mean, stdev

COND_ZH = {"audio_only": "只听", "generic": "通用场景", "personal": "个人共建场景"}
STYLE = """
<style>
body{font-family:"PingFang SC","Microsoft YaHei",sans-serif;margin:24px auto;max-width:1100px;padding:0 16px;color:#2b2118;background:#fbf7f0}
h1{font-weight:500} h2{font-weight:500;margin-top:32px;border-bottom:1px solid #d9c9b0;padding-bottom:4px}
table{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0} th,td{border:1px solid #e0d2bc;padding:6px 8px;text-align:left;vertical-align:top}
th{background:#f1e6d4;font-weight:500} .num{text-align:right;font-variant-numeric:tabular-nums}
.note{color:#7a6650;font-size:13px} .init{color:#2f6f3e} .q{color:#7a6650}
.bar{display:inline-block;height:10px;background:#c8955a;border-radius:2px;vertical-align:middle}
</style>"""

LABELS = [
    ("elder_speech_s", "讲述总时长（秒）"), ("elder_turns", "讲述段数"), ("elder_initiated_turns", "主动开口次数"),
    ("elder_chars", "讲述字数"), ("mean_turn_s", "平均每段时长（秒）"), ("details_total", "具体细节总数"),
    ("details_persons", "人名"), ("details_places", "地名"), ("details_times", "时间"), ("details_sensory", "感官描述"),
    ("details_sensory_sound", "　其中：声音"), ("details_sensory_smell", "　　　　气味"), ("details_sensory_color", "　　　　颜色"),
    ("details_sensory_temperature", "　　　　温度"), ("details_events", "事件"), ("edits_executed", "修改次数"),
    ("narration_additions", "讲述生成的元素"), ("undo", "撤销次数"), ("confirm_asked", "确认提问"),
    ("viewpoint_switches", "点位切换次数"), ("stories", "故事数量"), ("companion_utterances", "陪聊发言次数"),
    ("quiet_requests", "要求安静次数"), ("barge_ins", "老人插话次数"),
]


def e(x) -> str:
    return html.escape(str(x if x is not None else ""))


def session_html(m: dict, details, turns: list[dict], flags: list[bool], stories: list[dict]) -> str:
    rows = "".join(f"<tr><td>{e(lbl)}</td><td class=num>{e(m.get(k))}</td></tr>" for k, lbl in LABELS)
    ops = "".join(f"<tr><td>{e(k)}</td><td class=num>{v}</td></tr>" for k, v in sorted(m.get("edits_by_op", {}).items()))

    def det(title, items, kind=False):
        if not items:
            return f"<h3>{title}（0）</h3>"
        lis = "".join(f"<tr><td>{e(getattr(i, 'kind', '')) if kind else ''}</td><td>{e(i.text)}</td><td class=q>{e(i.quote)}</td></tr>"
                      for i in items)
        return f"<h3>{title}（{len(items)}）</h3><table><tr><th>{'子类' if kind else ''}</th><th>内容</th><th>原话</th></tr>{lis}</table>"

    ts = "".join(
        f"<tr><td class=num>{t['start_s']:.0f}s</td><td class=num>{t['speech_s']:.1f}</td>"
        f"<td>{'<span class=init>主动</span>' if f else '回应'}</td><td>{e(t['text'])}</td></tr>"
        for t, f in zip(turns, flags))
    st = "".join(f"<tr><td>{e(s.get('title'))}</td><td>{e(s.get('anchor', {}).get('kind'))}</td>"
                 f"<td class=num>{s.get('chars')}</td></tr>" for s in stories)
    return f"""<!doctype html><html lang=zh-CN><head><meta charset=utf-8><title>会谈报告 {e(m['session_id'])}</title>{STYLE}</head><body>
<h1>会谈报告</h1>
<p>会谈 <code>{e(m['session_id'])}</code> ｜ 参与者 {e(m['participant_id'])} ｜ 条件：{e(COND_ZH.get(m['condition'], m['condition']))}
｜ 地域：{e(m['region_style'])} ｜ 时长 {e(m['session_minutes'])} 分钟 ｜ 结束原因：{e(m['end_reason'])}</p>
<p class=note>细节由大语言模型自动抽取，请在 details.csv 中逐条人工核对后再用于统计。主动开口的定义见 backend/app/research/analysis.py。</p>
<h2>指标</h2><table>{rows}</table>
<h2>修改类型分布</h2><table><tr><th>指令</th><th>次数</th></tr>{ops or '<tr><td colspan=2>无</td></tr>'}</table>
<h2>故事</h2><table><tr><th>标题</th><th>锚点</th><th>字数</th></tr>{st or '<tr><td colspan=3>无</td></tr>'}</table>
<h2>讲述细节</h2>
{det('人名', details.persons)}{det('地名', details.places)}{det('时间', details.times)}{det('感官描述', details.sensory, True)}{det('事件', details.events)}
<h2>讲述逐段</h2><table><tr><th>开始</th><th>时长(s)</th><th>类型</th><th>内容</th></tr>{ts}</table>
</body></html>"""


def build_comparison(ctx) -> tuple[str, str]:
    rows = []
    for s in ctx.db.query("SELECT id FROM sessions ORDER BY started_at"):
        p = ctx.settings.data_dir / "reports" / s["id"] / "metrics.json"
        if p.exists():
            rows.append(json.loads(p.read_text("utf-8")))
    from .analysis import FLAT_KEYS
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(FLAT_KEYS)
    for r in rows:
        w.writerow([json.dumps(r.get(k), ensure_ascii=False) if isinstance(r.get(k), dict) else r.get(k) for k in FLAT_KEYS])
    csv_text = "﻿" + buf.getvalue()

    conds = ["audio_only", "generic", "personal"]
    def stat(vals):
        vals = [v for v in vals if isinstance(v, (int, float))]
        if not vals:
            return "—"
        sd = stdev(vals) if len(vals) > 1 else 0.0
        return f"{mean(vals):.1f} ± {sd:.1f}"
    head = "".join(f"<th>{COND_ZH[c]}<br><span class=note>n={sum(1 for r in rows if r['condition'] == c)}</span></th>" for c in conds)
    body = ""
    for k, lbl in LABELS:
        cells = "".join(f"<td class=num>{stat([r.get(k) for r in rows if r['condition'] == c])}</td>" for c in conds)
        body += f"<tr><td>{e(lbl)}</td>{cells}</tr>"
    # 同一参与者跨条件的配对比较（被试内设计时有用）
    by_p: dict[str, dict[str, list]] = {}
    for r in rows:
        by_p.setdefault(r["participant_id"], {}).setdefault(r["condition"], []).append(r)
    prow = ""
    for pid, d in sorted(by_p.items()):
        cells = ""
        for c in conds:
            vs = d.get(c, [])
            cells += (f"<td class=num>{mean(v['elder_speech_s'] for v in vs):.0f} s / {mean(v['details_total'] for v in vs):.0f}</td>"
                      if vs else "<td class=num>—</td>")
        prow += f"<tr><td>{e(pid)}</td>{cells}</tr>"
    sess = "".join(f"<tr><td><a href='/api/sessions/{e(r['session_id'])}/report'>{e(r['session_id'])}</a></td><td>{e(r['participant_id'])}</td>"
                   f"<td>{e(COND_ZH.get(r['condition']))}</td><td class=num>{e(r['elder_speech_s'])}</td>"
                   f"<td class=num>{e(r['elder_turns'])}</td><td class=num>{e(r['elder_initiated_turns'])}</td>"
                   f"<td class=num>{e(r['details_total'])}</td><td class=num>{e(r['edits_executed'])}</td>"
                   f"<td class=num>{e(r['stories'])}</td></tr>" for r in rows)
    html_text = f"""<!doctype html><html lang=zh-CN><head><meta charset=utf-8><title>跨条件对比</title>{STYLE}</head><body>
<h1>跨会谈 / 跨条件对比</h1>
<p class=note>数值为 平均值 ± 标准差。共 {len(rows)} 场已分析的会谈。<a href="/api/reports/compare?fmt=csv">下载 CSV</a>（可导入 SPSS / R / Excel 做统计检验）。
样本量很小时请谨慎解读，建议同时报告每位参与者的数据。</p>
<h2>按条件汇总</h2><table><tr><th>指标</th>{head}</tr>{body}</table>
<h2>按参与者（讲述时长 / 细节总数）</h2><table><tr><th>参与者</th>{''.join(f'<th>{COND_ZH[c]}</th>' for c in conds)}</tr>{prow}</table>
<h2>全部会谈</h2><table><tr><th>会谈</th><th>参与者</th><th>条件</th><th>讲述(s)</th><th>段数</th><th>主动开口</th><th>细节</th><th>修改</th><th>故事</th></tr>{sess}</table>
</body></html>"""
    return html_text, csv_text
